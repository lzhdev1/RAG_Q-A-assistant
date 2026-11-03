# -*- coding: utf-8 -*-
"""
四探针失败归因脚本（failure analysis）v2

对测试集中 RAGAS 分数偏低的题目逐题定位问题层：

  探针①  原文回查    —— ground_truth/question 中的"锚点词"在 docs/*.pdf 原文里能否搜到
  探针②  召回检查    —— BM25 单独 top-10、向量单独 top-10 里有没有相关 chunk
  探针③  排序检查    —— 完整链路（BM25+向量 Ensemble → Rerank top-3）里相关 chunk 是否保留
  探针④  Golden Context —— 把原文相关 chunk 直接塞给 LLM 生成，judge LLM 判定覆盖度

锚点提取（解决中文 ground_truth 与英文原文的鸿沟）：
  - 括号内英文：ground_truth 里的（closed-world）（Heterogeneous Data）等 → 原文术语
  - 全大写缩写：CMC / mAP / AGW / PAN / STN / Res2 / FC 等
  - 英文短语：Attention Generalized mean pooling 等连续词组
  - 停用词过滤：data / feature / learning / model 等通用词剔除

用法（容器内）:
  python failure_analysis.py
输出:
  app/failure_analysis_results.json + 控制台归因摘要
"""

import glob
import json
import os
import re
import sys
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "docs"
DATASET = "test_dataset.json"
EVAL = "eval_results_v2.json"
OUTPUT = "failure_analysis_results.json"

AR_THRESHOLD = 0.60
CP_THRESHOLD = 0.60

# 通用停用词：这些词出现在每段英文里，不能当锚点
STOPWORDS = {
    "data", "feature", "features", "learning", "loss", "network", "model",
    "system", "method", "approach", "image", "video", "person", "identification",
    "recognition", "based", "using", "with", "for", "and", "the", "of", "in",
    "paper", "methods", "approaches", "performance", "results", "two", "three",
    "four", "five", "six", "one", "new", "main", "major", "what", "which", "how",
    "training", "trained", "process", "part", "parts", "problems", "challenge",
}


def normalize_text(text):
    import unicodedata
    return unicodedata.normalize("NFKC", text or "")


def ng4(text):
    t = re.sub(r"\s+", "", normalize_text(text).lower())
    return {t[i:i + 4] for i in range(len(t) - 3)}


def dice4(a, b):
    sa, sb = ng4(a), ng4(b)
    if not sa or not sb:
        return 0.0
    return 2.0 * len(sa & sb) / (len(sa) + len(sb))


def extract_anchors(question, ground_truth):
    """提取用于原文回查与相关性判定的锚点词集合"""
    text = f"{ground_truth} {question}"
    anchors = set()

    # 1) 括号内英文：中文答案里（English Term）的原文术语
    for m in re.findall(r"[（(]\s*([A-Za-z][A-Za-z0-9\-\s/]{2,60}?)\s*[)）]", text):
        for part in re.split(r"[,\s]+", m):
            p = part.strip()
            if p and p.lower() not in STOPWORDS and len(p) >= 2:
                anchors.add(p.lower())

    # 2) 全大写缩写或含数字的代号：CMC / mAP / AGW / PAN / STN / Res2 / FC
    for m in re.findall(r"\b[A-Z][A-Za-z0-9\-]{1,10}\b", text):
        low = m.lower()
        if low not in STOPWORDS and not low.isdigit() and len(low) >= 2:
            # 纯小写单词（如 feature）排除，只留含大写的专名/缩写
            if not (len(m) > 2 and m.isalpha() and m[1:].islower() and low not in ("cmc", "map")):
                anchors.add(low)

    # 3) 英文连续短语（≥2 词且总长≥8）：Attention Generalized mean pooling
    for m in re.findall(r"\b[A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){1,4}\b", text):
        low = m.lower()
        words = low.split()
        meaningful = [w for w in words if w not in STOPWORDS]
        if len(meaningful) >= 2 or (len(meaningful) == 1 and len(meaningful[0]) >= 5):
            anchors.add(low)

    return sorted(a for a in anchors if a)


def find_all(text, kw):
    low = text.lower()
    k = kw.lower()
    idx, start = [], 0
    while True:
        i = low.find(k, start)
        if i < 0:
            break
        idx.append(i)
        start = i + len(k)
    return idx


# ── 探针①：原文回查 ───────────────────────────────────────────────

def probe1(texts, anchors):
    """在全部原文（归一化后）搜索锚点词。返回 (found, evidence)"""
    if not anchors:
        return None, {}, anchors  # 无锚点 → 无法自动验证

    found = False
    evidence = {}
    for fname, text in texts.items():
        hits_total = 0
        per_kw = {}
        for kw in anchors:
            n = len(find_all(text, kw))
            per_kw[kw] = n
            hits_total += n
            if n > 0:
                found = True
        evidence[fname] = {"hits": hits_total, "per_kw": per_kw}
    return found, evidence, anchors


# ── 探针②：召回检查（BM25 / 向量分别） ───────────────────────────

def _score_docs(docs, anchors):
    best = {"dice": 0.0, "kw_hit": 0, "content": "", "idx": -1}
    rows = []
    for i, d in enumerate(docs):
        c = d.page_content
        d4 = dice4(c, "")
        # 用锚点命中数判定相关性（更可靠）
        low = normalize_text(c).lower()
        kh = sum(1 for kw in anchors if kw in low)
        # 更新 best：优先锚点命中数，其次 dice
        if (kh, d4) > (best["kw_hit"], best["dice"]):
            best = {"dice": round(d4, 4), "kw_hit": kh,
                    "content": c[:200], "idx": i}
        rows.append({"idx": i, "dice": round(d4, 4), "kw_hit": kh,
                     "head": c[:50].replace("\n", " ")})
    return {"top": rows, "best": best}


def probe2(bm25_retriever, vectorstore, q, anchors, top_n=10):
    result = {}
    try:
        bm_docs = bm25_retriever.invoke(q)
    except Exception as e:
        return {"error": str(e)}
    result["bm25"] = _score_docs(bm_docs, anchors)
    try:
        vec_docs = vectorstore.similarity_search(q, k=top_n)
    except Exception as e:
        result["vector"] = {"error": str(e)}
        return result
    result["vector"] = _score_docs(vec_docs, anchors)
    return result


# ── 探针③：排序检查 ──────────────────────────────────────────────

def probe3(reranker_retriever, q, anchors):
    docs = reranker_retriever.invoke(q)
    return _score_docs(docs, anchors), [d.page_content for d in docs]


# ── 探针④：Golden Context ────────────────────────────────────────

def probe4(llm, judge_llm, chunk_text, q, gt):
    prompt = (
        "你是专业学术问答助手。根据以下参考资料回答问题。\n\n"
        "【参考资料】\n" + chunk_text[:1500] + "\n\n"
        "【问题】\n" + q + "\n\n"
        "【回答】"
    )
    answer = llm.invoke(prompt).content
    judge_prompt = (
        "请判断模型回答是否完整覆盖了参考答案的核心要点。\n\n"
        "【参考答案】\n" + gt + "\n\n"
        "【模型回答】\n" + answer + "\n\n"
        "只输出 JSON：{\"covered\": true/false, \"reason\": \"一句话说明\"}"
    )
    judge_out = judge_llm.invoke(judge_prompt).content
    try:
        m = re.search(r"\{.*\}", judge_out, re.DOTALL)
        verdict = json.loads(m.group())
    except Exception:
        verdict = {"covered": None, "reason": judge_out[:100]}
    return {"answer": answer[:200], "verdict": verdict}


# ── 主流程 ─────────────────────────────────────────────────────────

def main():
    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(EVAL, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    per_sample = eval_data["per_sample"]

    print(f"测试集 {len(dataset)} 题，评估 {len(per_sample)} 题")

    print("加载 PDF 原文...")
    from langchain_community.document_loaders import PyPDFLoader
    texts = {}
    for pdf_path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf"))):
        pages = PyPDFLoader(pdf_path).load()
        texts[os.path.basename(pdf_path)] = "\n".join(
            normalize_text(p.page_content) for p in pages
        )
    print(f"已加载 {len(texts)} 篇 PDF")

    print("构建检索组件...")
    from qa_chain import build_rag_components, _load_and_chunk_docs
    from langchain_community.retrievers import BM25Retriever
    from vector_store import get_vectorstore
    from langchain_openai import ChatOpenAI

    llm, reranker_retriever = build_rag_components()
    bm25 = BM25Retriever.from_documents(_load_and_chunk_docs("docs"), k=10)
    vs = get_vectorstore()
    judge_llm = ChatOpenAI(
        model=os.getenv("JUDGE_LLM_MODEL", os.getenv("LLM_MODEL")),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
        extra_body={"enable_thinking": False},
    )

    results = []
    summary_rows = []
    for i, item in enumerate(dataset):
        q = item["question"]
        gt = item["ground_truth"]
        scores = per_sample[i]["scores"] if i < len(per_sample) else {}
        ar = scores.get("answer_relevancy") or 0
        cp = scores.get("context_precision") or 0
        fa = scores.get("faithfulness") or 0

        is_fail = (ar < AR_THRESHOLD) or (cp < CP_THRESHOLD)
        rec = {
            "idx": i + 1,
            "question": q,
            "scores": {"AR": round(ar, 4), "CP": round(cp, 4), "F": round(fa, 4)},
            "probes": {},
        }

        anchors = extract_anchors(q, gt)
        rec["probes"]["anchors"] = anchors

        found, ev, _ = probe1(texts, anchors)
        rec["probes"]["p1_original"] = found
        rec["probes"]["p1_hits_per_doc"] = {
            k: v["hits"] for k, v in ev.items()
        } if ev else {}

        if not is_fail:
            rec["label"] = "OK"
            results.append(rec)
            summary_rows.append((i + 1, q[:28], ar, cp, "OK"))
            continue

        print(f"\n[{i+1}] 深挖: {q[:45]}  (AR={ar:.2f} CP={cp:.2f} F={fa:.2f})")
        print(f"    锚点({len(anchors)}): {anchors[:10]}")

        # 探针②
        p2 = probe2(bm25, vs, q, anchors)
        rec["probes"]["p2"] = {
            "bm25_best": p2.get("bm25", {}).get("best", {}),
            "vector_best": p2.get("vector", {}).get("best", {}),
        }
        bm_best = p2.get("bm25", {}).get("best", {})
        vec_best = p2.get("vector", {}).get("best", {})
        print(f"    BM25 best kw_hit={bm_best.get('kw_hit')} dice={bm_best.get('dice')}")

        # 探针③
        p3, docs3 = probe3(reranker_retriever, q, anchors)
        rec["probes"]["p3"] = {"best": p3["best"], "top3": p3["top"]}
        print(f"    Rerank top3 best kw_hit={p3['best'].get('kw_hit')} dice={p3['best'].get('dice')}")

        # 探针④：原文有答案且 top3 未保留相关 → 检查生成层
        p4 = None
        if found and p3["best"]["kw_hit"] == 0:
            best_chunk = None
            best_score = 0
            for pdf_path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf"))):
                for page in PyPDFLoader(pdf_path).load():
                    c = normalize_text(page.page_content)
                    kh = sum(1 for kw in anchors if kw in c.lower())
                    if kh > best_score:
                        best_score = kh
                        best_chunk = c
            if best_chunk:
                print("    原文有答案但 top3 未命中 → Golden Context 检查生成层...")
                p4 = probe4(llm, judge_llm, best_chunk, q, gt)
                rec["probes"]["p4"] = p4
                print(f"    judge covered={p4['verdict']}")
        rec["probes"]["p4_ran"] = p4 is not None

        label = _classify(found, p2, p3, p4)
        rec["label"] = label
        print(f"    → 归因: {label}")

        summary_rows.append((i + 1, q[:28], round(ar, 3), round(cp, 3), label))
        results.append(rec)  # 修复：失败题也要写入 JSON

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("归因汇总（失败题）")
    print("=" * 80)
    labels = Counter()
    for row in summary_rows:
        idx, q, ar, cp, label = row
        if label != "OK":
            labels[label] += 1
            print(f"  Q{idx:>2} AR={ar:.3f} CP={cp:.3f}  [{label}]  {q}")
    print("\n标签统计:")
    for k, v in labels.most_common():
        print(f"  {k}: {v}")
    print(f"\n详细结果已保存至 {OUTPUT}")


def _classify(found, p2, p3, p4):
    bm_best = p2.get("bm25", {}).get("best", {})
    vec_best = p2.get("vector", {}).get("best", {})
    p3_best = p3["best"]

    if found is None:
        return "需人工复核(无锚点)"
    if not found:
        return "测试集/解析问题(原文搜不到)"

    bm_hit = bm_best.get("kw_hit", 0) >= 1
    vec_hit = vec_best.get("kw_hit", 0) >= 1
    top3_hit = p3_best.get("kw_hit", 0) >= 1

    if not bm_hit and not vec_hit:
        return "召回问题(两侧都检索不到)"
    if not bm_hit:
        return "召回问题(BM25侧)"
    if not vec_hit:
        return "召回问题(向量侧)"
    if not top3_hit:
        return "排序问题(Rerank未保留)"
    if p4 and p4.get("verdict", {}).get("covered") is False:
        return "生成问题(Golden Context未答对)"
    return "边界案例(需人工复核)"


if __name__ == "__main__":
    main()
