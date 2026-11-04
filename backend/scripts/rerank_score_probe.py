"""
Rerank 分数分布探针（动态阈值决策的数据采集）

目标：对 30 题评估集，绕过 top_k=3 截断，调用 DashScope Rerank 全量打分
（top_n = 候选数），采集每个 query 的真实相关性分数分布，用于判断：
  1. 分数形态是"悬崖状"（可定绝对阈值）还是"平缓下降"（需相对阈值）
  2. 悬崖位置（knee point）在哪个名次附近

用法（容器内 /app 目录，依赖已在镜像）：
  docker exec -w /app rag-app python scripts/rerank_score_probe.py
输出：/app/rerank_scores_probe.json（含每题全量分数 + 来源）
"""
import json
import os
import sys

# 确保 /app 在 sys.path（脚本位于 /app/scripts/ 下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import httpx

load_dotenv()


def collect():
    from qa_chain import build_rag_components, _rewrite_query

    llm, rret = build_rag_components()
    inner = rret.inner_retriever
    model = os.getenv("RERANK_MODEL", rret.model)
    api_key = os.getenv("OPENAI_API_KEY")

    with open("test_dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)

    out = []
    for i, item in enumerate(dataset):
        q = item["question"]
        # 与线上完全一致：查询改写 → 三路混合检索 → 去重合并
        queries = _rewrite_query(llm, q)
        seen, candidates = set(), []
        for qq in queries:
            for doc in inner.invoke(qq):
                h = hash(doc.page_content[:200])
                if h not in seen:
                    seen.add(h)
                    candidates.append(doc)

        if not candidates:
            print(f"[{i+1}/30] 无候选")
            out.append({"question": q, "n": 0, "scores": []})
            continue

        # 全量 rerank 打分（不截断）
        resp = httpx.post(
            "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "query": q,  # 与线上一致：原始 query 做 rerank
                "documents": [d.page_content for d in candidates],
                "top_n": len(candidates),
                "task": "Given a web search query, retrieve relevant passages that answer the query.",
            },
            timeout=30.0,
        )
        if resp.status_code != 200:
            print(f"[{i+1}/30] rerank 失败 {resp.status_code}: {resp.text[:150]}")
            continue

        ranked = resp.json()["results"]  # Cohere 兼容：按分数降序
        scored = []
        for r in ranked:
            doc = candidates[r["index"]]
            scored.append({
                "score": float(r["relevance_score"]),
                "source": str(doc.metadata.get("source", "")),
                "page": str(doc.metadata.get("page", "")),
                "head": doc.page_content[:60],
            })
        out.append({"question": q, "n": len(candidates), "scores": scored})
        s0, s1 = scored[0]["score"], scored[1]["score"] if len(scored) > 1 else None
        print(f"[{i+1}/30] n={len(candidates):>2} top1={s0:.3f} top2={s1 if s1 is None else round(s1,3)} min={scored[-1]['score']:.3f}")

    with open("rerank_scores_probe.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n共 {len(out)} 题，已保存 /app/rerank_scores_probe.json")


def summarize():
    """分布统计：分数形态 + 悬崖检测"""
    with open("rerank_scores_probe.json", encoding="utf-8") as f:
        data = json.load(f)

    n_ok = [o for o in data if o["n"] > 0]
    print(f"\n=== 汇总（{len(n_ok)} 题有候选）===")

    # 1. 候选数分布
    ns = [o["n"] for o in n_ok]
    print(f"候选数: min={min(ns)} max={max(ns)} 均值={sum(ns)/len(ns):.1f}")

    # 2. 全局分数直方图
    all_scores = [s["score"] for o in n_ok for s in o["scores"]]
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist = [0] * (len(bins) - 1)
    for s in all_scores:
        for bi in range(len(bins) - 1):
            if bins[bi] <= s < bins[bi + 1] or (s == 1.0 and bi == len(bins) - 2):
                hist[bi] += 1
                break
    print("\n全局分数直方图 (总 {} 条):".format(len(all_scores)))
    for bi in range(len(bins) - 1):
        bar = "#" * hist[bi]
        print(f"  [{bins[bi]:.1f}-{bins[bi+1]:.1f}) {hist[bi]:>4} {bar}")

    # 3. 每题的悬崖检测：排序后相邻最大 gap
    print("\n悬崖检测（每题的分数排序与最大 gap 位置）:")
    for o in n_ok:
        sc = [s["score"] for s in o["scores"]]
        if len(sc) < 2:
            continue
        gaps = [(sc[j] - sc[j + 1], j + 1) for j in range(len(sc) - 1)]  # (gap, 第j+1名后断开)
        gmax, cut = max(gaps)
        tail = " | ".join(f"{s:.2f}" for s in sc[:6])
        print(f"  n={o['n']:>2} 最大gap={gmax:.2f}@第{cut}名 | 分数前6: {tail}")


if __name__ == "__main__":
    collect()
    summarize()
