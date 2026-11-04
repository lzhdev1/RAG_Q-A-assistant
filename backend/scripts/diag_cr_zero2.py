"""
CR=0 归因诊断第二轮：
- Q5/Q15：打印全部候选的分数+来源页+头部，定位"答案细节页"的分数（判断:检索漏了 vs 阈值砍了）
- Q28：打印保留 chunk 全文，核对 reference 的"4096 维"是否在 context（判断 judge 严苛 vs 真漏）
"""
import json
import os
import sys

sys.path.insert(0, "/app")
from dotenv import load_dotenv
import httpx

load_dotenv()

with open("/app/test_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

from qa_chain import build_rag_components, _rewrite_query

llm, rret = build_rag_components()
inner = rret.inner_retriever
model = os.getenv("RERANK_MODEL", rret.model)
api_key = os.getenv("OPENAI_API_KEY")


def run(ti, print_all=False, full_chunks=False):
    item = dataset[ti - 1]
    q = item["question"]
    ref = item.get("ground_truth") or item.get("reference") or item.get("answer")
    print("=" * 90)
    print(f"Q{ti}: {q}")
    print(f"[reference] {ref}")

    queries = _rewrite_query(llm, q)
    seen, candidates = set(), []
    for qq in queries:
        for doc in inner.invoke(qq):
            h = hash(doc.page_content[:200])
            if h not in seen:
                seen.add(h)
                candidates.append(doc)

    resp = httpx.post(
        "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "query": q,
            "documents": [d.page_content for d in candidates],
            "top_n": min(len(candidates), 100),
            "task": "Given a web search query, retrieve relevant passages that answer the query.",
        },
        timeout=30.0,
    )
    ranked = [(float(r["relevance_score"]), candidates[r["index"]]) for r in resp.json()["results"]]
    threshold = ranked[0][0] * 0.6
    kept = [(s, d) for s, d in ranked if s >= threshold][:6]
    print(f"\n候选 {len(ranked)} 条 | 阈值(α=0.6) {threshold:.3f} | 保留 {len(kept)} 条 | 阈值(α=0.5) {ranked[0][0]*0.5:.3f}")

    if print_all:
        print("--- 全部候选按分数降序（含被砍部分） ---")
        for j, (s, d) in enumerate(ranked, 1):
            src = d.metadata.get("source", "")
            pg = d.metadata.get("page", "")
            mark = "KEEP" if s >= threshold and j <= 6 else "cut "
            print(f"  [{j:>2}] {mark} {s:.3f} {src} p{pg} | {d.page_content[:70].replace(chr(10),' ')}")

    if full_chunks:
        print("--- 保留 chunk 全文 ---")
        for j, (s, d) in enumerate(kept, 1):
            print(f"[{j}] score={s:.3f} {d.metadata.get('source','')} p{d.metadata.get('page','')}")
            print(d.page_content)
            print()


run(5, print_all=True)
run(15, print_all=True)
run(28, full_chunks=True)
