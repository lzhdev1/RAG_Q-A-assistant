"""
Q5/Q15/Q28 三题 CR=0 归因诊断：
1. 打印 test_dataset.json 的 reference（ground_truth）
2. 走线上检索链路（改写→三路检索→去重→全量rerank→动态阈值 α=0.6）
3. 打印保留 chunk 的来源页与全文，人工比对 reference 声明是否在 context 中
用途：区分"真实召回缺口" vs "RAGAS judge 抖动"
"""
import json
import os
import sys

sys.path.insert(0, "/app")
from dotenv import load_dotenv
import httpx

load_dotenv()

TARGETS = [5, 15, 28]  # 1-based

with open("/app/test_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

from qa_chain import build_rag_components, _rewrite_query

llm, rret = build_rag_components()
inner = rret.inner_retriever
model = os.getenv("RERANK_MODEL", rret.model)
api_key = os.getenv("OPENAI_API_KEY")

for ti in TARGETS:
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
    print(f"\n候选 {len(ranked)} 条 → 阈值 {threshold:.3f} → 保留 {len(kept)} 条:")
    for j, (s, d) in enumerate(kept, 1):
        src = d.metadata.get("source", "")
        pg = d.metadata.get("page", "")
        print(f"  [{j}] score={s:.3f} {src} p{pg}")
        print(f"      {d.page_content[:250]}")
