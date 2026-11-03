# Reranker 稳定性测试：同 query 同 documents 连续调用 5 次，看 top3 是否每次都一致
import os, sys, json, httpx
sys.path.insert(0, "/app")
os.chdir("/app")
from dotenv import load_dotenv
load_dotenv()

from qa_chain import build_rag_components
from vector_store import normalize_text

QUERY = "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？"

llm, reranker_retriever = build_rag_components()
inner = reranker_retriever.inner_retriever

# 构建固定 candidates
seen_hashes = set()
candidates = []
for q in [QUERY]:
    docs = inner.invoke(q)
    for doc in docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            candidates.append(doc)

print(f"candidates: {len(candidates)} 个")
target_idx = None
for i, d in enumerate(candidates):
    if "two side effects" in normalize_text(d.page_content).lower():
        target_idx = i
        print(f"目标在 candidates 位置: #{i+1}（传参顺序 {i}）")

# 直接调 Reranker API 5 次，固定 candidates 顺序
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("RERANK_MODEL", "qwen3-rerank")
documents = [d.page_content for d in candidates]

for run in range(5):
    resp = httpx.post(
        "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "query": QUERY,
            "documents": documents,
            "top_n": 3,
            "task": "Given a web search query, retrieve relevant passages that answer the query.",
        },
        timeout=30.0,
    )
    results = resp.json().get("results", [])
    order = [r["index"] for r in results]
    scores = [round(r["relevance_score"], 4) for r in results]
    target_hit = target_idx in order[:3] if target_idx is not None else None
    # 标记目标在 top3 中的位置
    mark = "目标命中!" if target_hit else "目标未进top3"
    print(f"Run{run+1}: order={order} scores={scores} -> {mark}")

print("\n结论: 若 order 多次不同 -> Reranker API 输出不稳定，Q19 是排序层随机性导致的偶发失败")
