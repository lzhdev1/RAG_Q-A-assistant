"""
Q5 漏检定位实验：
1. 答案 chunk 存在性：在 chroma 中定位含五方面枚举（Heterogeneous/Noisy Annotation 等）的 chunk
2. 现网模拟（单 query、无改写、三路各 top-10）：答案 chunk 是否进候选、rerank 排名
3. 对照实验：扩大窗口 top-20/30 vs 英文 query 直检，量化哪个方案能召回
"""
import json
import os
import sys

sys.path.insert(0, "/app")
from dotenv import load_dotenv
import httpx

load_dotenv()

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from vector_store import get_vectorstore
from qa_chain import HyDEVectorRetriever, build_rag_components, _load_and_chunk_docs

Q = "开放世界（open-world）行人重识别主要包含哪五个方面的挑战？"
Q_EN = "What are the five aspects of challenges in open-world person re-identification?"
SRC = "docs/Deep Learning for Person Re-identification A Survey and Outlook.pdf"

print(f"EMBEDDING_MODEL = {os.getenv('EMBEDDING_MODEL')}")
print(f"LLM_MODEL = {os.getenv('LLM_MODEL')}")

llm, _ = build_rag_components()
vs = get_vectorstore()

# ── 1. 定位答案 chunk ─────────────────────────────────────────────
col = vs._collection
all_docs = col.get(where={"source": SRC})
kw = ["Heterogeneous", "Raw Images", "Unavailable", "Noisy Annotation", "Open-set"]
print("\n=== 1. 答案 chunk 定位（综述论文中含五方面关键词的 chunk） ===")
answer_chunks = {}
for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
    hits = [w for w in kw if w.lower() in doc.lower()]
    if len(hits) >= 2:
        answer_chunks[meta["page"]] = (hits, doc)
        print(f"  p{meta['page']} hits={hits}")
        print(f"    {doc[:160].replace(chr(10),' ')}")
print(f"  命中≥2关键词的页: {sorted(answer_chunks.keys())}")

# 取最强命中 chunk 作为"答案 chunk"
ans_page = min(answer_chunks) if answer_chunks else None
ans_doc = answer_chunks[ans_page][1] if ans_page is not None else None
print(f"  → 取 p{ans_page} 作为答案 chunk（{len(ans_doc)} 字符）" if ans_doc else "  → 未找到答案 chunk（库中不存在？）")


def ensemble_k(k):
    bm25 = BM25Retriever.from_documents(_load_and_chunk_docs("docs"), k=k)
    vec = vs.as_retriever(search_kwargs={"k": k})
    hyde = HyDEVectorRetriever(vectorstore=vs, llm=llm, k=k, enabled=True)
    return EnsembleRetriever(retrievers=[bm25, vec, hyde], weights=[0.4, 0.3, 0.3])


def rerank(cands, query):
    resp = httpx.post(
        "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("RERANK_MODEL", "gte-rerank-hybrid"),
            "query": query,
            "documents": [d.page_content for d in cands],
            "top_n": min(len(cands), 100),
            "task": "Given a web search query, retrieve relevant passages that answer the query.",
        },
        timeout=30.0,
    )
    return [(float(r["relevance_score"]), cands[r["index"]]) for r in resp.json()["results"]]


def probe(label, query, k, do_rerank=True):
    """三路 top-k 检索 → (可选)rerank，报告答案 chunk 是否被召回及其位置"""
    seen, cands = set(), []
    for d in ensemble_k(k).invoke(query):
        h = hash(d.page_content[:200])
        if h not in seen:
            seen.add(h)
            cands.append(d)
    ans_hit = [i for i, d in enumerate(cands) if d.page_content[:200] == ans_doc[:200] or (ans_doc and d.page_content[:800] == ans_doc[:800])]
    ans_in_cands = ans_hit[0] if ans_hit else None
    msg = f"候选{len(cands)}条 | 答案chunk: {'在候选 #%d' % ans_in_cands if ans_in_cands is not None else '未进候选'}"
    if do_rerank and cands:
        ranked = rerank(cands, query)
        r_hit = next((j for j, (s, d) in enumerate(ranked) if d.page_content[:800] == (ans_doc or '')[:800]), None)
        msg += f" | rerank: {'#%d score=%.3f' % (r_hit, ranked[r_hit][0]) if r_hit is not None else '未进rerank结果'}"
    print(f"  [{label}] {msg}")
    return ans_in_cands


print("\n=== 2. 现网模拟（单query无改写） ===")
probe("top-10 (现网)", Q, 10)
probe("top-20", Q, 20)
probe("top-30", Q, 30)

print("\n=== 3. 英文 query 直检 ===")
probe("top-10 EN", Q_EN, 10)
probe("top-20 EN", Q_EN, 20)

print("\n=== 4. 改写变体（诊断脚本曾用）对照 ===")
from qa_chain import _rewrite_query
queries = _rewrite_query(llm, Q)
print(f"  改写变体: {queries}")
for i, qq in enumerate(queries[1:], 1):
    probe(f"变体{i} top-10", qq, 10)
