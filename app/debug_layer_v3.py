# 临时调试：Q19/Q29 在 v3 链路中每层检索的命中情况
from qa_chain import _load_and_chunk_docs, build_rag_components
from langchain_community.retrievers import BM25Retriever
from vector_store import get_vectorstore, normalize_text
import os

def hit(doc, kws):
    t = normalize_text(doc.page_content).lower()
    return sum(1 for kw in kws if kw in t)

Q19_KWS = ["two side effects", "extra computational", "disturb the system"]
Q29_KWS = ["market-1501", "cuhk03", "dukemtmc"]

print("加载组件...")
llm, reranker_retriever = build_rag_components()
bm = BM25Retriever.from_documents(_load_and_chunk_docs("docs"), k=10)
vs = get_vectorstore()

for name, q, kws in [
    ("Q19", "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？", Q19_KWS),
    ("Q29", "论文在哪些大规模行人重识别数据集上验证了PAN的性能？", Q29_KWS),
]:
    print(f"\n===== {name} =====")
    bm_docs = bm.invoke(q)
    vec_docs = vs.similarity_search(q, k=10)
    print(f"  BM25 top10: 命中={[hit(d, kws) for d in bm_docs]}")
    print(f"  向量 top10: 命中={[hit(d, kws) for d in vec_docs]}")

    # Ensemble 融合后的候选（模拟：直接用 reranker 的 inner_retriever）
    inner = reranker_retriever.inner_retriever
    ens_docs = inner.invoke(q)
    print(f"  Ensemble top10: 命中={[hit(d, kws) for d in ens_docs]}  总数={len(ens_docs)}")

    # Rerank 最终
    final = reranker_retriever.invoke(q)
    print(f"  Rerank top{len(final)}: 命中={[hit(d, kws) for d in final]}")
