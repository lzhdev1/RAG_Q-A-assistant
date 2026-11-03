# 定位候选不稳定源：同一进程内连续多次运行向量检索/BM25/Ensemble
import os, sys
sys.path.insert(0, "/app")
os.chdir("/app")
from dotenv import load_dotenv
load_dotenv()

from qa_chain import build_rag_components, _load_and_chunk_docs
from vector_store import get_vectorstore, normalize_text
from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

QUERY = "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？"

def is_target(t):
    return "two side effects" in normalize_text(t).lower()

llm, reranker_retriever = build_rag_components()

# 1. 向量检索稳定性（同一 vectorstore，多次查询）
vs = get_vectorstore()
print("=== 向量检索 top-10 连续 5 次（目标位置）===")
for run in range(5):
    vdocs = vs.similarity_search(QUERY, k=10)
    pos = None
    for i, d in enumerate(vdocs):
        if is_target(d.page_content):
            pos = i
            break
    print(f"  Run{run+1}: 目标在 top-10 位置 {pos}")

# 2. BM25 稳定性（同一 doc_chunks，多次检索）
print("\n=== BM25 top-10 连续 5 次（目标位置）===")
doc_chunks = _load_and_chunk_docs("docs")
bm25 = BM25Retriever.from_documents(doc_chunks, k=10)
for run in range(5):
    bdocs = bm25.invoke(QUERY)
    pos = None
    for i, d in enumerate(bdocs):
        if is_target(d.page_content):
            pos = i
            break
    print(f"  Run{run+1}: 目标在 top-10 位置 {pos}")

# 3. Ensemble 稳定性（同一组件，多次调用）
print("\n=== Ensemble top-10 连续 5 次（目标位置）===")
vretr = vs.as_retriever(search_kwargs={"k": 10})
bm25_10 = BM25Retriever.from_documents(doc_chunks, k=10)
ens = EnsembleRetriever(retrievers=[bm25_10, vretr], weights=[0.5, 0.5])
for run in range(5):
    edocs = ens.invoke(QUERY)
    pos = None
    for i, d in enumerate(edocs):
        if is_target(d.page_content):
            pos = i
            break
    print(f"  Run{run+1}: 目标在 top-10 位置 {pos}")

# 4. 关键检查：doc_chunks 中目标 chunk 是否稳定存在
print("\n=== doc_chunks 目标 chunk 检查 ===")
tgt = [d for d in doc_chunks if is_target(d.page_content)]
print(f"  doc_chunks 含目标: {len(tgt)} 个")
if tgt:
    print(f"  目标 chunk 元数据: {tgt[0].metadata}")
    print(f"  目标 chunk 前60: {tgt[0].page_content[:60]!r}")
