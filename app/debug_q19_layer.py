# Q19 深度归因：目标 chunk 在各检索层的命中位置
# 验证目标 chunk 是否进入候选池，还是被 Reranker 排掉
import os, sys
sys.path.insert(0, "/app")
os.chdir("/app")

from vector_store import get_vectorstore, normalize_text, _load_manifest
from qa_chain import _load_and_chunk_docs
from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

QUERY = "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？"
TARGET = "two side effects"  # 目标 chunk 的锚点

def is_target(t):
    return "two side effects" in normalize_text(t).lower()

# 1. 确认目标 chunk 在库中的原始内容
vs = get_vectorstore()
col = vs._collection
res = col.get(where_document={"$contains": "two side effects"}, limit=3)
target_docs = res.get("documents") or []
target_ids = res.get("ids") or []
print(f"目标 chunk 在库: {len(target_docs)} 个, id={target_ids[:1]}")
if target_docs:
    print(f"  目标内容头: {target_docs[0][:100]!r}")

# 2. 向量检索 top-20 命中位置
vdocs = vs.similarity_search_with_score(QUERY, k=20)
print(f"\n=== 向量检索 top-20 ===")
for i, (d, s) in enumerate(vdocs):
    mark = "<<< TARGET" if is_target(d.page_content) else ""
    print(f"  #{i+1} score={s:.4f} {mark} {d.page_content[:60]!r}")

# 3. BM25 检索 top-20 命中位置
doc_chunks = _load_and_chunk_docs("docs")
bm25 = BM25Retriever.from_documents(doc_chunks, k=20)
bdocs = bm25.invoke(QUERY)
print(f"\n=== BM25 top-20 ===")
for i, d in enumerate(bdocs):
    mark = "<<< TARGET" if is_target(d.page_content) else ""
    print(f"  #{i+1} {mark} {d.page_content[:60]!r}")

# 4. Ensemble 融合后目标是否在
vretr = vs.as_retriever(search_kwargs={"k": 10})
bm25_10 = BM25Retriever.from_documents(doc_chunks, k=10)
ens = EnsembleRetriever(retrievers=[bm25_10, vretr], weights=[0.5, 0.5])
edocs = ens.invoke(QUERY)
print(f"\n=== Ensemble top-10（当前生产配置 k=10）===")
for i, d in enumerate(edocs):
    mark = "<<< TARGET" if is_target(d.page_content) else ""
    print(f"  #{i+1} {mark} {d.page_content[:60]!r}")

# 5. Ensemble 扩容 k=20 后目标是否在
vretr20 = vs.as_retriever(search_kwargs={"k": 20})
bm25_20 = BM25Retriever.from_documents(doc_chunks, k=20)
ens20 = EnsembleRetriever(retrievers=[bm25_20, vretr20], weights=[0.5, 0.5])
edocs20 = ens20.invoke(QUERY)
print(f"\n=== Ensemble top-20（扩容后）===")
for i, d in enumerate(edocs20):
    mark = "<<< TARGET" if is_target(d.page_content) else ""
    print(f"  #{i+1} {mark} {d.page_content[:60]!r}")
