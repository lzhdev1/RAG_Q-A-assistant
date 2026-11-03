# 临时验证：Q19/Q18/Q23 的完整链路检索，看相关 chunk 排到第几
from qa_chain import build_rag_components
from vector_store import normalize_text

llm, retriever = build_rag_components()

print("=" * 70)
print("Q19: 时间特征副作用")
docs = retriever.invoke("在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？")
for i, d in enumerate(docs):
    t = normalize_text(d.page_content)
    mark = "★★★" if "two side effects" in t.lower() else ""
    print(f"  top{i+1}: {t[:70]!r} {mark}")

print()
print("完整候选链路检查：Ensemble 检索 top-10 是否含 two side effects chunk")
from qa_chain import _load_and_chunk_docs
from langchain_community.retrievers import BM25Retriever
from vector_store import get_vectorstore
bm = BM25Retriever.from_documents(_load_and_chunk_docs("docs"), k=10)
vs = get_vectorstore()
bm_docs = bm.invoke("时间特征会带来哪两个副作用")
vec_docs = vs.similarity_search("时间特征会带来哪两个副作用", k=10)
print(f"  BM25 top10 含 two side effects: {sum(1 for d in bm_docs if 'two side effects' in normalize_text(d.page_content).lower())}")
print(f"  向量 top10 含 two side effects: {sum(1 for d in vec_docs if 'two side effects' in normalize_text(d.page_content).lower())}")

print()
print("=" * 70)
print("Q18: 弱监督定义")
docs18 = retriever.invoke("弱监督学习（weakly supervised learning）在视频行人重识别中是如何定义的？")
for i, d in enumerate(docs18):
    t = normalize_text(d.page_content)
    mark = "★★★" if "regardless of the dedicated annotation" in t.lower() or "identifying the person" in t.lower() else ""
    print(f"  top{i+1}: {t[:70]!r} {mark}")

print()
print("=" * 70)
print("Q23: PAN 训练三组件")
docs23 = retriever.invoke("PAN的训练过程由哪三个组件构成？")
for i, d in enumerate(docs23):
    t = normalize_text(d.page_content)
    mark = "★★★" if "training process of PAN" in t.lower() or "3) another" in t.lower() else ""
    print(f"  top{i+1}: {t[:70]!r} {mark}")
