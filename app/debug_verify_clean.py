# 干净重建后的最终验证
from vector_store import get_vectorstore, normalize_text

vs = get_vectorstore()
col = vs._collection
print("collection count:", col.count())

# 目标 chunk 存在性
res = col.get(where_document={"$contains": "two side effects"}, limit=5)
docs = res.get("documents") or []
print("含 two side effects 的 chunk:", len(docs))
if docs:
    print("  内容头:", repr(docs[0][:80]))

# Q19 完整链路检索
from qa_chain import build_rag_components
llm, retriever = build_rag_components()
docs = retriever.invoke("在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？")
print("\nQ19 Rerank top3:")
for i, d in enumerate(docs):
    t = normalize_text(d.page_content).lower()
    hit = "two side effects" in t or "extra computational" in t
    print(f"  top{i+1}: hits={hit} {d.page_content[:70]!r}")

# Q29 完整链路检索
docs = retriever.invoke("论文在哪些大规模行人重识别数据集上验证了PAN的性能？")
print("\nQ29 Rerank top3:")
for i, d in enumerate(docs):
    t = normalize_text(d.page_content).lower()
    hit = "market-1501" in t or "dukemtmc" in t or "cuhk03" in t
    print(f"  top{i+1}: hits={hit} {d.page_content[:70]!r}")
