# 临时调试：Q19 目标 chunk 的向量库定位
from qa_chain import _load_and_chunk_docs, build_rag_components
from vector_store import get_vectorstore, normalize_text, PARSER_VERSION, MANIFEST_PATH
import json, os

print("PARSER_VERSION =", PARSER_VERSION)
print("manifest version =", json.load(open(MANIFEST_PATH, encoding="utf-8")).get("version"))

# 1. 在 v3 chunk 里找到含 two side effects 的 chunk
chunks = _load_and_chunk_docs("docs")
print(f"\nv3 总 chunk 数: {len(chunks)}")
targets = []
for i, c in enumerate(chunks):
    t = normalize_text(c.page_content).lower()
    if "two side effects" in t:
        targets.append((i, c.page_content))
        print(f"  ★ chunk#{i} 含 'two side effects': {c.page_content[:300]!r}")
print(f"  命中目标 chunk: {len(targets)} 个")

# 2. 用向量库查询，看目标 chunk 排第几
vs = get_vectorstore()
q = "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？"
docs = vs.similarity_search_with_score(q, k=20)
print(f"\n向量 top20 相似度分布：")
for i, (d, s) in enumerate(docs):
    t = normalize_text(d.page_content).lower()
    hit = "two side effects" in t or "extra computational" in t
    mark = " <<<TARGET" if hit else ""
    print(f"  #{i+1:>2} score={s:.4f} {d.page_content[:60]!r}{mark}")

# 3. 如果目标在 top20 之外，看它的实际相似度（通过直接 embedding 对比太贵，用 query 逐个 chunk 算不现实）
# 先看目标 chunk 是否在 top20 里
in_top20 = any("two side effects" in normalize_text(d.page_content).lower() for d, s in docs)
print(f"\n目标 chunk 是否在向量 top20: {in_top20}")
