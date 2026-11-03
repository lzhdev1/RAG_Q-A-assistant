# 对比 similarity_search 与 similarity_search_with_score 对同一 query 的结果
import os, sys
sys.path.insert(0, "/app")
os.chdir("/app")
from dotenv import load_dotenv
load_dotenv()

from vector_store import get_vectorstore, normalize_text

QUERY = "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？"

def is_target(t):
    return "two side effects" in normalize_text(t).lower()

vs = get_vectorstore()

# 方式1: similarity_search(k=10) —— 与生产链路一致
print("=== similarity_search(k=10) ===")
docs = vs.similarity_search(QUERY, k=10)
for i, d in enumerate(docs):
    mark = "<<<TARGET" if is_target(d.page_content) else ""
    print(f"  #{i+1} {mark} {d.page_content[:60]!r}")

# 方式2: similarity_search_with_score(k=20)
print("\n=== similarity_search_with_score(k=20) ===")
docs2 = vs.similarity_search_with_score(QUERY, k=20)
for i, (d, s) in enumerate(docs2):
    mark = "<<<TARGET" if is_target(d.page_content) else ""
    print(f"  #{i+1} score={s:.4f} {mark} {d.page_content[:60]!r}")

# 方式3: 手动 embedding + collection.query 看原始向量距离
print("\n=== 手动 query embedding + collection.query(k=20) ===")
from vector_store import _get_embeddings
emb = _get_embeddings()
qvec = emb.embed_query(QUERY)
res = vs._collection.query(query_embeddings=[qvec], n_results=20, include=["documents", "distances"])
for i, (d, dist) in enumerate(zip(res["documents"][0], res["distances"][0])):
    mark = "<<<TARGET" if is_target(d) else ""
    print(f"  #{i+1} dist={dist:.4f} {mark} {d[:60]!r}")
