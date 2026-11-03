# 对照验证：similarity_search vs similarity_search_with_score 对 Q19 的结果
from vector_store import get_vectorstore, normalize_text

vs = get_vectorstore()
q = "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？"

print("=== similarity_search(k=10) ===")
docs = vs.similarity_search(q, k=10)
for i, d in enumerate(docs):
    t = normalize_text(d.page_content).lower()
    hits = [kw for kw in ["two side effects", "extra computational", "disturb the system"] if kw in t]
    print(f"  #{i+1} hits={hits} {d.page_content[:70]!r}")

print("\n=== similarity_search_with_score(k=10) ===")
docs2 = vs.similarity_search_with_score(q, k=10)
for i, (d, s) in enumerate(docs2):
    t = normalize_text(d.page_content).lower()
    hits = [kw for kw in ["two side effects", "extra computational", "disturb the system"] if kw in t]
    print(f"  #{i+1} score={s:.4f} hits={hits} {d.page_content[:70]!r}")
