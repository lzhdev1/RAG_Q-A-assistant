# 决定性测试：build_rag_components() 的 ensemble 检索 + 去重后，目标 chunk 是否真的在 candidates 里
import os, sys
sys.path.insert(0, "/app")
os.chdir("/app")
from dotenv import load_dotenv
load_dotenv()

from qa_chain import build_rag_components
from vector_store import normalize_text

QUERY = "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？"

llm, reranker_retriever = build_rag_components()
inner = reranker_retriever.inner_retriever

# 手动复刻 RerankerRetriever 的去重逻辑，看 candidates 内容
seen_hashes = set()
candidates = []
for q in [QUERY]:
    docs = inner.invoke(q)
    for doc in docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            candidates.append(doc)

print(f"candidates 数量: {len(candidates)}")
target_pos = None
for i, d in enumerate(candidates):
    t = normalize_text(d.page_content).lower()
    is_target = "two side effects" in t
    if is_target:
        target_pos = i
    print(f"  #{i+1} 目标={is_target} {d.page_content[:60]!r}")

print(f"\n目标在 candidates 位置: {target_pos}")
if target_pos is None:
    print(">>> 结论：目标 chunk 根本不在 candidates 里（召回层就丢了）")
else:
    print(">>> 结论：目标在 candidates 里，但 Reranker 排序后掉出 top3（排序层问题）")
    # 直接调 Reranker 看它怎么排
    ranked = reranker_retriever.invoke(QUERY)
    print("Rerank top3:")
    for i, d in enumerate(ranked):
        t = normalize_text(d.page_content).lower()
        print(f"  top{i+1} 目标={'two side effects' in t} {d.page_content[:60]!r}")
