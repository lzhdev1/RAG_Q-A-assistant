# 测试 Reranker API 行为：目标 chunk + 无关 chunk 一起提交，检查返回 index/score 映射
import os, sys, json, httpx
sys.path.insert(0, "/app")
os.chdir("/app")
from dotenv import load_dotenv
load_dotenv()

QUERY = "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？"

# 目标 chunk（从库里取出）+ 若干明显无关的 chunk
from vector_store import get_vectorstore, normalize_text
vs = get_vectorstore()
col = vs._collection
res = col.get(where_document={"$contains": "two side effects"}, limit=1)
target_doc = (res.get("documents") or [None])[0]
print(f"目标chunk长度: {len(target_doc) if target_doc else 0}")

# 取 4 个明显无关的（页眉/引用）
res2 = col.get(limit=10)
all_docs = res2.get("documents") or []
docs = [target_doc]
for d in all_docs:
    if "Page " in d and "Saad et al." in d:
        docs.append(d)  # 页眉
    if len(docs) >= 2: break
for d in all_docs:
    if "arxiv" in d or "http" in d:
        docs.append(d)  # 引用
    if len(docs) >= 3: break
for d in all_docs:
    if "ZHENG" in d or "IEEE" in d:
        docs.append(d)
    if len(docs) >= 4: break
docs = docs[:4]
print(f"提交文档数: {len(docs)}")
for i, d in enumerate(docs):
    print(f"  doc[{i}] 目标={('two side effects' in normalize_text(d).lower())} 头={d[:50]!r}")

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("RERANK_MODEL", "qwen3-rerank")
resp = httpx.post(
    "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={
        "model": model,
        "query": QUERY,
        "documents": docs,
        "top_n": 4,
        "task": "Given a web search query, retrieve relevant passages that answer the query.",
    },
    timeout=30.0,
)
print(f"\nHTTP {resp.status_code}")
print(json.dumps(resp.json(), ensure_ascii=False, indent=2)[:2000])
