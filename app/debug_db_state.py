# 决定性验证：向量库内容、collection、目标 chunk 是否在库
from vector_store import get_vectorstore, normalize_text, MANIFEST_PATH
import json, os

print("=== manifest ===")
with open(MANIFEST_PATH, encoding="utf-8") as f:
    m = json.load(f)
print("version:", m.get("version"))
print("files:", {k: len(v.get("chunk_ids", [])) for k, v in m.get("files", {}).items()})

vs = get_vectorstore()
col = vs._collection
print("\n=== collection ===")
print("name:", col.name)
print("count:", col.count())

# 列出前 5 个 chunk 的 head
print("\n=== 库内前 5 个 chunk ===")
for i, item in enumerate(col.get(limit=5)["documents"]):
    print(f"  #{i}: {item[:70]!r}")

# 检查目标 chunk 是否在库：用 text_contains 或直接遍历找
print("\n=== 查找含 'two side effects' 的 chunk 是否在库 ===")
# chroma 支持 where_document contains
try:
    res = col.get(where_document={"$contains": "two side effects"}, limit=10)
    docs = res.get("documents") or []
    print(f"库中含 'two side effects' 的 chunk 数: {len(docs)}")
    for d in docs[:3]:
        print(f"  {d[:100]!r}")
except Exception as e:
    print("contains 查询失败:", e)

# 用精确 ID 查询（manifest 记录的是 _load_and_chunk_docs 生成的 id，格式 {name}_chunk_{i}）
# 但注意：sync_vectorstore 用同样的 id 写入，_load_and_chunk_docs 只是读取用
print("\n=== chroma_data 目录 ===")
for root, dirs, files in os.walk("chroma_data"):
    depth = root.count(os.sep)
    if depth <= 1:
        print(f"  {root}/")
        for f in files:
            sz = os.path.getsize(os.path.join(root, f))
            print(f"    {f} ({sz} bytes)")
