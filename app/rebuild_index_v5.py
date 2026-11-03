# 全量重建脚本 v5b：删除旧 collection 目录（不删挂载根），以新 metadata 重建
import os, sys, time, shutil, glob
sys.path.insert(0, "/app")
os.chdir("/app")

from vector_store import sync_vectorstore, _load_manifest, PARSER_VERSION, PERSIST_DIR

print(f"PARSER_VERSION={PARSER_VERSION}, PERSIST_DIR={PERSIST_DIR}")

# 只清掉 collection 目录 + manifest + sqlite，保留目录本身
if os.path.exists(PERSIST_DIR):
    for item in os.listdir(PERSIST_DIR):
        p = os.path.join(PERSIST_DIR, item)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
print("已清空 PERSIST_DIR 内容（保留目录挂载）")

t0 = time.time()
vs = sync_vectorstore("docs")
print(f"同步完成，耗时 {time.time()-t0:.1f}s")

manifest = _load_manifest()
total = sum(len(v["chunk_ids"]) for v in manifest["files"].values())
print(f"manifest 总 chunks: {total}")
col = vs._collection
print(f"collection count: {col.count()}")

print("\n=== HNSW 文件检查 ===")
ll_size = 0
pickle_size = 0
for f in sorted(glob.glob(PERSIST_DIR + "/**/*.bin", recursive=True)) + sorted(glob.glob(PERSIST_DIR + "/**/*.pickle", recursive=True)):
    size = os.path.getsize(f)
    name = os.path.basename(f)
    if name == "link_lists.bin": ll_size = size
    if name == "index_metadata.pickle": pickle_size = size
    status = "✅" if size > 0 else "⚠️空"
    print(f"  {status} {name}: {size} 字节")

if ll_size > 0 and pickle_size > 0:
    print("\n✅ HNSW 图已落盘（link_lists.bin + index_metadata.pickle 均非空）→ 检索将稳定")
else:
    print("\n❌ HNSW 图未落盘")
