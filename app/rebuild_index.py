# 全量重建脚本：单进程、无并发环境下的干净重建
# 用法：docker run 一次性容器执行，重建后立即退出
import os, sys, time
sys.path.insert(0, "/app")
os.chdir("/app")

from vector_store import sync_vectorstore, _load_manifest, get_vectorstore
from vector_store import PARSER_VERSION, PERSIST_DIR

print(f"PARSER_VERSION={PARSER_VERSION}, PERSIST_DIR={PERSIST_DIR}")

# 确保从空目录开始（调用方负责清空）
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("警告: PERSIST_DIR 非空，仍继续（增量同步逻辑会处理）")

t0 = time.time()
vs = sync_vectorstore("docs")
print(f"同步完成，耗时 {time.time()-t0:.1f}s")

# 验证
manifest = _load_manifest()
total = sum(len(v["chunk_ids"]) for v in manifest["files"].values())
print(f"manifest 总 chunks: {total}")
col = vs._collection
print(f"collection count: {col.count()}")

# 强制持久化一次（确保 HNSW 文件落盘）
vs.persist()
print("persist() 完成")

# 检查 HNSW 文件
import glob
for f in sorted(glob.glob(PERSIST_DIR + "/*/*.bin")) + sorted(glob.glob(PERSIST_DIR + "/*/*.pickle")):
    size = os.path.getsize(f)
    print(f"  {os.path.basename(f)}: {size} 字节")
