# 验证：batch_size=50 + sync_threshold=50 重建，link_lists 是否落盘
import os, shutil, glob
os.chdir("/app")
import numpy as np
import chromadb

TEST_DIR = "/tmp/test_persist4"
if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR)

client = chromadb.PersistentClient(path=TEST_DIR)
col = client.create_collection(
    "test_config",
    metadata={
        "hnsw:batch_size": 50,
        "hnsw:sync_threshold": 50,
        "hnsw:num_threads": 1,
    },
)

vecs = np.random.rand(120, 1024).astype(np.float32)
col.add(ids=[f"id{i}" for i in range(120)], embeddings=vecs, documents=[f"doc{i}" for i in range(120)])

print("=== add 120 条后文件 ===")
for f in sorted(glob.glob(TEST_DIR + "/**/*", recursive=True)):
    if os.path.isfile(f) and 'sqlite' not in f:
        print(f"  {os.path.relpath(f, TEST_DIR)}: {os.path.getsize(f)} 字节")

# 查询验证
res = col.query(query_embeddings=[vecs[7]], n_results=3)
print(f"\n查询 id7 最近邻: {res['ids'][0]}")

# 重启进程模拟：重新打开
print("\n=== 重新打开（模拟重启）===")
client2 = chromadb.PersistentClient(path=TEST_DIR)
col2 = client2.get_collection("test_config")
res2 = col2.query(query_embeddings=[vecs[7]], n_results=3)
print(f"重启后查询 id7 最近邻: {res2['ids'][0]}")
