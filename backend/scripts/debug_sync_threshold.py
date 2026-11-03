# 验证 sync_threshold 假设：小阈值插入后 link_lists 是否落盘
import os, glob, shutil
os.chdir("/app")

TEST_DIR = "/tmp/test_chroma"
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR)

import chromadb

# 配置 sync_threshold=10（默认是1000）
client = chromadb.PersistentClient(path=TEST_DIR)
col = client.create_collection(
    "test_sync",
    metadata={
        "hnsw:space": "l2",
        "hnsw:sync_threshold": 10,
        "hnsw:batch_size": 100,
    },
)

# 插入 20 个随机向量（1024维）
import numpy as np
ids = [f"id{i}" for i in range(20)]
vecs = np.random.rand(20, 1024).astype(np.float32)
col.add(ids=ids, embeddings=vecs, documents=[f"doc{i}" for i in range(20)])

# 检查落盘文件
print(f"\n=== {TEST_DIR} 文件 ===")
for root, dirs, files in os.walk(TEST_DIR):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, TEST_DIR)}: {os.path.getsize(p)} 字节")

# 查询验证
res = col.query(query_embeddings=[vecs[3]], n_results=3)
print(f"\n查询 id3 最近邻: {res['ids'][0]}")
