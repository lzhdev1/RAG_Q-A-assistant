# 精确实验：sync_threshold 计数逻辑 + persist 是否真的写文件
import os, shutil, glob
os.chdir("/app")
import numpy as np
import chromadb
from chromadb.segment.impl.vector import local_persistent_hnsw as m
import inspect

# 1. 看 _num_log_records_since_last_persist 在哪里 +=
src = inspect.getsource(m)
idx = src.find("_num_log_records_since_last_persist += 1")
if idx >= 0:
    line_start = src.rfind("\n", 0, idx)
    print("=== 计数递增处上下文 ===")
    print(src[line_start-200:line_start+100])

# 2. 实验：sync_threshold=3，分2批 add 10条，看文件
TEST_DIR = "/tmp/test_chroma2"
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR)

client = chromadb.PersistentClient(path=TEST_DIR)
col = client.create_collection(
    "test_persist",
    metadata={"hnsw:sync_threshold": 3, "hnsw:num_threads": 1},
)

vecs1 = np.random.rand(5, 1024).astype(np.float32)
vecs2 = np.random.rand(5, 1024).astype(np.float32)
col.add(ids=[f"a{i}" for i in range(5)], embeddings=vecs1, documents=[f"d{i}" for i in range(5)])
print("\n=== 第1批 add 后 ===")
for f in sorted(glob.glob(TEST_DIR + "/**/*", recursive=True)):
    if os.path.isfile(f):
        print(f"  {os.path.relpath(f, TEST_DIR)}: {os.path.getsize(f)} 字节")

col.add(ids=[f"b{i}" for i in range(5)], embeddings=vecs2, documents=[f"e{i}" for i in range(5)])
print("\n=== 第2批 add 后（累计10条 > 3）===")
for f in sorted(glob.glob(TEST_DIR + "/**/*", recursive=True)):
    if os.path.isfile(f):
        print(f"  {os.path.relpath(f, TEST_DIR)}: {os.path.getsize(f)} 字节")

# 3. 查询确认数据在
res = col.query(query_embeddings=[vecs1[0]], n_results=3)
print(f"\n查询 a0 最近邻: {res['ids'][0]}")
