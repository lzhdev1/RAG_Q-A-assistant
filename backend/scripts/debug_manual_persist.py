# 验证：手动触发 chromadb 内部 _persist()，能否让 HNSW 图落盘
import os, sys, glob, pickle
os.chdir("/app")
import numpy as np

import chromadb
print(f"chromadb {chromadb.__version__}")
import hnswlib
print(f"hnswlib 版本: {getattr(hnswlib, '__version__', 'unknown')}")

# 建一个小测试库
import shutil
TEST_DIR = "/tmp/test_persist3"
if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR)

client = chromadb.PersistentClient(path=TEST_DIR)
col = client.create_collection(
    "test_manual",
    metadata={"hnsw:sync_threshold": 1000, "hnsw:num_threads": 1},
)

# 关键：如何访问 chromadb 内部的 hnsw segment？
# chromadb 0.5.23 API: client._server / client._admin_api ...
print("\n=== client 内部结构 ===")
print("client 类型:", type(client))
print("client 属性:", [a for a in dir(client) if not a.startswith('__')][:30])

# 尝试从 collection 拿到 segment
print("\n=== col 内部 ===")
print("col 类型:", type(col))
print("col 属性:", [a for a in dir(col) if 'segment' in a.lower() or 'client' in a.lower()][:20])

# 尝试访问 _client
try:
    inner = col._client
    print("\ncol._client:", type(inner))
    print("_client 属性:", [a for a in dir(inner) if not a.startswith('__')][:30])
except Exception as e:
    print(f"_client 访问失败: {e}")

# 插入 20 条测试
vecs = np.random.rand(20, 1024).astype(np.float32)
col.add(ids=[f"id{i}" for i in range(20)], embeddings=vecs, documents=[f"doc{i}" for i in range(20)])

# 尝试各种强制持久化方式
print("\n=== 强制持久化尝试 ===")
# 方式1: 通过 segment api
for method_name in ["persist", "_persist", "force_persist", "flush"]:
    for obj_name, obj in [("col", col), ("client", client)]:
        if hasattr(obj, method_name):
            try:
                r = getattr(obj, method_name)()
                print(f"  {obj_name}.{method_name}() 调用成功: {r}")
            except Exception as e:
                print(f"  {obj_name}.{method_name}() 报错: {e}")

# 方式2: 直接看 segment 对象
try:
    # chromadb 0.5.23: client._server 是 ServerAPI
    server = client._server
    print("\nclient._server:", type(server))
    # 尝试列出 segments
    segs = server.get_segments(collection_id=col.id)
    print(f"segments: {segs}")
    for seg in segs:
        print(f"  segment type={seg['type']} id={seg['id']}")
        if seg['type'] == 'urn:chroma:segment/vector/hnsw-local-persisted':
            # 找 segment 对象
            if hasattr(server, 'get_segment'):
                seg_obj = server.get_segment(seg['id'])
                print(f"  segment 对象: {type(seg_obj)}")
                if hasattr(seg_obj, '_persist'):
                    seg_obj._persist()
                    print("  _persist() 调用成功!")
except Exception as e:
    print(f"segment 访问失败: {type(e).__name__}: {e}")

# 检查文件
print("\n=== 强制持久化后文件 ===")
for f in sorted(glob.glob(TEST_DIR + "/**/*", recursive=True)):
    if os.path.isfile(f):
        rel = os.path.relpath(f, TEST_DIR)
        if 'sqlite' not in rel:
            print(f"  {rel}: {os.path.getsize(f)} 字节")
