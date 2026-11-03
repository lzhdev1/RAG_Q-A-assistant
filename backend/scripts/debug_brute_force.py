# 决定性验证：从 SQLite 暴力搜索，目标向量是否真的在库里
import os, sys, sqlite3, numpy as np, pickle
sys.path.insert(0, "/app")
os.chdir("/app")
from dotenv import load_dotenv
load_dotenv()
from vector_store import _get_embeddings, normalize_text

QUERY = "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？"
emb = _get_embeddings()
qvec = np.array(emb.embed_query(QUERY), dtype=np.float32)

db = sqlite3.connect("/app/chroma_data/chroma.sqlite3")
cur = db.cursor()

# 找目标 chunk 的 embedding_id（metadata 里存 document 文本）
cur.execute("""
SELECT e.embedding_id FROM embeddings e
JOIN embedding_metadata m ON m.id = e.id
WHERE m.key='chroma:document' AND m.string_value LIKE '%temporal features act as a detailed%'
""")
target_ids = [r[0] for r in cur.fetchall()]
print(f"目标 chunk embedding_id: {target_ids}")

# 向量实际存在哪个文件？chromadb 0.5.23 向量在 hnsw data_level0.bin，不在 SQLite！
# SQLite embeddings 表只有 id 映射，向量在 segment 的 hnsw 文件里
# 所以正确做法：用 chromadb API 读取
import chromadb
client = chromadb.PersistentClient(path="/app/chroma_data")
col = client.get_collection("langchain")

# get() 不带向量：先确认目标存在
got = col.get(ids=target_ids, include=["documents"])
print(f"col.get 目标: {got['documents']}")

# 暴力验证：用 hnswlib 直接加载，如果加载成功并查询
import hnswlib
index = hnswlib.Index(space="l2", dim=1024)
try:
    index.load_index("/app/chroma_data/9bd38a13-011c-4a04-96ab-29111eec817b", is_persistent_index=True)
    print("\nhnswlib 加载成功!")
    print(f"元素数: {index.get_current_count()}")
    labels, dists = index.knn_query(qvec, k=10)
    print("top-10 标签:", labels[0])
    # 检查目标是否在内
    print(f"目标在 top-10: {any(t in labels[0] for t in target_ids)}")
except Exception as e:
    print(f"\nhnswlib 加载失败: {e}")
    # 尝试非持久化加载
    try:
        index.load_index("/app/chroma_data/9bd38a13-011c-4a04-96ab-29111eec817b")
        print("非持久化模式加载成功")
    except Exception as e2:
        print(f"非持久化也失败: {e2}")

db.close()
