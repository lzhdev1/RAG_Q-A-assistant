# 检查 chroma SQLite 的 collection 配置与 chromadb 版本
import sqlite3, os
os.chdir("/app")
import chromadb

print(f"chromadb 版本: {chromadb.__version__}")

db = sqlite3.connect("/app/chroma_data/chroma.sqlite3")
cur = db.cursor()

print("\n=== collections 表 ===")
cols = [r[1] for r in cur.execute("PRAGMA table_info(collections)")]
print(f"列: {cols}")
for row in cur.execute("SELECT * FROM collections"):
    print(row)

print("\n=== collection_metadata 表 ===")
try:
    for row in cur.execute("SELECT * FROM collection_metadata"):
        print(row)
except Exception as e:
    print(f"err: {e}")

print("\n=== segments 表 ===")
try:
    for row in cur.execute("SELECT id, type, scope, collection FROM segments"):
        print(row)
except Exception as e:
    print(f"err: {e}")

print("\n=== segment_metadata 表 ===")
try:
    for row in cur.execute("SELECT * FROM segment_metadata LIMIT 20"):
        print(row)
except Exception as e:
    print(f"err: {e}")

print("\n=== embeddings 计数 ===")
try:
    for row in cur.execute("SELECT COUNT(*) FROM embeddings"):
        print(f"embeddings 表行数: {row[0]}")
except Exception as e:
    print(f"err: {e}")
db.close()
