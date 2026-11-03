# 对比新旧库 collection 配置 + 读 _write_records 完整代码
import sqlite3, os, inspect
os.chdir("/app")

# 1. 对比新旧库 config
for path, label in [("/app/chroma_data/chroma.sqlite3", "新库9bd38a13"), 
                    ("/app/chroma_data_bak_20261028/chroma.sqlite3", "旧库cfaf7d8e(681)"),
                    ("/app/chroma_data_bak_broken_20261103/chroma.sqlite3", "坏库18bfd855(536)")]:
    try:
        db = sqlite3.connect(path)
        cur = db.cursor()
        cur.execute("SELECT id, name, dimension, config_json_str FROM collections")
        rows = cur.fetchall()
        print(f"\n=== {label} ===")
        for r in rows:
            print(f"  collection: id={r[0]} name={r[1]} dim={r[2]}")
            print(f"    config: {r[3][:300]}")
        db.close()
    except Exception as e:
        print(f"\n{label}: {e}")

# 2. 读 _write_records 完整逻辑
from chromadb.segment.impl.vector import local_persistent_hnsw as m
src = inspect.getsource(m)
idx = src.find("def _write_records")
if idx >= 0:
    next_def = src.find("\n    def ", idx + 10)
    print(f"\n\n=== _write_records (len={next_def-idx}) ===")
    print(src[idx:next_def if next_def > 0 else idx+2000])
