# 看 hnsw_params.py：配置 key 的解析逻辑 + _init_index 完整代码
import inspect
from chromadb.segment.impl.vector import hnsw_params as hp
src = inspect.getsource(hp)
print("=== hnsw_params.py ===")
print(src[:3500])

print("\n\n=== _init_index 完整代码 ===")
from chromadb.segment.impl.vector import local_persistent_hnsw as m
src2 = inspect.getsource(m)
idx = src2.find("def _init_index")
if idx >= 0:
    next_def = src2.find("\n    def ", idx + 10)
    print(src2[idx:next_def if next_def > 0 else idx+2500])
