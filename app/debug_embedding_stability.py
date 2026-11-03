# 验证 query embedding 稳定性：DashScopeEmbeddings 对同一文本多次生成的向量是否一致
import os, sys, hashlib
sys.path.insert(0, "/app")
os.chdir("/app")
from dotenv import load_dotenv
load_dotenv()

from vector_store import _get_embeddings

QUERY = "在讨论部分，论文认为视频行人重识别系统中的时间特征会带来哪两个副作用？"
emb = _get_embeddings()

hashes = []
for run in range(5):
    vec = emb.embed_query(QUERY)
    h = hashlib.md5(str(vec).encode()).hexdigest()[:12]
    hashes.append(h)
    print(f"Run{run+1}: embedding 维度={len(vec)} md5={h} 前5维={vec[:5]}")

print(f"\n唯一哈希数: {len(set(hashes))}/5")
if len(set(hashes)) == 1:
    print(">>> embedding 稳定")
else:
    print(">>> embedding 不稳定！同一文本每次向量不同 -> 向量检索结果随机")
