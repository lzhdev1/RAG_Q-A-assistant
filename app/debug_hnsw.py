# 用 hnswlib 直接打开索引，检查图是否真的为空
import os, glob, struct
os.chdir("/app")

# 找最新的 collection 目录
cols = [d for d in glob.glob("/app/chroma_data/*/") if os.path.isdir(d)]
print(f"collection 目录: {cols}")
col_dir = cols[0] if cols else None
if not col_dir:
    print("无 collection 目录"); exit()

# 读 header.bin 看索引元数据
hdr_path = os.path.join(col_dir, "header.bin")
with open(hdr_path, "rb") as f:
    data = f.read()
print(f"\nheader.bin ({len(data)} 字节) 原始内容:")
print(data[:60])

# hnswlib 打开
try:
    import hnswlib
    print(f"\nhnswlib 版本: {hnswlib.__version__ if hasattr(hnswlib,'__version__') else '?'}")
    idx = hnswlib.Index(space="l2", dim=1024)
    idx.load_index(col_dir, max_elements=10000)
    print(f"索引元素数: {idx.get_max_elements()}")
    print(f"当前元素数: {idx.get_current_count()}")
    # 随机取一个元素看它的邻居
    labels = idx.get_ids_list()
    print(f"IDs 数: {len(labels)}, 前5: {labels[:5]}")
except Exception as e:
    print(f"hnswlib 打开失败: {e}")
