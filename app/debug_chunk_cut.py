# 临时验证：chunk 切分是否把关键定义句截断
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import normalize_text
import glob

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)

def find_in_chunks(chunks, kw):
    """返回包含 kw 的所有 chunk 及其前后文"""
    hits = []
    for i, c in enumerate(chunks):
        t = normalize_text(c.page_content).lower()
        idx = t.find(kw.lower())
        if idx >= 0:
            hits.append((i, c.page_content))
    return hits

# 检查弱监督定义句
print("=" * 70)
print("① 弱监督定义句切分检查")
kw1 = "regardless of the dedicated annotation"
for pdf in glob.glob("docs/*.pdf"):
    pages = PyPDFLoader(pdf).load()
    for p in pages:
        p.page_content = normalize_text(p.page_content)
    chunks = splitter.split_documents(pages)
    hits = find_in_chunks(chunks, kw1)
    print(f"  [{pdf.split(chr(92))[-1]}] 含 '{kw1}' 的 chunk: {len(hits)} 个")
    for i, c in hits[:2]:
        # 显示该 chunk 是否覆盖整句定义
        t = c.lower()
        has_begin = "weakly supervised" in t
        has_identify = "identifying the person" in t
        print(f"    chunk#{i}: 含开头='weakly supervised'={has_begin}, 含'identifying the person'={has_identify}")
        print(f"      内容: {c[:120]!r}")

# 检查时间特征副作用句
print("=" * 70)
print("② 时间特征副作用句切分检查")
kw2 = "extra computational cost"
for pdf in glob.glob("docs/*.pdf"):
    pages = PyPDFLoader(pdf).load()
    for p in pages:
        p.page_content = normalize_text(p.page_content)
    chunks = splitter.split_documents(pages)
    hits = find_in_chunks(chunks, kw2)
    print(f"  [{pdf.split(chr(92))[-1]}] 含 '{kw2}' 的 chunk: {len(hits)} 个")
    for i, c in hits[:2]:
        t = c.lower()
        has_side = "two side effects" in t
        has_disturb = "disturb the system" in t
        print(f"    chunk#{i}: 含'two side effects'={has_side}, 含'disturb the system'={has_disturb}")

# 检查 PAN 三组件
print("=" * 70)
print("③ PAN 训练三组件句切分检查")
kw3 = "and 3) another"
for pdf in glob.glob("docs/*.pdf"):
    pages = PyPDFLoader(pdf).load()
    for p in pages:
        p.page_content = normalize_text(p.page_content)
    chunks = splitter.split_documents(pages)
    hits = find_in_chunks(chunks, kw3)
    print(f"  [{pdf.split(chr(92))[-1]}] 含 '{kw3}' 的 chunk: {len(hits)} 个")
    for i, c in hits[:2]:
        print(f"    chunk#{i}: {c[:250]!r}")
