from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import json
import os
import unicodedata


PERSIST_DIR = "./chroma_data"
MANIFEST_PATH = os.path.join(PERSIST_DIR, "manifest.json")
PARSER_VERSION = 4  # 解析逻辑版本：v2 引入 NFKC 连字归一化；v3 chunk_size 500→800；v4 修复 chromadb HNSW 图不落盘（batch_size/sync_threshold 调小强制 persist）。变更时全量重建向量库

# chromadb 0.5.23 持久化缺陷修复：
#   默认 batch_size=100 / sync_threshold=1000，一次性写入 <1000 条时 _persist() 永不触发，
#   HNSW 图（link_lists.bin）不落盘 → 每次新进程从 SQLite 重建随机图 → ANN 检索间歇性漏检（Q19 根因）。
#   调小两者并让 batch_size==sync_threshold：每写满一个 batch 立即持久化，图稳定落盘。
CHROMA_COLLECTION_METADATA = {
    "hnsw:batch_size": 50,
    "hnsw:sync_threshold": 50,
    "hnsw:num_threads": 1,  # 单线程建图，图确定性更高
}


# ── 文本归一化（连字字符展开） ─────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Unicode NFKC 归一化：展开 PDF 中常见的连字字符（ligature）。
    例：'afﬁne'（U+FB00 ﬀ/U+FB01 ﬁ）→ 'affine'，
        避免 BM25 关键词检索因连字字符而匹配失败。
    同时兼容中文全角→半角等 NFKC 标准行为（对中文影响极小）。
    """
    if not text:
        return text
    return unicodedata.normalize("NFKC", text)


# ── Manifest（记录已索引的文件状态） ─────────────────────────────

def _load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            # 兼容旧 manifest：无 version 字段视为 v1
            manifest.setdefault("version", 1)
            manifest.setdefault("files", {})
            return manifest
    return {"version": PARSER_VERSION, "files": {}}


def _save_manifest(manifest):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    manifest["version"] = PARSER_VERSION
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _file_hash(filepath):
    """计算文件 SHA256，用于判断内容是否变化"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── 文本分块 ─────────────────────────────────────────────────────

def _is_supported(filename):
    """支持的文档类型：PDF / TXT / Markdown"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".pdf", ".txt", ".md"}


def _chunk_file(filepath):
    """加载单个文档并切分为文档块（含连字归一化）"""
    ext = os.path.splitext(filepath)[1].lower()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
        pages = loader.load()
    else:  # .txt / .md：按文本直接读取
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        pages = [Document(page_content=text, metadata={"source": filepath, "page": 0})]

    # 连字归一化：展开 afﬁne→affine 等，保证 BM25 关键词能命中
    for page in pages:
        page.page_content = normalize_text(page.page_content)
    return splitter.split_documents(pages)


# ── 核心：增量同步 ───────────────────────────────────────────────

def sync_vectorstore(docs_dir="docs"):
    """
    增量同步向量数据库：
      - 新增/修改的文件 → 向量化并写入
      - 已删除的文件 → 从向量库移除
      - 未变化的文件 → 跳过
    """
    embedding_model = _get_embeddings()

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_metadata=CHROMA_COLLECTION_METADATA,
    )

    manifest = _load_manifest()
    old_files = manifest["files"]
    old_version = manifest.get("version", 1)

    # 扫描当前 docs 目录
    current_files = {}
    if os.path.exists(docs_dir):
        for name in sorted(os.listdir(docs_dir)):
            if _is_supported(name):
                path = os.path.join(docs_dir, name)
                current_files[name] = _file_hash(path)

    # 解析器版本变更 → 全量重建（旧 chunk 全部删除，再重新向量化）
    version_changed = old_version != PARSER_VERSION
    if version_changed:
        print(f"解析器版本变更 v{old_version} → v{PARSER_VERSION}，全量重建向量库...")
        added = list(current_files.keys())
        removed = []
        # 删除旧版本的全部 chunk id
        collection = vectorstore._collection
        for name, info in old_files.items():
            ids_to_del = info.get("chunk_ids", [])
            if ids_to_del:
                collection.delete(ids=ids_to_del)
                print(f"  ✗ 清除旧版本 {name} 的 {len(ids_to_del)} 个向量")
        old_files = {}
    else:
        # 对比：新增 / 删除 / 未变
        added   = [n for n, h in current_files.items() if n not in old_files or old_files[n]["hash"] != h]
        removed = [n for n in old_files if n not in current_files]

    if not added and not removed:
        print("文档无变化，跳过向量化")
        return vectorstore

    # ── 处理新增 / 修改的文件 ──
    if added:
        print(f"新增 {len(added)} 个文档，正在向量化...")
        for name in added:
            chunks = _chunk_file(os.path.join(docs_dir, name))
            ids = [f"{name}_chunk_{i}" for i in range(len(chunks))]

            # DashScope embedding API 限制每批 ≤ 10 条
            batch_size = 6
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                vectorstore.add_documents(batch_chunks, ids=batch_ids)

            manifest["files"][name] = {
                "hash": current_files[name],
                "chunk_ids": ids,
            }
            print(f"  ✓ {name} → {len(chunks)} 个片段")

    # ── 处理已删除的文件 ──
    if removed:
        print(f"移除 {len(removed)} 个已删除文档的向量...")
        collection = vectorstore._collection
        for name in removed:
            ids_to_del = old_files[name].get("chunk_ids", [])
            if ids_to_del:
                collection.delete(ids=ids_to_del)
            del manifest["files"][name]
            print(f"  ✗ {name} 已移除")

    _save_manifest(manifest)
    print("向量数据库同步完成")
    return vectorstore


# ── 加载已有向量库（不触发同步） ─────────────────────────────────

def get_vectorstore():
    """仅加载已有向量库，不做任何同步"""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=_get_embeddings(),
        collection_metadata=CHROMA_COLLECTION_METADATA,
    )


def _get_embeddings():
    return DashScopeEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        dashscope_api_key=os.getenv("OPENAI_API_KEY"),
    )
