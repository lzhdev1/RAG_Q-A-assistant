from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import json
import os


PERSIST_DIR = "./chroma_data"
MANIFEST_PATH = os.path.join(PERSIST_DIR, "manifest.json")


# ── Manifest（记录已索引的文件状态） ─────────────────────────────

def _load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}}


def _save_manifest(manifest):
    os.makedirs(PERSIST_DIR, exist_ok=True)
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

def _chunk_file(filepath):
    """加载单个 PDF 并切分为文档块"""
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
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
    )

    manifest = _load_manifest()
    old_files = manifest["files"]

    # 扫描当前 docs 目录
    current_files = {}
    if os.path.exists(docs_dir):
        for name in sorted(os.listdir(docs_dir)):
            if name.lower().endswith(".pdf"):
                path = os.path.join(docs_dir, name)
                current_files[name] = _file_hash(path)

    # 对比：新增 / 删除 / 未变
    added   = [n for n, h in current_files.items() if n not in old_files or old_files[n]["hash"] != h]
    removed = [n for n in old_files if n not in current_files]
    # unchanged = [n for n in current_files if n not in added]

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
    )


def _get_embeddings():
    return DashScopeEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        dashscope_api_key=os.getenv("OPENAI_API_KEY"),
    )
