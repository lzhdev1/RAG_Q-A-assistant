# main.py
import asyncio
import json
import os
import re
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vector_store import sync_vectorstore, get_vectorstore
from qa_chain import build_rag_components, CUSTOM_PROMPT

app = FastAPI(title="RAG Q&A Assistant")

# 允许前端（Next.js 独立容器/本地 dev server）跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发期全放行；生产可收紧到前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 运行时组件（启动时构建一次，供 /ask 与 /ask/stream 复用）
qa_chain = None
llm = None
reranker_retriever = None

DOCS_DIR = "docs"
ALLOWED_EXT = {".pdf", ".docx", ".pptx", ".txt", ".md"}


def _rebuild_components():
    """重建 LLM + Reranker 检索器（含 BM25）。文档上传/删除后必须调用，
    否则 BM25 关键词索引停留在启动时状态，检索不到新文档。"""
    global llm, reranker_retriever
    llm, reranker_retriever = build_rag_components()
    print("RAG 组件已重建（BM25 已刷新）")


def _safe_filename(filename: str) -> str:
    """去除路径分隔符与危险字符，只保留安全文件名"""
    name = os.path.basename(filename.replace("\\", "/"))
    name = re.sub(r"[^\w.\-\u4e00-\u9fff]", "_", name)
    return name


@app.on_event("startup")
async def startup_event():
    """启动时同步文档并初始化链"""
    global qa_chain, llm, reranker_retriever
    print("系统启动中...")

    # 增量同步：只处理新增/修改/删除的文档
    sync_vectorstore(DOCS_DIR)

    # 构建 LLM + Reranker 检索器
    _rebuild_components()
    print("RAG 组件就绪！")


class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QueryRequest):
    """提问接口（同步，返回完整回答 + 来源）"""
    if not llm or not reranker_retriever:
        raise HTTPException(status_code=500, detail="系统尚未初始化完成")

    try:
        # 手动走检索 -> 组装 context -> LLM 生成，保证来源可见
        docs = reranker_retriever.invoke(request.question)
        if not docs:
            return {
                "answer": "根据现有资料，暂时无法回答该问题。",
                "sources": [],
            }

        context = "\n\n".join(d.page_content for d in docs)
        prompt = CUSTOM_PROMPT.format(context=context, question=request.question)
        result = llm.invoke(prompt).content

        return {
            "answer": result,
            "sources": [
                {"content": d.page_content, "metadata": getattr(d, "metadata", {})}
                for d in docs
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_question_stream(request: QueryRequest):
    """提问接口（SSE 流式：先推来源，再逐 token 推答案）"""
    if not llm or not reranker_retriever:
        raise HTTPException(status_code=500, detail="系统尚未初始化完成")

    async def event_generator():
        try:
            # 1) 检索（同步阻塞调用放到线程池）
            docs = await asyncio.to_thread(reranker_retriever.invoke, request.question)

            if not docs:
                yield _sse("sources", [])
                yield _sse("token", {"text": "根据现有资料，暂时无法回答该问题。"})
                yield _sse("done", {})
                return

            # 2) 先推送检索来源
            sources = [
                {"content": d.page_content, "metadata": getattr(d, "metadata", {})}
                for d in docs
            ]
            yield _sse("sources", sources)

            # 3) 组装 prompt，流式生成
            context = "\n\n".join(d.page_content for d in docs)
            prompt = CUSTOM_PROMPT.format(context=context, question=request.question)
            async for chunk in llm.astream(prompt):
                text = getattr(chunk, "content", "") or ""
                if text:
                    yield _sse("token", {"text": text})

            yield _sse("done", {})
        except Exception as e:
            yield _sse("error", {"detail": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse(event: str, data) -> str:
    """构造 SSE 帧：event: xxx\ndata: json\n\n"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── 文档管理：上传 / 列表 / 删除 ──────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档：保存到 docs/ 并触发增量同步索引"""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，仅支持 {sorted(ALLOWED_EXT)}",
        )

    safe_name = _safe_filename(file.filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="文件名无效")

    os.makedirs(DOCS_DIR, exist_ok=True)
    save_path = os.path.join(DOCS_DIR, safe_name)

    # 同名覆盖时保留原文件并加时间戳后缀，避免增量同步误判
    if os.path.exists(save_path):
        stem, ext = os.path.splitext(safe_name)
        import time
        save_path = os.path.join(
            DOCS_DIR, f"{stem}_{int(time.time())}{ext}"
        )

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 增量索引新文件 + 重建检索组件（BM25 需包含新文档）
    try:
        await asyncio.to_thread(sync_vectorstore, DOCS_DIR)
        await asyncio.to_thread(_rebuild_components)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档保存成功但索引失败: {e}")

    return {
        "filename": os.path.basename(save_path),
        "size": len(content),
        "status": "indexed",
    }


@app.get("/documents")
def list_documents():
    """列出知识库中的文档"""
    files = []
    if os.path.exists(DOCS_DIR):
        for name in sorted(os.listdir(DOCS_DIR)):
            path = os.path.join(DOCS_DIR, name)
            if os.path.isfile(path):
                files.append({
                    "name": name,
                    "size": os.path.getsize(path),
                })
    return {"documents": files}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """删除文档并触发增量同步（移除对应向量）"""
    safe_name = _safe_filename(filename)
    path = os.path.join(DOCS_DIR, safe_name)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="文档不存在")

    os.remove(path)

    try:
        await asyncio.to_thread(sync_vectorstore, DOCS_DIR)
        await asyncio.to_thread(_rebuild_components)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件已删除但向量清理失败: {e}")

    return {"filename": safe_name, "status": "removed"}


@app.get("/health")
def health_check():
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
