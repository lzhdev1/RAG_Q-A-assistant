# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loader import load_documents
from vector_store import get_vectorstore
from qa_chain import create_qa_chain
import uvicorn

app = FastAPI(title="RAG Q&A Assistant")

# 全局变量存储 QA 链，避免每次请求都重新加载模型
qa_chain = None

@app.on_event("startup")
async def startup_event():
    """启动时加载数据并初始化链"""
    global qa_chain
    print("系统启动中...")
    
    # 1. 加载并处理文档
    docs = load_documents("docs")
    
    # 2. 构建向量库
    get_vectorstore(documents=docs)
    
    # 3. 初始化问答链
    qa_chain = create_qa_chain()
    print("系统就绪！")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """提问接口"""
    if not qa_chain:
        raise HTTPException(status_code=500, detail="系统尚未初始化完成")
        
    try:
        result = qa_chain.invoke({"query": request.question})
        return {
            "answer": result["result"],
            "sources": [doc.page_content[:100] + "..." for doc in result.get("source_documents", [])]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)