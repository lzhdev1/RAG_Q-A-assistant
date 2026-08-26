from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from vector_store import get_vectorstore
import os
import glob
import httpx


# 自定义 Prompt 模板
CUSTOM_PROMPT = PromptTemplate(
    template="""你是一个专业的知识问答助手。请严格根据以下检索到的参考资料来回答用户的问题。

【回答规则】
1. 仅使用参考资料中提供的信息作答，不要编造、猜测或补充参考资料中没有的内容。
2. 如果参考资料中包含与问题相关的信息，请基于这些信息进行准确、简洁的回答。
3. 如果参考资料中没有足够的信息来回答问题，请明确回复："根据现有参考资料，暂时无法回答该问题。"
4. 回答时请保持客观中立，不要添加个人观点或主观评价。
5. 如果引用了参考资料中的内容，请在相关句子末尾用 [来源X] 的格式标注出处。

【参考资料】
{context}

【用户问题】
{question}

【你的回答】""",
    input_variables=["context", "question"],
)


def _load_and_chunk_docs(docs_dir="docs"):
    """加载并切分文档，用于 BM25 检索器"""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )

    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

    return all_chunks


class DashScopeRerankerRetriever(BaseRetriever):
    """DashScope Reranker 重排序检索器（HTTP API）"""
    inner_retriever: BaseRetriever
    top_k: int = 3
    model: str = "gte-rerank-hybrid"

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager):
        # 1. 从内部检索器获取候选文档
        candidates = self.inner_retriever.invoke(query)

        if not candidates:
            return []

        # 2. 调用 DashScope Reranker API（Cohere 兼容格式）
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("RERANK_MODEL", self.model)

            response = httpx.post(
                "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "query": query,
                    "documents": [doc.page_content for doc in candidates],
                    "top_n": min(self.top_k, len(candidates)),
                    "task": "Given a web search query, retrieve relevant passages that answer the query.",
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                print(f"Reranker API 错误 [{response.status_code}]: {response.text}")
                return candidates[:self.top_k]

            result = response.json()

            # 3. 按 reranker 排序结果返回（Cohere 兼容格式：results 在顶层）
            if "results" in result:
                ranked_docs = []
                for item in result["results"]:
                    ranked_docs.append(candidates[item["index"]])
                return ranked_docs

        except Exception as e:
            print(f"Reranker 调用失败，降级使用原始排序: {e}")

        # 降级：直接返回前 top_k 个
        return candidates[:self.top_k]


def create_qa_chain():
    """
    创建问答链（混合检索 + Reranker）
    """

    # 1. 向量检索器（召回 top-10 供 reranker 筛选）
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    # 2. BM25 关键词检索器（召回 top-10）
    doc_chunks = _load_and_chunk_docs("docs")
    bm25_retriever = BM25Retriever.from_documents(
        doc_chunks,
        k=10,
    )

    # 3. 混合检索器（BM25 + 向量）
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    # 4. Reranker 重排序（DashScope API，返回 top-3）
    reranker_retriever = DashScopeRerankerRetriever(
        inner_retriever=ensemble_retriever,
        top_k=3,
    )

    # 5. 初始化 LLM
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
    )

    # 6. 创建 RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=reranker_retriever,
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    )

    return qa_chain