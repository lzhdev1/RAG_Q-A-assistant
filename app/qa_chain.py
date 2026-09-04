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


# ── 查询改写 Prompt ──────────────────────────────────────────────
REWRITE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""请将以下问题改写为 3 个不同的版本，用于从学术文献中检索相关信息。
每个改写版本应从不同角度或关键词描述同一问题，保持语义一致但表达方式多样化。

原始问题：{question}

请用 JSON 数组格式输出 3 个改写版本，不要包含其他内容。
示例格式：["改写1", "改写2", "改写3"]

改写版本：""",
)


def _rewrite_query(llm, question: str):
    """
    使用 LLM 将原始问题改写为 3 个变体，用于多查询检索。
    返回 [原始问题, 改写1, 改写2, 改写3]
    """
    try:
        import json
        import re
        response = llm.invoke(REWRITE_PROMPT.format(question=question))
        text = response.content
        # 提取 JSON 数组
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            variants = json.loads(match.group())
            return [question] + variants
    except Exception as e:
        print(f"查询改写失败，使用原始查询: {e}")
    return [question]


# ── 自定义 Prompt 模板 ───────────────────────────────────────────
CUSTOM_PROMPT = PromptTemplate(
    template="""你是一个专业的学术知识问答助手。请根据参考资料回答用户的问题。

【回答规则】
1. **直接回答问题**：先给出简明答案，再根据需要补充必要细节。不要复述问题，不要添加与问题无关的背景信息。
2. **严格基于参考资料**：仅使用参考资料中的信息作答，不要编造或推测参考资料中没有的内容。
3. **资料不足时明确说明**：如果参考资料中没有足够的信息，请回复："根据现有资料，暂时无法回答该问题。"
4. **保持简洁客观**：回答要精炼，不添加个人观点或主观评价。

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
    """DashScope Reranker 重排序检索器（HTTP API）+ 查询改写"""
    inner_retriever: BaseRetriever
    llm: ChatOpenAI
    top_k: int = 3
    model: str = "gte-rerank-hybrid"
    enable_query_rewrite: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager):
        # 1. 查询改写：生成多个查询变体
        if self.enable_query_rewrite:
            queries = _rewrite_query(self.llm, query)
            print(f"查询改写: {queries}")
        else:
            queries = [query]

        # 2. 对每个查询变体进行检索，去重合并
        seen_hashes = set()
        candidates = []
        for q in queries:
            docs = self.inner_retriever.invoke(q)
            for doc in docs:
                # 用内容哈希去重
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    candidates.append(doc)

        if not candidates:
            return []

        print(f"检索到 {len(candidates)} 个候选文档（去重后）")

        # 3. 调用 DashScope Reranker API（Cohere 兼容格式）
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
                    "query": query,  # 用原始查询做 rerank
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

            # 4. 按 reranker 排序结果返回（Cohere 兼容格式：results 在顶层）
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
    创建问答链（混合检索 + 查询改写 + Reranker）
    """

    # 1. 初始化 LLM（查询改写需要）
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
    )

    # 2. 向量检索器（召回 top-10 供 reranker 筛选）
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    # 3. BM25 关键词检索器（召回 top-10）
    doc_chunks = _load_and_chunk_docs("docs")
    bm25_retriever = BM25Retriever.from_documents(
        doc_chunks,
        k=10,
    )

    # 4. 混合检索器（BM25 + 向量）
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    # 5. Reranker 重排序（DashScope API，返回 top-3）
    reranker_retriever = DashScopeRerankerRetriever(
        inner_retriever=ensemble_retriever,
        llm=llm,
        top_k=3,
        enable_query_rewrite=False,
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