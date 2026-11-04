from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from vector_store import get_vectorstore
from typing import Any
import os
import glob
import httpx


# ── 开关（环境变量可覆盖，便于 A/B 对照） ─────────────────────────
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "1") == "1"  # HyDE 假设文档检索
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "0") == "1"  # 查询改写（历史负收益，默认关）
ENABLE_EN_QUERY = os.getenv("ENABLE_EN_QUERY", "1") == "1"  # 英文翻译检索路（v8：补中英语言鸿沟漏检）


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


# ── HyDE Prompt（假设文档生成） ───────────────────────────────────
HYDE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""请根据以下问题，撰写一段假设性的文档摘录（hypothetical passage）。

要求：
1. 内容直接陈述该问题的答案，像从学术论文正文摘录的段落；不要出现"根据文献""我查找了""论文指出"等元描述。
2. 尽量包含关键术语、专有名词、缩写、数据集名称等具体信息（问题涉及时）。
3. 控制在 150 字以内，用陈述性语言写成连贯段落。

问题：{question}

假设性文档摘录：""",
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
    """加载并切分文档，用于 BM25 检索器（支持 PDF/DOCX/PPTX/TXT/MD）"""
    from vector_store import _chunk_file, _is_supported

    all_chunks = []
    for name in sorted(os.listdir(docs_dir)):
        if not _is_supported(name):
            continue
        path = os.path.join(docs_dir, name)
        all_chunks.extend(_chunk_file(path))

    return all_chunks


class HyDEVectorRetriever(BaseRetriever):
    """
    HyDE（Hypothetical Document Embeddings）检索器（NeurIPS 2022, Gao et al.）：
      查询 → LLM 生成一段"假设性文档摘录"（含实体词/数据集名） → 用假设文档做向量检索。
    解决"查询里没有实体词、语义距离远"导致的召回漏（如 Q29：问题不含 Market-1501 等数据集名）。
    代价：每次查询多 1 次 LLM 调用（远低于 Rerank 翻倍的账单）。
    """
    vectorstore: Any
    llm: ChatOpenAI
    k: int = 10
    enabled: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager):
        # 关闭时回退：直接用原始查询做向量检索
        if not self.enabled:
            return self.vectorstore.similarity_search(query, k=self.k)
        try:
            response = self.llm.invoke(HYDE_PROMPT.format(question=query))
            hyde_doc = response.content
            print(f"  HyDE 假设文档: {hyde_doc[:120]}...")
            # 关键：用假设文档（而非原始查询）做向量检索
            return self.vectorstore.similarity_search(hyde_doc, k=self.k)
        except Exception as e:
            print(f"  HyDE 生成失败，回退原始查询: {e}")
            return self.vectorstore.similarity_search(query, k=self.k)


# ── 英文翻译 Prompt（v8：语言对齐检索路） ──────────────────────────
EN_TRANSLATE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Translate the following question into English for searching English academic papers.
Rules:
1. Output ONLY the English translation, no explanations, no quotes, no prefixes.
2. If the question is already in English, output it unchanged.

Question: {question}

English:""",
)


def _translate_to_english(llm, question: str) -> str:
    """LLM 将问题翻译为英文，用于对英文语料库做语言对齐检索；失败回退原 query"""
    try:
        response = llm.invoke(EN_TRANSLATE_PROMPT.format(question=question))
        text = response.content.strip()
        # 清理常见杂质前缀（模型偶尔输出 "Here is the translation:" 等）
        for prefix in ("Here is the English translation:", "Translation:", "English translation:", "Here is the translation:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text or question
    except Exception as e:
        print(f"  英文翻译失败，回退原始查询: {e}")
        return question


class EnglishQueryRetriever(BaseRetriever):
    """
    英文翻译检索路（v8）：中文 query → LLM 翻译为英文 → 向量检索。
    背景：text-embedding-v3 中→英跨语言对齐弱，中文 query 对英文论文库召回漏检
    （Q5 五方面枚举句：中文三路 top-30 均漏，英文 query top-10 即命中且过 rerank 阈值）。
    定位：检索语言与库语言对齐，噪声由 rerank 用原始 query 过滤。
    代价：每次查询多 1 次 LLM 翻译调用。
    """
    vectorstore: Any
    llm: ChatOpenAI
    k: int = 10
    enabled: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager):
        if not self.enabled:
            return self.vectorstore.similarity_search(query, k=self.k)
        en_q = _translate_to_english(self.llm, query)
        print(f"  英文检索路: {en_q[:100]}")
        return self.vectorstore.similarity_search(en_q, k=self.k)


class DashScopeRerankerRetriever(BaseRetriever):
    """DashScope Reranker 重排序检索器（HTTP API）+ 查询改写

    动态截断策略（v7，替代固定 top-3）：
      1. 全量 rerank 打分（top_n = 候选数，不预先截断）
      2. 相对阈值过滤：score >= top1 * alpha（alpha=0.6，top1 为每题天然锚点）
      3. 硬顶兜底：最多返回 hard_cap 条（防平缓长队爆窗口）
    数据依据：30 题评估集 rerank 分数探针——top1 跨 0.62~0.98，绝对阈值不可行；
    相对阈值下保留条数 1~6，大多 2~4，硬顶 6 平时不生效、极端兜底。
    """
    inner_retriever: BaseRetriever
    llm: ChatOpenAI
    top_k: int = 3          # 降级 fallback 用（API 失败时取前 top_k）
    alpha: float = 0.6      # 相对阈值系数：score >= top1 * alpha 才保留
    hard_cap: int = 6       # 动态截断后的硬顶条数
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

        # 3. 调用 DashScope Reranker API（Cohere 兼容格式），全量打分不截断
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
                    "top_n": min(len(candidates), 100),  # 全量打分（上限 100 防 API 限制）
                    "task": "Given a web search query, retrieve relevant passages that answer the query.",
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                print(f"Reranker API 错误 [{response.status_code}]: {response.text}")
                return candidates[:self.top_k]

            result = response.json()

            # 4. 全量排序 + 相对阈值过滤 + 硬顶（动态截断，替代固定 top-3）
            if "results" in result:
                ranked = [
                    (float(item["relevance_score"]), candidates[item["index"]])
                    for item in result["results"]
                ]
                if not ranked:
                    return []

                threshold = ranked[0][0] * self.alpha
                kept = [(s, d) for s, d in ranked if s >= threshold]
                if len(kept) > self.hard_cap:
                    kept = kept[:self.hard_cap]
                print(
                    f"Rerank 动态截断: 全量{len(ranked)}条 top1={ranked[0][0]:.3f} "
                    f"阈值={threshold:.3f} → 保留{len(kept)}条"
                )
                return [d for _, d in kept]

        except Exception as e:
            print(f"Reranker 调用失败，降级使用原始排序: {e}")

        # 降级：直接返回前 top_k 个
        return candidates[:self.top_k]


def build_rag_components():
    """
    构建 RAG 核心组件：LLM + Reranker 检索器（混合检索 + 精排）。
    返回 (llm, reranker_retriever)，供同步问答链与 SSE 流式接口复用。
    """

    # 1. 初始化 LLM
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
        extra_body={"enable_thinking": False},
    )

    # 2. 向量检索器（原始查询召回 top-10）
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    # 2b. HyDE 向量检索器（LLM 生成假设文档后召回 top-10，三路混合中的语义增强路）
    hyde_retriever = HyDEVectorRetriever(
        vectorstore=vectorstore,
        llm=llm,
        k=10,
        enabled=ENABLE_HYDE,
    )

    # 3. BM25 关键词检索器（召回 top-10）
    doc_chunks = _load_and_chunk_docs("docs")
    bm25_retriever = BM25Retriever.from_documents(
        doc_chunks,
        k=10,
    )

    # 3b. 英文翻译检索路（v8：中文 query 对英文论文库的语言鸿沟补漏）
    en_retriever = EnglishQueryRetriever(
        vectorstore=vectorstore,
        llm=llm,
        k=10,
        enabled=ENABLE_EN_QUERY,
    )

    # 4. 四路混合检索器（BM25 词面 + 原始向量语义 + HyDE 假设文档 + 英文翻译路）
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever, hyde_retriever, en_retriever],
        weights=[0.35, 0.25, 0.25, 0.15],  # BM25 让 0.05 给英文路；英文路只补漏不喧宾
    )

    # 5. Reranker 重排序（DashScope API，动态截断：相对阈值 α=0.6 + 硬顶 6 条；打分用原始查询）
    reranker_retriever = DashScopeRerankerRetriever(
        inner_retriever=ensemble_retriever,
        llm=llm,
        top_k=3,  # 仅 API 失败时的降级 fallback
        alpha=0.6,
        hard_cap=6,
        enable_query_rewrite=ENABLE_QUERY_REWRITE,
    )

    return llm, reranker_retriever


def create_qa_chain():
    """
    创建问答链（混合检索 + 查询改写 + Reranker）
    """
    llm, reranker_retriever = build_rag_components()

    # 创建 RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=reranker_retriever,
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    )

    return qa_chain