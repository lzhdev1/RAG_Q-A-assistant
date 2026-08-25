from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from vector_store import get_vectorstore
import os


def create_qa_chain():
    """
    创建问答链
    """

    # 1. 获取向量库检索器
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # 2. 初始化百炼兼容 OpenAI 接口
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
    )

    # 3. 创建 RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    return qa_chain