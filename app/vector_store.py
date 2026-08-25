from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import os


PERSIST_DIR = "./chroma_data"


def get_vectorstore(documents=None):

    embedding_model = DashScopeEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        dashscope_api_key=os.getenv("OPENAI_API_KEY")
    )


    if documents:

        print("正在初始化/更新向量数据库...")

        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model
        )


        batch_size = 6

        for i in range(0, len(documents), batch_size):

            batch = documents[i:i+batch_size]

            print(
                f"正在写入 {i + 1}-{min(i+batch_size, len(documents))}/{len(documents)}"
            )

            vectorstore.add_documents(batch)

        print("向量数据库构建完成")

    else:

        print("加载已有的向量数据库...")

        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model
        )


    return vectorstore