from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def load_documents(docs_dir="docs"):
    """
    加载 PDF 文档并切片
    """

    if not os.path.exists(docs_dir):
        print(f"文档目录 {docs_dir} 不存在")
        return []


    # 加载 PDF
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()

    print(f"成功加载 {len(documents)} 个 PDF 页面")


    # 文本切片
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )


    splits = text_splitter.split_documents(documents)

    print(f"文档已切分为 {len(splits)} 个片段")


    return splits