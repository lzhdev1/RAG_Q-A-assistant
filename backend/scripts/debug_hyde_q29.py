# HyDE 单题验证：Q29（PAN 验证数据集）是否翻盘
# 用法：docker run 一次性容器
import os, sys
sys.path.insert(0, "/app")
os.chdir("/app")

from qa_chain import build_rag_components, HYDE_PROMPT
from langchain_openai import ChatOpenAI

llm, reranker = build_rag_components()

Q29 = "论文在哪些大规模行人重识别数据集上验证了PAN的性能？"

print(f"\n=== 测试 Q29 ===")
print(f"问题: {Q29}")
print(f"\nHyDE 假设文档: {llm.invoke(HYDE_PROMPT.format(question=Q29)).content}")

docs = reranker.invoke(Q29)
print(f"\nRerank top-{len(docs)}:")
for i, d in enumerate(docs):
    head = d.page_content[:150].replace("\n", " ")
    src = d.metadata.get("source", "?")
    print(f"  [{i+1}] ({src.split(chr(92))[-1][:40]}) {head}...")
    # 检查是否含数据集名
    for kw in ["Market", "CUHK", "Duke", "VIPeR", "数据集", "benchmark"]:
        if kw in d.page_content:
            print(f"       ↑ 含关键词: {kw}")
