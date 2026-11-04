"""
RAG 系统评估脚本 (基于 RAGAS 框架)

评估指标:
  - Faithfulness (忠实度): 回答是否忠实于检索到的上下文，无幻觉
  - Answer Relevancy (答案相关性): 回答与问题的相关程度
  - Context Precision (上下文精确度): 检索到的上下文中相关内容的占比

用法 (Docker 容器内):
  python evaluate.py
  python evaluate.py --dataset test_dataset.json --output eval_results.json
"""

import asyncio
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "./chroma_data"


# ── 1. 加载测试集 ──────────────────────────────────────────────────
def load_test_dataset(dataset_path):
    """加载测试集，过滤掉不完整的条目"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid = [
        item for item in data
        if item.get("question") and item.get("ground_truth")
    ]
    print(f"加载测试集: {len(data)} 条, 有效 {len(valid)} 条")
    return valid


# ── 2. 初始化 LLM / Embedding ─────────────────────────────────────
def init_models():
    """初始化 LangChain 模型 (DashScope OpenAI 兼容接口)"""
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import DashScopeEmbeddings

    # RAG 系统用的 LLM
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
        extra_body={"enable_thinking": False},  # 关闭思考模式，避免 n 参数冲突
    )

    # RAGAS 裁判 LLM（用更强的模型当裁判）
    judge_llm = ChatOpenAI(
        model=os.getenv("JUDGE_LLM_MODEL", os.getenv("LLM_MODEL")),  # 默认回退到 LLM_MODEL
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
        extra_body={"enable_thinking": False},
    )

    embeddings = DashScopeEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        dashscope_api_key=os.getenv("OPENAI_API_KEY"),
    )

    return llm, judge_llm, embeddings


# ── 3. 加载 RAG 系统 ──────────────────────────────────────────────
def load_rag_system():
    """构建向量库 + 问答链"""
    from vector_store import sync_vectorstore
    from qa_chain import create_qa_chain

    print("同步向量数据库...")
    sync_vectorstore("docs")

    print("创建问答链...")
    return create_qa_chain()


# ── 4. 收集 RAG 回答 ─────────────────────────────────────────────
def collect_answers(qa_chain, test_data):
    """遍历测试集，收集 RAG 生成的回答和检索到的上下文"""
    print("\n收集 RAG 回答...")
    results = []

    for i, item in enumerate(test_data):
        q = item["question"]
        print(f"  [{i+1}/{len(test_data)}] {q[:50]}...")

        response = qa_chain.invoke({"query": q})
        contexts = [
            doc.page_content
            for doc in response.get("source_documents", [])
        ]

        results.append({
            "question": q,
            "answer": response["result"],
            "retrieved_contexts": contexts,
            "ground_truth": item["ground_truth"],
        })

    print(f"收集完毕: {len(results)} 条\n")
    return results


# ── 5. RAGAS 评估 ─────────────────────────────────────────────────
def run_ragas_evaluation(results, llm, embeddings):
    """使用 RAGAS 计算各项指标"""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    # 包装 LangChain 模型供 RAGAS 使用
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # 构建评估数据集
    eval_dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["retrieved_contexts"],
            "reference": r["ground_truth"],
        }
        for r in results
    ])

    # 配置评估指标
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),  # v7：量化"召回是否全"，验证动态阈值防漏
    ]

    print("运行 RAGAS 评估...")
    eval_results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return eval_results


# ── 6. 输出结果 ────────────────────────────────────────────────────
def print_results(eval_results, results):
    """格式化打印评估结果"""
    import numpy as np

    # RAGAS 0.2.x: eval_results.scores 是 list[dict]，每个 dict 是一个样本的各指标得分
    scores_list = eval_results.scores

    print("=" * 55)
    print("           RAG 系统评估结果 (RAGAS)")
    print("=" * 55)
    print()

    # 提取所有指标名称
    metric_names = list(scores_list[0].keys()) if scores_list else []

    # 计算每个指标的均值（忽略 NaN）
    overall = {}
    for metric in metric_names:
        values = [s.get(metric) for s in scores_list if s.get(metric) is not None and not (isinstance(s.get(metric), float) and np.isnan(s.get(metric)))]
        overall[metric] = np.nanmean(values) if values else float('nan')

    # 打印总体得分
    for metric, score in overall.items():
        if np.isnan(score):
            print(f"  {metric:<22} N/A (全部为 NaN)")
        else:
            print(f"  {metric:<22} {score:.4f}")
    print()
    print("=" * 55)

    # 逐条打印每个样本的各指标得分
    for metric in metric_names:
        print(f"\n  [{metric}]")
        for i, score_dict in enumerate(scores_list):
            val = score_dict.get(metric)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                print(f"    Q{i+1}: NaN")
            else:
                print(f"    Q{i+1}: {val:.4f}")

    # 统计汇总
    print("\n" + "=" * 55)
    print("  汇总统计")
    print("=" * 55)

    for metric in metric_names:
        values = [s.get(metric) for s in scores_list if s.get(metric) is not None and not (isinstance(s.get(metric), float) and np.isnan(s.get(metric)))]
        if values:
            avg = np.nanmean(values)
            std = np.nanstd(values)
            valid_count = len(values)
            total_count = len(scores_list)
            print(f"  {metric:<22} 均值={avg:.4f}  标准差={std:.4f}  (有效={valid_count}/{total_count})")
        else:
            print(f"  {metric:<22} 无有效数据")

    print()


def save_results(eval_results, results, output_file):
    """将评估结果保存为 JSON"""
    import numpy as np
    import math

    scores_list = eval_results.scores
    metric_names = list(scores_list[0].keys()) if scores_list else []

    # 计算总体均值（忽略 NaN）
    overall = {}
    for metric in metric_names:
        values = [s.get(metric) for s in scores_list if s.get(metric) is not None and not (isinstance(s.get(metric), float) and math.isnan(s.get(metric)))]
        overall[metric] = float(np.nanmean(values)) if values else None

    # 清理 per_sample 中的 NaN（JSON 不支持 NaN）
    def clean_score(val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return float(val)

    output = {
        "overall_scores": overall,
        "per_sample": [
            {
                "question": r["question"],
                "answer": r["answer"],
                "scores": {k: clean_score(v) for k, v in score_dict.items()},
            }
            for r, score_dict in zip(results, scores_list)
        ],
        "evaluated_at": datetime.now().isoformat(),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"详细结果已保存至: {output_file}")


# ── 主入口 ─────────────────────────────────────────────────────────
def main(dataset_path="test_dataset.json", output_file="eval_results.json"):
    print("=" * 55)
    print("       RAG 系统评估 (RAGAS)")
    print("=" * 55)

    # 1. 加载测试集
    test_data = load_test_dataset(dataset_path)
    if not test_data:
        print("错误: 测试集为空或格式不正确")
        sys.exit(1)

    # 2. 初始化模型
    llm, judge_llm, embeddings = init_models()

    # 3. 加载 RAG 系统
    qa_chain = load_rag_system()

    # 4. 收集回答
    results = collect_answers(qa_chain, test_data)

    # 5. RAGAS 评估
    eval_results = run_ragas_evaluation(results, judge_llm, embeddings)

    # 6. 输出 & 保存
    print_results(eval_results, results)
    save_results(eval_results, results, output_file)

    print("评估完成!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 系统评估")
    parser.add_argument(
        "--dataset",
        default="test_dataset.json",
        help="测试集 JSON 文件路径 (默认: test_dataset.json)",
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="评估结果输出路径 (默认: eval_results.json)",
    )
    args = parser.parse_args()

    main(args.dataset, args.output)
