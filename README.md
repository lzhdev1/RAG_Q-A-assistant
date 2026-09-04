# RAG Q&A Assistant

基于 RAG（检索增强生成）的学术文献问答系统，专注于行人重识别（Person Re-identification）领域的论文问答。

## 技术架构

```
用户提问
  ↓
混合检索（BM25 关键词 + 向量语义）
  ↓
DashScope Reranker 重排序
  ↓
Qwen LLM 生成回答
```

| 组件 | 技术选型 |
|------|---------|
| LLM | Qwen3.6-Flash（DashScope API） |
| Embedding | text-embedding-v3（DashScope API） |
| Reranker | qwen3-rerank（DashScope API） |
| 向量数据库 | ChromaDB（独立部署） |
| 检索策略 | BM25 + 向量混合检索 → Reranker 精排 |
| 评估框架 | RAGAS（Faithfulness / Answer Relevancy / Context Precision） |
| Web 服务 | FastAPI + Uvicorn（Docker 容器化） |

## 项目结构

```
RAG_Q-A-assistant/
├── app/
│   ├── main.py              # FastAPI 入口，/ask 问答接口
│   ├── qa_chain.py           # RAG 问答链（混合检索 + Reranker + Prompt）
│   ├── vector_store.py       # ChromaDB 向量库（增量同步 + 本地持久化）
│   ├── evaluate.py           # RAGAS 评估脚本
│   ├── loader.py             # 文档加载器
│   ├── test_dataset.json     # 30 条测试问答对
│   ├── .env                  # 环境变量配置
│   ├── requirements.txt      # Python 依赖
│   └── Dockerfile            # 应用镜像
├── docs/                     # PDF 论文（放入此目录自动索引）
├── docker-compose.yml        # 编排 rag-app + chroma-db
└── README.md
```

## 快速开始

### 1. 配置环境变量

编辑 `app/.env`：

```env
OPENAI_API_KEY=your-dashscope-api-key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.6-flash
JUDGE_LLM_MODEL=qwen3.7-plus-2026-05-26
EMBEDDING_MODEL=text-embedding-v3
RERANK_MODEL=qwen3-rerank
CHROMA_HOST=chroma-db
CHROMA_PORT=8000
```

### 2. 添加论文

将 PDF 文件放入 `docs/` 目录：

```bash
cp /path/to/your/paper.pdf docs/
```

### 3. 启动服务

```bash
docker-compose up --build -d
```

### 4. 提问

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "AGW基线在多少个数据集上取得了有竞争力的性能？"}'
```

响应示例：

```json
{
  "answer": "AGW 基线在四个 Re-ID 任务的十二个数据集上取得了有竞争力的性能。",
  "sources": ["...相关参考文献片段..."]
}
```

## 评估

运行 RAGAS 评估（3 个指标）：

```bash
docker-compose run --rm rag-app python evaluate.py
```

输出指标：
- **Faithfulness** — 回答是否忠实于检索到的上下文
- **Answer Relevancy** — 回答与问题的相关程度
- **Context Precision** — 检索到的上下文中相关内容的占比

可自定义测试集和输出路径：

```bash
docker-compose run --rm rag-app python evaluate.py \
  --dataset test_dataset.json --output eval_results.json
```

## 检索流水线

```
用户问题
  ↓
┌─────────────────────────────────┐
│  BM25 关键词检索 (top-10)       │
│  +                              │── 合并去重
│  向量语义检索 (top-10)           │
└─────────────────────────────────┘
  ↓
DashScope Reranker 精排 (top-3)
  ↓
Qwen LLM 基于参考资料生成回答
```

### 可调参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_size` | `qa_chain.py` | 500 | 文档切分块大小 |
| `BM25Retriever.k` | `qa_chain.py` | 10 | BM25 召回数量 |
| `vector_retriever.k` | `qa_chain.py` | 10 | 向量检索召回数量 |
| `Reranker top_k` | `qa_chain.py` | 3 | Reranker 返回数量 |
| `weights` | `qa_chain.py` | [0.5, 0.5] | BM25 与向量检索权重 |
| `CUSTOM_PROMPT` | `qa_chain.py` | — | LLM 回答的 Prompt 模板 |

## 健康检查

```bash
curl http://localhost:8000/health
# {"status": "running"}
```

ChromaDB 控制台：http://localhost:8001
