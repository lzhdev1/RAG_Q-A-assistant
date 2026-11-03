# RAG Q&A Assistant

基于 RAG（检索增强生成）的企业知识库问答系统，支持用户自主上传文档、增量索引、混合检索 + Rerank 精排 + 防幻觉生成，并提供自研的 Open WebUI 风格 Web 聊天界面（SSE 流式输出）。当前知识库为行人重识别（Person Re-identification）领域的 3 篇论文，用于开发测试；上传任意 PDF/TXT/MD 即可扩展。

## 技术架构

```
用户提问（浏览器）
  ↓  SSE 流式
Next.js 前端（web/ 容器，服务端代理 /api/*）
  ↓
FastAPI 后端（rag-app 容器）
  ↓
混合检索（BM25 关键词 top-10 + text-embedding-v3 向量 top-10，权重 0.5/0.5）
  ↓
DashScope Reranker 重排序（top-3）
  ↓
Qwen LLM 基于参考资料生成回答（防幻觉 Prompt）
```

| 组件 | 技术选型 |
|------|---------|
| 前端 | Next.js 14 + React 18 + TypeScript（自研，Open WebUI 风格深色主题） |
| 后端 | FastAPI + Uvicorn |
| LLM | Qwen（DashScope API） |
| Embedding | text-embedding-v3（DashScope API） |
| Reranker | DashScope Rerank（Cohere 兼容接口） |
| 向量数据库 | ChromaDB（独立容器，本地持久化） |
| 检索策略 | BM25 + 向量混合检索（EnsembleRetriever）→ Reranker 精排 |
| 流式输出 | SSE（sources → token* → done） |
| 评估框架 | RAGAS（Faithfulness / Answer Relevancy / Context Precision） |
| 编排 | Docker Compose（web + rag-app + chroma-db 三容器） |

## 项目结构

```
RAG_Q-A-assistant/
├── app/
│   ├── main.py              # FastAPI 入口：/ask /ask/stream /upload /documents /health
│   ├── qa_chain.py          # RAG 问答链（混合检索 + Reranker + Prompt，build_rag_components）
│   ├── vector_store.py      # ChromaDB 向量库（增量同步 + 连字归一化 + manifest）
│   ├── evaluate.py          # RAGAS 评估脚本
│   ├── loader.py            # 文档加载器
│   ├── test_dataset.json    # 30 条测试问答对
│   ├── eval_results_v2.json # 最近一次评估结果
│   ├── .env                 # 环境变量配置
│   ├── requirements.txt     # Python 依赖
│   └── Dockerfile           # 后端镜像
├── web/
│   ├── app/page.tsx         # 聊天主界面（侧边栏会话 + 文档管理 + 消息流）
│   ├── app/globals.css      # Open WebUI 风格深色主题样式
│   ├── lib/api.ts           # 前端 API 客户端（SSE 解析 / 上传 / 文档管理）
│   ├── next.config.js       # rewrites 服务端代理 /api/* → rag-app:8000
│   └── Dockerfile           # 前端镜像（Next.js standalone）
├── docs/                    # 知识库文档（PDF/TXT/MD，可通过前端上传）
├── docker-compose.yml       # 编排 web + rag-app + chroma-db
├── ERROR_ANALYSIS.md        # 7 道失败题归因分析
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

### 2. 启动服务（三容器）

```bash
docker-compose up --build -d
```

- 前端：http://localhost:3000 （Open WebUI 风格聊天界面）
- 后端 API：http://localhost:8000
- ChromaDB 控制台：http://localhost:8001

### 3. 使用方式

**方式 A：Web 界面（推荐）**
- 打开 http://localhost:3000，输入问题（Enter 发送，Shift+Enter 换行）
- 左下角「知识库文档」面板可上传 PDF/TXT/MD 更新知识库、删除文档
- 回答下方展示 Rerank 精排后的引用来源

**方式 B：API 调用**

提问（同步）：

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "视频行人重识别系统由哪两个主要组件构成？"}'
```

提问（SSE 流式）：

```bash
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "视频行人重识别系统由哪两个主要组件构成？"}'
```

上传文档：

```bash
curl -X POST http://localhost:8000/upload -F "file=@your_doc.pdf"
```

## 评估

运行 RAGAS 评估（3 个指标）：

```bash
docker-compose run --rm rag-app python evaluate.py
```

输出指标：
- **Faithfulness** — 回答是否忠实于检索到的上下文（防幻觉指标）
- **Answer Relevancy** — 回答与问题的相关程度
- **Context Precision** — 检索到的上下文中相关内容的占比（检索质量指标）

最近一次评估（30 题，v2 归一化后）：F=0.9040 / AR=0.6031 / CP=0.7056。

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
| `API_PROXY_TARGET` | `web/next.config.js` | rag-app:8000 | 前端代理后端地址 |

## 健康检查

```bash
curl http://localhost:8000/health
# {"status": "running"}
```
