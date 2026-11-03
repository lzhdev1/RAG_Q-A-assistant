# RAG Q&A Assistant

基于 RAG（检索增强生成）的企业知识库问答系统，支持用户自主上传文档、增量索引、混合检索 + HyDE 查询理解 + Rerank 精排 + 防幻觉生成，并提供自研的 Open WebUI 风格 Web 聊天界面（SSE 流式输出）。当前知识库为行人重识别（Person Re-identification）领域的 3 篇论文，用于开发测试；上传任意 PDF/TXT/MD 即可扩展。

## 技术架构

```
用户提问（浏览器）
  ↓  SSE 流式
Next.js 前端（front/ 容器，服务端代理 /api/*）
  ↓
FastAPI 后端（rag-app 容器）
  ↓
三路混合检索：
  BM25 关键词（top-10，原始查询）     权重 0.4
  text-embedding-v3 向量（top-10）    权重 0.3
  HyDE 假设文档向量（top-10，LLM 生成假设文档后检索）权重 0.3
  ↓
DashScope Reranker 重排序（top-3，打分用原始查询）
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
| 向量数据库 | ChromaDB（嵌入式，本地持久化；已配置 HNSW batch/sync=50 保证落盘） |
| 检索策略 | BM25 + 向量 + HyDE 三路混合（EnsembleRetriever 0.4/0.3/0.3）→ Reranker 精排 |
| 查询理解 | HyDE（Hypothetical Document Embeddings，NeurIPS 2022）：LLM 生成假设文档补实体词 |
| 流式输出 | SSE（sources → token* → done） |
| 评估框架 | RAGAS（Faithfulness / Answer Relevancy / Context Precision） |
| 编排 | Docker Compose（rag-app + web 两容器） |

## 项目结构

```
RAG_Q-A-assistant/
├── backend/                    # 后端（FastAPI + RAG 链路）
│   ├── main.py                 # FastAPI 入口：/ask /ask/stream /upload /documents /health
│   ├── qa_chain.py             # RAG 问答链（三路混合检索 + HyDE + Reranker + Prompt）
│   ├── vector_store.py         # ChromaDB 向量库（增量同步 + 连字归一化 + manifest + HNSW 落盘配置）
│   ├── evaluate.py             # RAGAS 评估脚本
│   ├── loader.py               # 文档加载器
│   ├── test_dataset.json       # 30 条测试问答对
│   ├── eval_results_v5/v6.json # 评估结果存档（v6 为当前权威）
│   ├── .env                    # 环境变量配置（密钥，不入库）
│   ├── requirements.txt / Dockerfile / entrypoint.sh
│   └── scripts/                # 证据脚本与测试脚本（HNSW 根因实验、HyDE 验证、稳定性测试、重建脚本）
├── front/                      # 前端（Next.js）
│   ├── app/page.tsx            # 聊天主界面（侧边栏会话 + 文档管理 + 消息流）
│   ├── app/globals.css         # Open WebUI 风格深色主题样式
│   ├── lib/api.ts              # 前端 API 客户端（SSE 解析 / 上传 / 文档管理）
│   ├── next.config.js          # rewrites 服务端代理 /api/* → rag-app:8000
│   └── Dockerfile              # 前端镜像（Next.js standalone）
├── data/                       # 运行时数据（挂载卷）
│   ├── docs/                   # 知识库文档（PDF/TXT/MD，可通过前端上传）
│   └── chroma_data/            # ChromaDB 向量库（HNSW 落盘数据，不入库）
├── docker-compose.yml          # 编排 rag-app + web
├── ERROR_ANALYSIS.md           # 失败题归因 + 附录 A 四探针 + 附录 B HNSW 根因 + 附录 C HyDE
├── Record.md                   # 指标演进存档（基线 → v6）
└── README.md
```

## 快速开始

### 1. 配置环境变量

编辑 `backend/.env`：

```env
OPENAI_API_KEY=your-dashscope-api-key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.6-flash
JUDGE_LLM_MODEL=qwen3.7-plus-2026-05-26
EMBEDDING_MODEL=text-embedding-v3
RERANK_MODEL=qwen3-rerank
```

### 2. 启动服务（两容器）

```bash
docker compose up --build -d
```

- 前端：http://localhost:3000 （Open WebUI 风格聊天界面）
- 后端 API：http://localhost:8000

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
docker compose run --rm rag-app python evaluate.py --dataset test_dataset.json --output eval_results_vN.json
```

输出指标：
- **Faithfulness** — 回答是否忠实于检索到的上下文（防幻觉指标）
- **Answer Relevancy** — 回答与问题的相关程度
- **Context Precision** — 检索到的上下文中相关内容的占比（检索质量指标）

指标演进（30 题，同 judge 模型）：基线 0.7763/0.5668/0.6167 → v5（HNSW 修复）0.9755/0.8175/0.9028 → **v6（+HyDE）0.9542/0.8490/0.9472**（AR/CP 历史新高，Q29 从全 0 翻盘）。

## 检索流水线

```
用户问题
  ↓
┌─────────────────────────────────────┐
│  BM25 关键词检索 (top-10, 原始查询) │
│  向量语义检索 (top-10, 原始查询)     │── 合并去重 → Rerank top-3
│  HyDE 假设文档检索 (top-10, LLM 生成)│   （打分仍用原始查询）
└─────────────────────────────────────┘
  ↓
Qwen LLM 基于参考资料生成回答
```

### 可调参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_size` | `vector_store.py` | 800 | 文档切分块大小（overlap 100） |
| `BM25Retriever.k` | `qa_chain.py` | 10 | BM25 召回数量 |
| `vector_retriever.k` | `qa_chain.py` | 10 | 向量检索召回数量 |
| `HyDEVectorRetriever.k` | `qa_chain.py` | 10 | HyDE 假设文档检索召回数量 |
| `Reranker top_k` | `qa_chain.py` | 3 | Reranker 返回数量 |
| `weights` | `qa_chain.py` | [0.4, 0.3, 0.3] | BM25 / 向量 / HyDE 权重 |
| `ENABLE_HYDE` | `qa_chain.py`（环境变量） | 1 | HyDE 开关（置 0 关闭，回退双路检索） |
| `CUSTOM_PROMPT` | `qa_chain.py` | — | LLM 回答的 Prompt 模板 |
| `hnsw:batch_size/sync_threshold` | `vector_store.py` | 50/50 | chromadb HNSW 落盘配置（仅 create 时生效） |
| `API_PROXY_TARGET` | `front/next.config.js` | rag-app:8000 | 前端代理后端地址 |

## 健康检查

```bash
curl http://localhost:8000/health
# {"status": "running"}
```
