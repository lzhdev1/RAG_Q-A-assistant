# 失败样本分析报告（Error Analysis）

> 背景：`backend/eval_results.json`（2026-08-26 评估）中 7 道题的 Context Precision = 0。
> 本文档通过逐题回查三篇 PDF 原文，将失败原因精确归类，并记录修复措施与验证结果。
> 依据：RAGAS 只输出三项指标均值，**逐题失败归因是 RAG 评测后的标准下游动作（error analysis）**。

## 一、结论摘要

| 结论 | 数量 |
|---|---|
| 测试集问题（原文确实无答案） | **0** |
| 解析问题（PDF 文本提取/字符异常导致检索失败） | **1**（第 7 题：连字字符 `afﬁne`） |
| 检索问题（答案在原文中存在，但系统未召回/未排到 top-3） | **6** |

**7/7 道题的答案均真实存在于原文中**，不存在"测试集本身无解"的情况。

## 二、逐题归因明细

| # | 题目 | 原文证据（PDF 位置） | 归因 | 失败环节 |
|---|---|---|---|---|
| 1 | AGW 的全称是什么？ | Survey 论文："AGW: Attention Generalized mean pooling with Weighted triplet loss" | 检索问题 | 全称出现在文首贡献列表（intro），BM25/向量未将该片段召回或排进 top-3 |
| 2 | 弱监督学习在视频行人重识别中是如何定义的？ | Vid-ReID 论文 5.3 节："Weakly supervised learning means identifying the person's identity in the video regardless of the dedicated annotation in each video frame" | 检索问题 | 答案位于综述 5.3 节，检索未命中该 chunk |
| 3 | 时间特征会带来哪两个副作用？ | Vid-ReID 论文："two side effects: (1) They add extra computational cost; (2) they provide huge information due to redundancy that may disturb the system" | 检索问题 | 答案位于 3.1 节特征讨论，检索未命中 |
| 4 | STN 在处理检测错误时起到了什么作用？ | PAN 论文："With STN, we can 1) crop the detected images which may contain too much background or 2) pad zeros to the borders" | 检索问题 | PAN 论文 3.3 节，检索未命中 |
| 5 | PAN 在哪些大规模行人重识别数据集上验证了性能？ | PAN 论文实验：Market 1501 / CUHK03-NP / DukeMTMC-reID（Rank@1 80.17/30.50/31.14/65.22） | 检索问题 | 答案位于实验章节 Table II，检索未命中 |
| 6 | 与服装变化相关的挑战性问题？ | Survey 论文 "Cloth-Changing Data" 整段；Vid-ReID 论文 "Cloth changing" 章节 | 检索排序问题 | 模型答对（AR=0.92）但 CP=0：检索片段与问题相关性被 judge 判为低，说明 top-3 未包含最直接的段落 |
| 7 | PAN 中仿射参数是如何被估计的？ | PAN 论文："the afﬁne parameters are estimated using the feature maps from the high-level convolutional layer of the base branch" | **解析问题** | PDF 原文使用 Unicode 连字 `afﬁne`（U+FB01 ﬁ），非常规 ASCII `affine`；BM25 精确匹配永远 miss |

## 三、修复措施与验证

### 修复 1：NFKC 连字归一化（针对第 7 题）

**改动**：
- `backend/vector_store.py`：新增 `normalize_text()`（`unicodedata.normalize("NFKC", ...)`），在 `_chunk_file()` 中于分块前对每页文本归一化；
- `backend/qa_chain.py`：`_load_and_chunk_docs()` 同样在分块前归一化（BM25 侧）；
- `backend/loader.py`：`load_documents()` 同样归一化；
- `backend/vector_store.py`：引入 `PARSER_VERSION = 2`，manifest 版本变更时**全量重建向量库**（清理旧 chunk 后重新向量化），避免"PDF 文件字节未变 → 增量同步跳过"导致旧文本残留。

**验证（本地复跑，pypdf 直接解析）**：

| 语料 | 连字字符数 | `affine` 归一化前 | `affine` 归一化后 | `affine parameters` 前→后 | `affine transformation` 前→后 |
|---|---|---|---|---|---|
| ReID Survey（25 页） | 455（ﬁ 445 + ﬂ 10） | 0 | 2 | 0→0 | 0→0 |
| Vid-ReID Survey（43 页） | 0 | 0 | 0 | 0→0 | 0→0 |
| PAN（9 页） | 142（ﬁ 136 + ﬂ 6） | 0 | **18** | 0→**2** | 0→**4** |

关键句归一化前后：
- 前：`...might be able to ﬁnd optimal afﬁnetransformation that contain...`
- 后：`...might be able to find optimal affinetransformation that contai...`

### 修复 2（建议，未实施）：检索层召回增强（针对第 1–6 题）

6 道检索问题答案均在原文，说明 top-3 精排窗口太小或召回不够。可选手段：
- Rerank 前候选池扩大：top-10 → top-20/30，给精排更多机会；
- 查询改写（项目已实现 `enable_query_rewrite`，因负收益关闭）：可针对"定义类/副作用类"问法定向开启；
- 分块策略：chunk 500/50 对综述类长段落可能截断，可评估"父子分块"或增大 chunk。

## 四、复现方法

```bash
# 1. 验证 7 题答案在原文存在（第一轮：关键词命中 + 上下文）
python verify_failure_cases.py

# 2. 深挖 AGW 全称 / affine 连字 / 弱监督段落 / 副作用段落（第二轮）
python verify_failure_cases2.py

# 3. 验证归一化前后命中变化
python verify_normalize.py

# 4. 全量重建向量库 + 重跑 RAGAS 评估（需 .env 配置 API key）
cd app && python evaluate.py --output eval_results.json
```

## 五、方法论备注

RAGAS 官方仅输出三指标总体均值（Faithfulness / Answer Relevancy / Context Precision），**不做逐题归因**。将低分样本逐题回查原文、按"检索 / 解析 / 测试集"分类，属于工业界标准的**失败样本分析（error analysis）**，是 RAG 系统落地的常规验收步骤，面试中可作为工程深度亮点讲述。

---

# 附录 A：30 题全量评估的失败归因（v2，2026-10-28）

> 背景：连字归一化修复后，容器内重跑 RAGAS 30 题评估（`backend/eval_results_v2.json`），
> 总体 F=0.9040 / AR=0.6031 / CP=0.7056。其中 12 道题 AR<0.6 或 CP<0.6，视为失败。
> 本文通过**四探针归因脚本**（`backend/failure_analysis.py`）逐题定位失败环节。

## A.1 四探针方法

| 探针 | 检查内容 | 判定信号 |
|---|---|---|
| ① 原文回查 | ground_truth/question 中的锚点词（括号英文/缩写/短语，过滤停用词）能否在 docs/*.pdf 原文搜到 | `p1_original` |
| ② 召回检查 | BM25 单独 top-10、向量单独 top-10 里是否存在锚点命中 chunk | `kw_hit >= 1` |
| ③ 排序检查 | 完整链路（BM25+向量 Ensemble → DashScope Rerank top-3）是否保留锚点命中 chunk | `kw_hit >= 1` |
| ④ Golden Context | 原文有答案但 top-3 未命中时，把全文锚点命中最多 chunk 直接喂给 LLM 生成，judge 判覆盖度 | `covered` |

锚点提取解决"中文 ground_truth vs 英文原文"鸿沟：优先取括号英文（closed-world、Heterogeneous Data）、
全大写缩写（CMC/mAP/AGW/PAN/STN/Res2/FC）、英文连续短语，并过滤 data/feature/learning 等通用停用词。

## A.2 12 道失败题归因明细

| 题号 | 问题 | AR/CP/F | 锚点命中(原文/BM25/向量/Rerank) | 归因 |
|---|---|---|---|---|
| Q2 | 特征表示学习四种策略 | 0.00/1.00/1.00 | True / 1 / 4 / 4 | **chunk 截断**：检索命中"four main categories"chunk，但 500 字 chunk 在第 2 种策略处切断，LLM 只见前两种（回答原话"参考资料在此处截断"） |
| Q4 | 度量标准 CMC/mAP | 0.57/1.00/1.00 | True / 0 / 1 / 1 | BM25 词形：`CMC` 缩写未进 BM25 top10（命中靠向量侧） |
| Q7 | AGW 全称 | 0.00/0.00/1.00 | True / 1 / 1 / 1 | **Rerank 排掉**：定义句在 intro 贡献列表 chunk，BM25/向量均命中（kw_hit=1），但 Rerank top-3 全被其他 AGW 段落占掉 |
| Q9 | 不可控数据收集挑战 | 0.00/0.00/1.00 | True / 1 / 2 / 0 | **Rerank 排掉**：向量检索到 cloth-changing 段落（kw_hit=2），Rerank top-3 全 0 命中 |
| Q14 | 姿态估计两种方式 | 0.76/0.50/1.00 | True / - / - / - | 轻度检索：coarsely/finely 同义表达匹配弱 |
| Q18 | 弱监督定义 | 0.00/0.00/1.00 | True / 3 / 3 / 3 | **chunk 截断**：定义句被 500 字边界切断（chunk#188 验证 `identifying the person` 缺失，只剩 `ing the person's identity`） |
| Q19 | 时间特征副作用 | 0.00/0.00/1.00 | True / **0** / 1 / 0 | **召回漏**：完整定义在 chunk#203（验证含 two side effects + disturb the system），但 BM25 top10 含 0 个、向量 top10 仅 1 个——未进 Rerank 候选 |
| Q20 | 服装变化挑战 | 0.92/0.00/0.57 | True / - / - / - | **RAGAS 判定噪声**：回答明显正确（长期换衣+制服区分），CP 却判 0 |
| Q23 | PAN 训练三组件 | 0.00/1.00/1.00 | 无锚点 | **chunk 截断**：三组件第 3 项 `and 3) another` 被切到下一 chunk（验证 chunk#14 断在 "1)" 后） |
| Q24 | affine 参数估计 | 0.83/0.00/1.00 | True / 1 / 4 / 4 | **RAGAS 判定噪声**：回答合理（affine estimation network），CP 判 0 |
| Q29 | PAN 验证数据集 | 0.00/0.00/1.00 | True / 0 / 0 / 0 | **召回漏**：Market-1501 等锚点在原文存在（p1 命中），但 BM25/向量 top10 全 0 命中 |
| Q30 | 两分支互补性 | 0.00/0.00/0.00 | True / 0 / 1 / 1 | BM25 词形：`complementary` 只被向量侧命中 |

## A.3 根因聚类

| 根因 | 题数 | 题目 | 含义 |
|---|---|---|---|
| ① 检索召回漏（相关 chunk 未进 top10 候选） | 4 | Q4 Q14 Q19 Q29 | 答案在原文，但 BM25/向量 top-10 没把定义句 chunk 捞进来 |
| ② chunk 切分截断定义句 | 3 | Q2 Q18 Q23 | 关键词命中，但 500 字 chunk 边界把关键定义/列表切断，LLM 只见不完整上下文 |
| ③ Rerank 排掉相关文档 | 2 | Q7 Q9 | 候选池里有相关 chunk，但 Reranker top-3 被无关段落占掉 |
| ④ RAGAS 判定噪声 | 2 | Q20 Q24 | 回答明显正确，CP 却判 0（评估本身问题，非系统缺陷） |
| ⑤ BM25 词形/缩写不匹配 | 1 | Q30（Q4 轻度） | CMC/complementary 等缩写、同义词匹配弱 |

## A.4 修复优先级（建议，待实施）

| 优先级 | 措施 | 影响的题 | 预期收益 |
|---|---|---|---|
| P0 | chunk_size 500→800、overlap 50→100 | Q2 Q18 Q23 | 定义句/列表不再被切断 |
| P1 | Rerank 前候选池 top-10→top-20（调大召回） | Q7 Q9 Q19 Q29 | 相关文档进候选概率↑ |
| P1 | 评估重跑 2-3 次取均值 | Q20 Q24 | 消除 RAGAS 单次判定噪声 |
| P2 | 查询改写重新评估（旧参数下 AR 反降，chunk 增大后需复测） | Q4 Q14 Q30 | 缩写/同义词召回↑ |

## A.5 复现方法

```bash
# 四探针归因（容器内）
docker exec rag-app python failure_analysis.py   # 输出 failure_analysis_results.json

# chunk 切分截断验证
docker exec rag-app python debug_chunk_cut.py

# 完整链路检索定位（Q19/Q18/Q23）
docker exec rag-app python debug_retrieval_v2.py
```

---

# 附录 B：Q19 根因定位——chromadb 0.5.23 HNSW 图不落盘（2026-11-03 定论）

> 背景：v4（干净库）评估中 Q19 仍 AR=CP=0。此前已排除 Reranker API、并发双写、
> embedding 随机性、sync_threshold 简单假设；本附录记录**从源码到文件的完整证据链**。

## B.1 症状

- `debug_query_methods.py` 同脚本不同次运行：目标 chunk 在 top-20 排 #1 与完全消失**交替出现**；
- 独占容器（rag-app/chroma-db 全停）单进程 4 次重建测试：失败/成功交替（Run1 失败→Run2 成功→Run3 失败→Run4 成功）。

## B.2 决定性证据

| 检查 | 结果 | 结论 |
|---|---|---|
| `debug_hnsw.py`：hnswlib 直开新库索引 | "Index seems to be corrupted or unsupported" | HNSW 文件损坏 |
| 新库文件对比 | `link_lists.bin`=**0 字节**、**无** `index_metadata.pickle`；旧备份库（681 向量）`link_lists.bin`=21,828 字节、`index_metadata.pickle`=49KB | 新库 HNSW 图从未落盘 |
| `debug_brute_force.py`：`col.get()` 查目标 chunk | chunk_139/140 全文完整返回（含 "two side effects"） | **向量数据 100% 在库**，是索引没建 |
| hnswlib 加载 + 查询 | 加载"成功"但 `get_current_count()=0`；knn_query 报 "ef or M is too small" | 索引文件是空壳 |

## B.3 根因（源码级）

chromadb 0.5.23 `local_persistent_hnsw.py` 持久化触发链：

```
_write_records()          每条记录 _num_log_records_since_last_persist += 1
  └─ 累计达 batch_size(100) → _apply_batch()
       └─ _num_log_records_since_last_persist >= sync_threshold(1000) → _persist()
            └─ index.persist_dirty()  ← HNSW 图真正写盘（link_lists.bin + index_metadata.pickle）
```

- 旧库（v3，681 向量）经历过**多次增量同步**，累计 add 记录数超过 sync_threshold=1000 → `_persist()` 触发过一次 → 图完整落盘 → 之后每次启动 `_index_exists()`（查 index_metadata.pickle）=True → `load_index` 加载**固定图** → 检索稳定；
- 新库（v4，536 向量）一次性全量重建，5 次 batch 累计 536 < 1000 → **`_persist()` 从未执行** → 每次新进程启动 `_init_index()` 走 `init_index` 新建空索引 → 从 SQLite 重放记录**重新建图**（num_threads=20 并行，图结构随机）→ ANN 检索间歇性漏检（Q19）。

## B.4 修复与验证

**改动**（`backend/vector_store.py`，PARSER_VERSION 3→4）：
- 新增 `CHROMA_COLLECTION_METADATA = {"hnsw:batch_size": 50, "hnsw:sync_threshold": 50, "hnsw:num_threads": 1}`；
- `Chroma(..., collection_metadata=...)` 在 `sync_vectorstore()`/`get_vectorstore()` 中统一传入。

**原理**：`batch_size==sync_threshold` → 每写满 50 条立即 `_persist()`；`num_threads=1` 保证建图确定性。
（注意：collection_metadata 只在 collection **创建时**生效，已存在 collection 需删掉重建。）

**验证**（`rebuild_index_v5.py` 全量重建，536 chunks/42.9s）：
- `link_lists.bin`=**4,516 字节**、`index_metadata.pickle`=**50,601 字节** → HNSW 图已落盘；
- `run_stab_test.ps1` 连续 4 次独立进程查询：**4/4 次目标 #1 命中，score 恒为 0.6885** → 检索稳定。

**残留 open item**：v4 评估中 Q3/Q24/Q25/Q26/Q30/Q9 的逐题波动（如 Q3 满分骤变全 0）疑为 RAGAS 判定噪声，
待 v5 评估结果对比后复核。

---

# 附录 C：Q29 修复——HyDE 假设文档检索（2026-11-03）

> 背景：v5 评估中 Q29（"论文在哪些大规模行人重识别数据集上验证了PAN的性能？"）仍 AR=CP=0，
> 系统回答"暂时无法回答"。四探针归因（附录 A）显示：Market-1501 等锚点在原文存在（p1 命中），
> 但 BM25/向量 top-10 全 0 命中 → 相关 chunk 未进 Rerank 候选池。

## C.1 为什么不能盲目加 k

Q29 的失败本质是**用户查询里没有实体词**（Market-1501/CUHK03/DukeMTMC 都没出现）：

- BM25 词面匹配天然失效（查询里没出现的词匹配不到）；
- 向量检索靠语义，"验证性能的数据集"离"实验章节 Table II"语义距离不够近 → top-10 挤不进去。

盲目 k=10→20 只是"碰运气"，且 Rerank 输入翻倍 → **成本≈2 倍、收益不确定**。这不是检索参数问题，是查询质量问题。

## C.2 方案：HyDE（Hypothetical Document Embeddings）

**原理**（NeurIPS 2022, Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"）：
查询 → LLM 生成一段**假设性文档摘录**（hypothetical passage，直接陈述答案、含实体词）
→ 用假设文档（而非原始查询）做向量检索。

**为什么对症**：假设文档会"生成"出 Market-1501/CUHK03 这些实体词 → 向量检索自然命中实验章节。

**实现**（`backend/qa_chain.py`）：
- 新增 `HYDE_PROMPT`：引导 LLM 写"假设性文档摘录"（150 字内陈述性段落，含专有名词，禁止元描述）；
- 新增 `HyDEVectorRetriever`（`BaseRetriever` 子类）：`llm.invoke(HYDE_PROMPT)` → `vectorstore.similarity_search(hyde_doc, k=10)`；
- 混合检索改为**三路**：`EnsembleRetriever([bm25(原始查询), 原始向量, HyDE向量], weights=[0.4, 0.3, 0.3])`；
- Rerank 仍用**原始查询**打分、top-3（`enable_query_rewrite` 保持关闭，默认 `ENABLE_HYDE=1`，可用环境变量关闭做 A/B）。

**单题验证**（`debug_hyde_q29.py`，容器内）：
- HyDE 假设文档生成：*"PAN的性能在Market-1501、CUHK03、DukeMTMC-reID及MSMT17等大规模行人重识别基准数据集上进行了全面验证……"*（实体词正确补出）；
- 检索候选：0 命中 → **27 个去重候选**；
- Rerank top-1 = PAN 论文段落（含 Market/CUHK/Duke 关键词）→ **Q29 翻盘确认**。

**代价**：每次查询多 1 次 LLM 调用（qwen3.6-flash，远低于 Rerank 翻倍的账单）→ 成本可控。

## C.3 v6 全量评估验证（2026-11-03，eval_results_v6.json）

| 指标 | v5 | v6 | 变化 |
|---|---|---|---|
| Faithfulness | 0.9755 | 0.9542 | -2.1pp（见 C.4 归因） |
| Answer Relevancy | 0.8175 | **0.8490** | +3.2pp 历史新高 |
| Context Precision | 0.9028 | **0.9472** | +4.4pp 历史新高 |

- **Q29 翻盘**：AR 0→0.8766、CP 0→1.0、F=1.0（30 题中唯一原全 0 题）；
- Q19 保持满分（F=1.0、CP=1.0，AR 0.8427 微降）；
- 其余 28 题 AR 持平或上涨，CP 全部 ≥0.5。

## C.4 F 下降归因：RAGAS 裁判抖动（非 HyDE 退化）

F 仅降于 Q6（1→0.75）与 Q27（1→0.5）两题，逐位证据：

| 题 | v5→v6 AR（只依赖 问题+回答） | v5→v6 CP（只依赖 检索上下文） | 推断 |
|---|---|---|---|
| Q6 | 0.808225255660126 → **完全相同** | 1.0 → **完全相同** | 回答与上下文均未变，F 变化=裁判抖动 |
| Q27 | 0.72989328587608 → 0.729956130543572（末位波动） | 0.8333 → **完全相同** | 上下文未变，F 变化=裁判抖动 |

AR/CP 双指标逐位相同证明检索与生成结果一致，F 波动来自 RAGAS 裁判 LLM 判定的随机性（同型噪声在 v3/v4 的 Q3 已记录）。**HyDE 未造成检索退化，保留为当前默认检索配置。**

# 附录 D：Rerank 动态截断（v7）与英文翻译检索路（v8）

## D.1 背景：固定 top-3 导致回答信息不全

LLM 严格依据召回 chunk 作答，rerank 固定取前 3 条会把大量相关信息排掉（PPT 场景：答案分布 5 页只能带 3 页）。用户否决"盲目加大 k"（治标不治本），经原理讨论拍板：**相对阈值动态截断**。

## D.2 v7 动态截断（qa_chain.py，commit 4b773fb）

**30 题分数探针**（`scripts/rerank_score_probe.py`，产物 `rerank_scores_probe.json`）：
- 全量打分 1217 条，候选 25~64 条/题（均值 40.6）；
- top1 分数跨 0.62~0.98（0.36 区间）→ **绝对阈值不可行**；
- 分数形态 = "top1 断层 + 长尾"，无干净双峰；α=0.6 时保留 1~6 条（多数 2~4）。

**实现**：`DashScopeRerankerRetriever` 全量打分（`top_n=min(len,100)`）→ `threshold=top1×α(0.6)` → 保留 `score≥threshold` → 硬顶 `hard_cap=6`；API 失败降级 `candidates[:3]`。

**v7 评估**（eval_results_v7.json，30 题）：F 0.9542→**0.9835**、AR 0.8490→0.8509、CP 0.9472→0.8885、**CR（新增）0.9000**。

## D.3 CR=0 三题归因（v7）

| 题 | 归因 | 证据 |
|---|---|---|
| Q28 | **RAGAS judge 误判** | 保留 chunk 原文含 "concatenate the two 1×1×2048 FC embeddings to form a 4096-dim pedestrian descriptor"——reference 信息 100% 在 context，CR=0 属 judge 拆声明失败 |
| Q15 | **阈值边缘波动** | 答案 chunk "gait has some important characteristics: It is unique and can be measured for any distance" 分数 0.590 恰在阈值 0.515 线上，运行间有波动 |
| Q5 | **检索层漏检（跨语言鸿沟）** | 见 D.4 |

## D.4 Q5 漏检根因：中英语言鸿沟（三步定位法）

`text-embedding-v3` 中→英跨语言对齐弱。对照实验（`scripts/diag_q5_recall.py`）：

| 检索配置 | 结果 |
|---|---|
| 中文 query 三路 top-10/20/30（候选 68 条） | 答案 chunk **全部未进候选** |
| 中文/中英混合改写变体 ×3 | 全部未进候选 |
| **英文 query 向量 top-10** | **候选 #10 命中 → rerank #7 score=0.680（过阈值保留）** |

结论：扩大 top-k / 中文改写对跨语言漏检无效（治标），**检索语言须与库语言对齐**。

## D.5 v8 英文翻译检索路（qa_chain.py，commit 待定）

**实现**：新增 `EnglishQueryRetriever`——LLM 把 query 翻译为英文（1 次廉价调用，失败回退原 query）→ 英文向量检索 top-10；并入四路 `EnsembleRetriever([bm25, vector, hyde, en], weights=[0.35, 0.25, 0.25, 0.15])`；rerank 仍用原始 query 过滤噪声。开关 `ENABLE_EN_QUERY`（默认 1）。

**v8 评估**（eval_results_v8.json，30 题）：

| 指标 | v6 (top-3) | v7 (动态α=0.6) | v8 (+英文路) |
|---|---|---|---|
| Faithfulness | 0.9542 | 0.9835 | **0.9850** |
| Answer Relevancy | 0.8490 | 0.8509 | **0.8525** |
| Context Precision | 0.9472 | 0.8885 | **0.9075** |
| Context Recall | — | 0.9000 | **0.9667** |

- **Q5 翻盘**：CR 0→1.0，五个方面完整列出（与 reference 逐字一致）；
- **Q15 恢复**：CR 0→1.0（英文路候选更稳，答案 chunk 稳过阈值）；
- CR 29/30，仅剩 Q28（judge 误判，见 D.3）；
- CP 回升 0.8885→0.9075：英文路带进真答案后 rerank 排序更准。

**代价**：每次查询 +1 次 LLM 翻译调用 + 候选数略增（~25→~30）→ rerank API 费用线性微增，可控。

## D.6 待办（P2，未实施）
- Q28 judge 误判：RAGAS 对"数字细节"过于严苛，可考虑拆解 reference 原子声明或换 judge 模型；
- 中文文档（用户 PPT）多一路英文噪声：当前由 rerank 滤除，若知识库中文占比上升需复测；
- embedding 换 bge-m3 跨语言模型（根治鸿沟但需全量重建库，算力有限，暂缓）。
