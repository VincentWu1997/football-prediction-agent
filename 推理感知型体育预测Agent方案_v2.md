# 推理感知型体育预测 Agent — 修订版实施方案 (v2)

> 版本：v2 · 2026-09
> 相对 v1（DeepSeek 初稿）的关系：保留"任务复杂度路由 + 可验证预测"主线，重写量化、MCP、Agent 形态、Spring AI 定位与排期。修订点见文末附录 A。

---

## 0. 一句话定位

一个**根据任务复杂度动态分配推理预算**的足球赛事预测 Agent：简单查询走轻量量化模型，深度分析走强模型 + 多工具 ReAct 循环；预测由可回测的统计/ML 模型给出，LLM 只负责信息聚合与解释，所有数字可溯源、可回测、可复现。

---

## 1. 设计原则（先立规矩）

1. **预测是地基，Agent 是外壳**。先证明 ML 预测对赔率 baseline 有信息量，再谈 Agent 包装。
2. **LLM 不产生数字，只组织数字**。概率、伤停、积分必须来自工具/RAG 返回，输出附来源。
3. **一切性能数字来自可复现实验**：benchmark 脚本 + 原始结果 CSV/JSON + 图表生成脚本三件套全部入仓，README 只引用实验产物，禁止手填目标值。
4. **量化名词必须名副其实**：本地 GGUF 实验不叫 AWQ；云端跑真 AWQ/GPTQ/FP8 checkpoint；论文数字必须标注引用。
5. **MCP 只用在它有真实价值的地方**：跨语言（Python↔Java）与工具能力标准化，不给本地 HTTP 调用无谓套壳。

---

## 2. 总体架构

```
                          ┌──────────────────────┐
                          │ Streamlit 演示前端    │  对话 + 推理 trace 可视化
                          └──────────┬───────────┘
                                     │ SSE
              ┌──────────────────────┴──────────────────────┐
              │  Spring AI 企业服务层 (Java 17 + Boot 3)     │
              │  对外 REST / 鉴权 / 限流 / 会话 / 可观测      │
              │  ChatClient + MCP Client + Advisor RAG(对照) │
              └──────────────────────┬──────────────────────┘
                                     │ OpenAI 兼容 HTTP / MCP(Streamable HTTP)
              ┌──────────────────────┴──────────────────────┐
              │  FastAPI Agent Runtime (Python 3.11)         │
              │  Planner(L1-L4+预算) → ReAct Tool Loop       │
              └───┬──────────┬──────────┬──────────┬────────┘
                  │          │          │          │
         ┌────────┴──┐ ┌─────┴────┐ ┌───┴────┐ ┌───┴─────────┐
         │ MCP: 数据  │ │ MCP: 预测 │ │MCP:知识│ │ 统一推理客户端 │
         │ SQL/积分榜 │ │ ELO/DC/XGB│ │ RAG    │ │ (OpenAI 兼容) │
         │ 战绩/赔率  │ │ 蒙特卡洛  │ │溯源    │ └──┬───────┬──┘
         └─────┬─────┘ └────┬─────┘ └───┬────┘    │       │
               │            │           │     Ollama   vLLM/SGLang
        ┌──────┴────────────┴───────────┴──┐  (本地 GGUF) (云 GPU, 实验/生产)
        │ PostgreSQL 16 + pgvector          │
        │ 结构化赛事数据 / 向量库 / 预测账本  │
        └───────────────────────────────────┘
```

### 2.1 关键设计决策

| 决策 | 理由 |
|---|---|
| 推理访问层统一为 **OpenAI 兼容协议**（`/v1/chat/completions`） | Ollama、vLLM、SGLang 全部原生兼容；换后端只改 `config/models.yaml`，Agent 代码零修改——这才是真正的后端解耦 |
| Planner 产出"层级 + 预算"，执行层是 **ReAct tool-calling 循环** | v1 的固定线性 DAG 不是 ReAct；工具应由模型在循环中自主选择，Planner 只约束层级预算（最大迭代数、是否允许蒙特卡洛、走 fast/precise） |
| MCP 只封装三类工具：数据、预测、知识 | 这三类需要被 Python Agent 与 Java 服务**跨语言复用**，MCP 有真实必要性；推理调用不套 MCP |
| Spring AI 定位为**企业服务层/BFF**，不是"重写一个 Python MCP Server" | 鉴权、限流、会话、任务管理、可观测与 Java 侧 RAG 对照，回答"为什么有 Java 模块" |
| fast/precise 模型选 **Qwen3-4B / Qwen3-8B**（非 1.5B/7B） | 4B Q4 约 2.5GB，中文与 function calling 可靠性明显好于 1.5B；8B 是当前 dense 甜点。**所有 Qwen3 请求必须关闭思考模式**（见 §6.4） |

### 2.2 任务分层（保留 v1 主线，补齐预算定义）

| 层级 | 类型 | 示例 | 模型档 | 工具迭代上限 | 允许重型工具 |
|---|---|---|---|---|---|
| L1 | 事实查询 | "英超积分榜" | fast 4B | 2 | 否 |
| L2 | 单场快速预测 | "曼联对利物浦谁赢" | fast 4B | 4 | predict_match |
| L3 | 深度单场分析 | "详细分析双红会，考虑伤停和赛程" | precise 8B | 8 | predict_match + RAG + 赔率 |
| L4 | 全局模拟 | "模拟本赛季英超前四概率" | precise 8B | 10 | simulate_season（蒙特卡洛） |

---

## 3. 数据层

### 3.1 数据源

| 数据 | 来源 | 说明 |
|---|---|---|
| 历史赛果、积分榜、进球、**博彩公司赔率（含 closing odds）** | [football-data.co.uk](https://www.football-data.co.uk/mmz4281/season/league.csv) | 免费 CSV，覆盖英超/西甲/德甲等 20+ 联赛、约 20 个赛季，含多家公司 1X2/大小球赔率。**是赔率 baseline 的关键** |
| 球队/球员资料、伤停新闻、赛前采访、战术分析 | 公开网页采集（限合规来源）→ 进 RAG | 非结构化，只进向量库 |
| 实时赛况/阵容（可选增强） | API-Football 免费层（约 100 次/日） | 仅 Demo 当日比赛用，非必需 |

明确边界：**积分榜、比分、交锋、赔率等结构化数据一律走 SQL 工具，不进向量库**；向量库只放非结构化文本。

### 3.2 存储

PostgreSQL 16 + pgvector（Docker Compose 一把拉起），逻辑分三库域：

- `f_matches / f_odds / f_standings`：结构化赛事数据
- `v_chunks`：文档切片 + embedding（HNSW，cosine）
- `pred_ledger`：预测账本（赛前写入预测，赛后回填结果），是准确率追踪的依据

### 3.3 管道

`data/scripts/`：`fetch_csv.py`（按赛季×联赛批量下载，带断点续传）→ `normalize.py`（统一队名主数据，队名映射表人工维护一次）→ `load_pg.py`（幂等 upsert）。每日增量用 Airflow 都嫌重，单文件 cron + 日志即可。

---

## 4. ML 预测层（项目地基）

### 4.1 模型矩阵（三层，互为对照与集成）

| 模型 | 作用 | 特点 |
|---|---|---|
| ELO（含主场优势、K 值按杯赛/联赛区分） | 基线评分 | 零依赖、可解释，为 XGBoost 提供特征 |
| Dixon-Coles 双变量 Poisson | 比分分布建模 | 直接输出 1X2/大小球概率，可解释、LLM 好引用 |
| XGBoost | 主模型 | 融合统计特征与市场特征 |

**XGBoost 特征（v1 完全缺失，这里给明确清单）**：

- ELO 差值与赛前胜率差
- 主/客队近 5/10 场滚动：积分、进球、失球、xG（无 xG 数据则用射门代理）
- Dixon-Coles 输出的主胜/平/客胜概率（模型堆叠）
- 赛程密度：近 7 天/14 天出场数、欧战客场旅行（疲劳指数）
- 市场特征：赔率隐含概率（去水后）、开盘→临场赔率漂移
- 伤停影响估计（从 RAG 新闻中结构化抽取的关键球员缺阵标记，W6 后接入）

### 4.2 评估体系（面试火力最集中处，必须做对）

1. **切分**：禁止随机 split。按日期做 **walk-forward**（如训练窗 5 个赛季 → 预测下一赛季 → 滚动），杜绝未来信息泄漏；特征计算时点必须严格早于比赛开球。
2. **指标组合**：
   - Log Loss、Brier Score（概率质量）
   - **RPS（Ranked Probability Score）**——有序结局 1X2 的行业标准指标
   - Accuracy 仅作辅助展示
   - 校准曲线 + ECE
3. **Baseline（必须有，否则模型无意义）**：
   - 博彩公司**赔率去水（overround 归一化）后的隐含概率**——这是最强公开预测，打不过是常态，要诚实展示差距
   - Dixon-Coles、ELO 单模型
4. **投注视角回测**：以临场赔率做 1X2 价值投注（模型概率 - 隐含概率 > 阈值），统计 ROI、命中率、最大回撤，按联赛分仓。**结论必须带样本量与置信区间，不做选择性汇报。**
5. L4 蒙特卡洛：用 Dixon-Coles 的 λ_home/λ_away 对剩余赛程逐轮抽样 10,000 次，输出各队进前四/夺冠/降级概率。

---

## 5. 推理实验体系（简历推理优化关键词的真实落点）

### 5.1 实验总表

| 实验 | 环境 | 模型/量化 | 产出 |
|---|---|---|---|
| E1 本地 GGUF 多精度 | Mac mini（Metal） | Qwen3-8B：Q4_K_M / Q5_K_M / Q6_K / Q8_0 | 延迟、tok/s、内存、任务质量 |
| E2 vLLM 量化对比 | 云 GPU 1 卡（4090/A100/H20 按量） | Qwen3-8B：BF16 / 官方 **AWQ** / 自量化 **GPTQ** / 官方 **FP8** | 同精度质量 + 量化间吞吐/显存 |
| E3 vLLM 服务化调优 | 同上 | AWQ 为主 | 并发 1→64 吞吐/TTFT 曲线、`gpu_memory_utilization` × `max_num_seqs` × `max_model_len` 网格、启动日志 GPU KV blocks 数 |
| E4 SGLang 对照 | 同上 | Qwen3-8B-AWQ | 公共前缀场景 **RadixAttention** vs vLLM prefix caching 的 TTFT/吞吐 |
| E5 任务质量评测 | 复用 E1/E2 端点 | 全部量化档 | 体育评测集上的结构化指标 |

成本预估：按量 GPU ¥2–5/小时 × 10–15 小时 ≈ **¥50 以内**可完成 E2–E4。

### 5.2 E1：本地 GGUF（措辞必须准确）

- Ollama 拉取 Qwen3-4B 与 Qwen3-8B 的多精度 GGUF tag；明确这是 **llama.cpp k-quant 量化**，与 AWQ/GPTQ 是不同算法族，文档中绝不写"GGUF 等价 AWQ"。
- 16GB 内存预算：fast=4B Q4_K_M（≈2.5GB），precise=8B Q6_K（≈6.5GB）；需要双模型驻留时设 `OLLAMA_MAX_LOADED_MODELS=2` 并实测内存/swap，默认接受 Ollama 按需换入。
- 测量：预热身后记录 prompt eval tok/s、generation tok/s、首 token 延迟、峰值内存。

### 5.3 E2/E3：vLLM（云端，真 AWQ/GPTQ/FP8）

```bash
pip install "vllm>=0.8.5"
vllm serve Qwen/Qwen3-8B            # BF16 基线
vllm serve Qwen/Qwen3-8B-AWQ        # 官方 AWQ checkpoint
vllm serve Qwen/Qwen3-8B-FP8        # 官方 FP8（分块量化；Hopper/H20 收益最佳，Ada 上仅验证可跑）
# GPTQ：用 GPTQModel 自行量化 Qwen3-8B 后 serve，量化脚本入仓
```

- 压测工具用 vLLM 自带 `benchmarks/benchmark_latency.py`、`benchmark_throughput.py`、`benchmark_serving.py`（SGLang 侧用 `sglang.bench_serving`），**不自造压测器**。
- 调参记录：`--gpu-memory-utilization`（0.85/0.90/0.95）、`--max-num-seqs`、`--max-model-len` 与启动日志中 **GPU KV cache blocks** 的对应关系；解释 PagedAttention 如何把 KV cache 按 block 分配、消除显存碎片（机制讲清，引用 [vLLM 论文](https://arxiv.org/abs/2309.06180)，论文的 24× 数字只作文献引用，不冒充实测）。
- Hopper/H20 上额外做 `--kv-cache-dtype fp8` 实验；4090（Ada）不作为 FP8 结论环境。

### 5.4 E4：SGLang 的真实卖点

构造**前缀复用负载**：同一段球队背景资料（固定 system+context 前缀）+ 50 个不同提问，分别请求 SGLang 与 vLLM（均开前缀缓存），对比前缀命中率、TTFT、整体吞吐。这是 RadixAttention 的主场，避免"SGLang 只在简历里出现"。

### 5.5 E5：体育领域评测集（量化对比的"质量"列）

`benchmarks/eval_sets/` 自建 200–500 条 jsonl：

- 结构化输出任务：**JSON 合法率**、schema 字段准确率、输出数字与工具返回的**一致率**（防 LLM 篡改概率）
- 知识问答：答案正确率（人工标注标准答案；LLM-as-judge 仅作辅助且公布裁判模型与 prompt）
- 路由评测：100 条 query 标注 L1–L4，报告 Planner 分类准确率/混淆矩阵

---

## 6. Agent 设计

### 6.1 Planner（规划，不包办执行）

fast 模型 + few-shot 输出结构化 JSON：

```json
{
  "level": "L3",
  "route": "precise",
  "max_tool_iters": 8,
  "allow_simulation": false,
  "reason": "含'详细分析'+ 涉及单场，需要伤停与赔率"
}
```

冷启动可保留 v1 的关键词规则，但必须有 §5.5 的 100 条标注集给出真实分类准确率，不能只靠关键词。

### 6.2 ReAct 执行循环（LangGraph）

```
入口 → bind_tools(模型) → 模型决策
            ↑                │
            │          有 tool_calls？
            │           ├─ 是 → ToolNode 执行（带 trace 记录）→ 回边
            │           └─ 否 → 结构化输出校验 → END
```

- 与 v1 区别：工具不是按 level 硬编码顺序调用，而是模型在预算内自主 ReAct；Planner 只通过 system prompt 注入预算与工具白名单。
- 每一步写入 `trace`（节点、工具名、入参摘要、耗时、tokens），供前端渲染与后续分析。

### 6.3 工具清单（三个 MCP Server）

| Server | 工具 |
|---|---|
| data | `query_standings`、`query_recent_form`、`query_h2h`、`query_odds` |
| prediction | `predict_match`（返回 XGBoost/DC/ELO 三套概率 + 校准信息）、`simulate_season` |
| knowledge | `search_knowledge(query, k)`（返回文本块 + 来源 URL/标题，强制溯源） |

Python Agent 作为 MCP client 以 Streamable HTTP 连接；Java 服务复用同一组 Server——**MCP 的跨语言价值在此闭环**。

### 6.4 Qwen3 思考模式（v1 完全没提的大坑）

Qwen3 默认先输出思考内容，fast 路由的延迟会成倍增加。生产对话统一**关闭思考**：

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

工具调用可靠性不足时，仅对 L4 规划开启思考模式做 A/B，并体现在评测数据里。

### 6.5 稳定性与降级（v1 空白项）

- Ollama 服务启动后用 `keep_alive` 预加载，避免冷加载数十秒超时
- 超时分层：连接 5s；read 超时 fast 30s / precise 120s；指数退避重试 2 次
- 结构化输出：schema 校验 + `json-repair` 修复 + 修复失败则一次"纠错重写"，再失败降级模板
- 模型熔断：滑动窗口失败率超阈值则临时摘除该后端并告警
- 全程 SSE 流式输出，避免长分析阻塞前端

---

## 7. RAG 体系

- Embedding：**BAAI/bge-m3**（1024 维，本地可跑，中英双语；满足 pgvector HNSW ≤2000 维限制）
- 摄入：文档按语义段落切片（512–1024 token，重叠 10%），元数据含联赛/球队/日期/来源/类型
- 检索：HNSW cosine top-k + 元数据过滤；L3 分析固定取 top-5 并拼入 prompt，逐条标注 `[1][2]` 编号，要求模型引用编号、不得使用编号外事实
- 评测：标注 100 个问题→相关文档，报告 **recall@5 / MRR**；并做"无 RAG vs 有 RAG"的答案正确率对照
- 内容更新：每周批量摄入伤停/赛前新闻，赛后过期内容降权

---

## 8. Spring AI 企业服务层（Java 17 + Spring Boot 3）

真实职责，不是重写 Python：

| 模块 | 实现 |
|---|---|
| 对外 API/BFF | `/api/analyze`（透传 Python runtime）、`/api/predictions`（预测账本查询）、SSE 转发 |
| 鉴权/限流 | Spring Security（API Key/JWT）+ Bucket4j 限流 |
| 会话记忆 | `ChatClient` + `MessageChatMemoryAdvisor`（会话 ID 维度） |
| **MCP 跨语言调用** | `spring-ai-starter-mcp-client`（Streamable HTTP 连 Python 三个 MCP Server），`ToolCallbackProvider` 直接注入 ChatClient |
| **Java RAG 对照** | `spring-ai-starter-vector-store-pgvector` 的 `PgVectorStore` + `QuestionAnswerAdvisor`，与 Python RAG 在同一评测集上对照 recall 与答案质量 |
| 可观测 | Micrometer + Actuator，指标：路由分布、各模型延迟/错误率、工具耗时 |

裁剪预案：时间不足时保留前三项 + 一个最小 MCP client 调通 `predict_match` 即可讲清"MCP 跨语言编排"，Java RAG 对照可砍。

---

## 9. 前端与演示

Streamlit 三栏：左栏对话；右栏实时 trace（L 级别、模型/量化档、工具调用序列、每步耗时、tokens）；底栏"预测账本"看板（历史预测数、LogLoss/Brier、相对赔率 baseline、投注 ROI 曲线，全部从 `pred_ledger` 实时算）。

**演示脚本**：同一会话内依次问 L1（"英超积分榜"）→ L2（快速预测）→ L3（伤停深度分析）→ L4（前四模拟），右栏直观呈现路由差异——这是 v1 最值得保留的演示设计。

---

## 10. 目录结构

```
sports-agent/
├── README.md                     # 只放实验产物数字，附复现命令
├── docker-compose.yml            # postgres+pgvector
├── config/models.yaml            # 推理后端注册表（base_url/model/量化/预算）
├── data/
│   ├── raw/  processed/  master/ # CSV、队名映射
│   └── scripts/                  # fetch / normalize / load
├── src/sports_agent/
│   ├── data/                     # repository, SQL
│   ├── ml/                       # elo / dixon_coles / features / xgb / simulate
│   ├── eval/                     # walkforward, metrics, calibration, betting_roi
│   ├── rag/                      # ingest, embed, retrieve
│   ├── inference/                # openai_compat client, router, resilience
│   ├── agent/                    # planner, graph(ReAct), state, prompts, tracing
│   └── api/                      # FastAPI + SSE
├── mcp_servers/                  # data / prediction / knowledge 三 Server
├── benchmarks/
│   ├── eval_sets/                # 路由/质量/检索 三类 jsonl
│   ├── local_gguf/  cloud_vllm/  sglang/
│   └── results/                  # 原始 CSV/JSON + 出图脚本（图表必须由数据生成）
├── spring-service/               # Spring AI BFF
├── frontend/app.py
└── tests/
```

---

## 11. 十周排期（按每周 12–15 小时业余投入；全职可压缩至 5–6 周）

| 周 | 目标 | 交付物（验收标准） |
|---|---|---|
| W1 | 数据管道 + EDA | 近 10 赛季 5 大联赛 CSV 入库；EDA 报告（主场优势、联赛进球分布） |
| W2 | 评估框架 + 统计基线 | walk-forward 脚手架；ELO、Dixon-Coles、**赔率去水 baseline** 的 LogLoss/Brier/RPS 对照表 |
| W3 | XGBoost 主模型 | 特征管道；模型 vs 赔率 baseline 对照表 + 校准曲线 + 投注回测（带样本量） |
| W4 | 蒙特卡洛 + 本地推理 | L4 模拟输出；Ollama Qwen3-4B/8B 跑通；OpenAI 兼容客户端 + FastAPI 骨架 + thinking 关闭验证 |
| W5 | RAG | bge-m3 + pgvector 摄入流水线；recall@5/MRR 评测；带溯源问答 |
| W6 | MCP + ReAct Agent | 三个 MCP Server；Planner + ReAct 循环；L1–L4 端到端；容错降级；路由分类准确率报告 |
| W7 | 云 GPU 推理实测 | E2/E3/E4 原始结果与图表入仓；README 量化对比表只引用实测值 |
| W8 | Spring AI BFF | 鉴权/限流/会话；MCP client 调通；Java RAG 对照（或按预案裁剪） |
| W9 | 前端 + 准确率追踪 | Streamlit trace 可视化；pred_ledger 回填与指标看板；端到端联调 |
| W10 | 收口 | README（架构图、复现命令、实测表）、3 分钟演示视频、测试与 CI |

每周结束必须有"可演示产物"；任一周不达标，先补达标再进入下一周，不借未来时间。

---

## 12. 简历映射（修订版，每个词都有实证）

| 简历关键词 | 项目实证 |
|---|---|
| vLLM/SGLang 部署调优 | E2–E4：vLLM AWQ/GPTQ/FP8 服务化、benchmark_serving 压测；SGLang RadixAttention 前缀复用对照；脚本与原始数据入仓 |
| PagedAttention / KV Cache | 并发吞吐曲线、KV blocks 随 `gpu_memory_utilization/max_model_len` 变化、FP8 KV cache 实验；机制解释 + 论文引用 |
| AWQ/GPTQ 量化 | 官方 AWQ checkpoint 与 GPTQModel 自量化同模型三方对比；本地 GGUF 精度矩阵单独成节、不混淆概念 |
| LangChain/LangGraph RAG | Planner + ReAct tool loop（bind_tools/ToolNode/回边）；RAG 摄入到检索全链路 |
| pgvector | 结构化/非结构化边界清晰；HNSW + 元数据过滤；recall@5 评测 |
| ReAct / Function Calling | 真实工具自主调用循环，非固定 DAG；JSON schema 校验与溯源 |
| 多工具 Agent 编排 / MCP | 三 MCP Server 被 Python 与 Java 双客户端跨语言调用 |
| Spring AI 企业级服务 | BFF：Security/限流/会话/MCP client/Advisor RAG/可观测 |

---

## 13. 风险与裁剪

| 风险 | 对策 |
|---|---|
| 业余时间不足 | 裁剪顺序：Java RAG 对照 → SGLang 仅做部署验证不做完整 E4 → L4 蒙特卡洛；核心（数据/ML/ReAct/本地双档/vLLM AWQ）不砍 |
| 云端 GPU 缺货/预算 | 选有货区域或换平台；AWQ 是必做项，FP8/GPTQ 可分批补做 |
| 模型中文/工具调用不稳 | 4B 起步而非 1.5B；严格 schema + 修复 + 熔断；评测集先行，质量不达标升 8B |
| 预测打不过赔率 | 这是正常结果，诚实呈现；项目价值在"概率校准 + LLM 可解释工作流 + 推理路由"，不夸大有盈利能力 |
| 数字诚信 | 所有图表由 `benchmarks/results/` 数据生成；无数据的格子写"未测"，禁止填估值（本方案中所有性能数字位置均为待测项） |

---

## 附录 A：相对 v1 的关键修正

1. 量化：删除"GGUF Q4_K_M 等价 AWQ""Q8_0 等同 BF16"的错误表述；本地与云端实验分离；模型更新为 Qwen3-4B/8B 及官方 AWQ/FP8 checkpoint。
2. 补 vLLM 代码中对 BF16 权重直接传 `quantization="fp8"` 的不严谨用法，改为官方预量化 checkpoint。
3. MCP：区分"IDE 编码助手的 MCP"与"项目运行时的 MCP"；推理不再套 MCP，统一 OpenAI 兼容协议 + 外置配置实现真正后端解耦；MCP 聚焦跨语言工具复用。
4. Agent：固定线性 DAG 改为 Planner + ReAct tool-calling 循环，与简历"ReAct/Function Calling"名副其实。
5. ML：补齐数据源、walk-forward、LogLoss/Brier/RPS、校准、赔率 baseline、投注回测、蒙特卡洛实现路径。
6. RAG：明确结构化/非结构化边界、bge-m3、chunk 策略、recall@k 评测、强制溯源。
7. Spring AI：从"重写 Python MCP Server"改为企业级 BFF + 跨语言 MCP client + Java RAG 对照。
8. 新增 Qwen3 思考模式关闭、超时/重试/熔断/JSON 修复等工程稳定性设计。
9. 排期：数据与 ML 基线从第 7–8 周前置到 W1–W3；每周设可验收交付物。
10. 数据纪律：README 所有性能/质量数字必须来自入仓实验产物。

## 附录 B：关键参考来源

- Qwen3 部署与预量化模型（Qwen3-8B-AWQ / Qwen3-8B-FP8、vLLM ≥0.8.5、思考模式开关）：https://qwen.readthedocs.io/zh-cn/stable/deployment/vllm.html
- Qwen3 量化速度基准（BF16/FP8/AWQ/GPTQ）：https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html
- Spring AI MCP Client Boot Starter：https://docs.spring.io/spring-ai/reference/api/mcp/mcp-client-boot-starter-docs.html
- Spring AI PgVectorStore（HNSW、维度限制 2000）：https://docs.spring.io/spring-ai/reference/api/vectordbs/pgvector.html
- vLLM 论文（PagedAttention）：https://arxiv.org/abs/2309.06180
- 足球数据（含 closing odds）：https://www.football-data.co.uk/
