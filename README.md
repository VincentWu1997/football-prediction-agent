# 推理感知型体育预测 Agent (sports-agent)

根据任务复杂度动态分配推理预算的足球赛事预测 Agent：L1/L2 走 Qwen3-4B 轻量量化模型，L3/L4 走 Qwen3-8B 强模型 + ReAct 多工具循环。预测由可回测的统计/ML 模型给出（ELO / Dixon-Coles / XGBoost），LLM 负责信息聚合与解释；知识层用 bge-m3 + pgvector 做带溯源的 RAG；工具能力通过 MCP 协议同时服务于 Python Agent 与 Java 企业服务。

完整设计见 [推理感知型体育预测Agent方案_v2.md](./推理感知型体育预测Agent方案_v2.md)。

## 架构一览

```
Streamlit ── Spring AI BFF(最小版) ── FastAPI Agent Runtime
                                        │
   Planner(L1-L4+预算) → ReAct Tool Loop (LangGraph)
                                        │
        MCP: 数据查询 │ MCP: 预测 │ MCP: 知识检索
                                        │
              PostgreSQL 16 + pgvector
   统一推理客户端(OpenAI 兼容): Ollama(本地) / vLLM·SGLang(云端)
```

## 环境要求

- Python 3.11+
- Docker（仅用于 postgres+pgvector）
- [Ollama](https://ollama.com)（macOS 原生，Metal 加速）
- JDK 21、Maven 3.9（仅 spring-service）

## 快速开始

```bash
# 1) Python 虚拟环境与依赖（按阶段装 extras；全量可一次性装 ml,mcp,ui）
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                       # 骨架阶段核心依赖
# pip install -e ".[ml,mcp,ui,dev]"          # W3 起

# 2) PostgreSQL + pgvector
cp .env.example .env
docker compose up -d                          # 等到 healthy：docker compose ps

# 3) Ollama 模型（W4 前需要；bge-m3 在 W5 RAG 时拉）
ollama pull qwen3:4b
ollama pull qwen3:8b
# ollama pull bge-m3

# 4) W1：下载历史数据（football-data.co.uk，含 closing odds）
python data/scripts/fetch_csv.py --leagues E0 SP1 D1 I1 F1 --seasons 2324 2425

# 5) Agent API（:9000）
uvicorn sports_agent.api.main:app --reload --port 9000
curl -s http://localhost:9000/health
curl -s -X POST http://localhost:9000/analyze -H 'Content-Type: application/json' \
  -d '{"query":"英超积分榜","dry_run":true}'   # 只看 Planner 分层，不调用 LLM

# 6) 冒烟测试
pytest -q
```

### MCP Servers（W6）

```bash
pip install -e ".[mcp]"
fastmcp run mcp_servers/data_server.py       --transport http --port 8001
fastmcp run mcp_servers/prediction_server.py --transport http --port 8002
fastmcp run mcp_servers/knowledge_server.py  --transport http --port 8003
```

### W6 验证（data 工具 / Planner / ReAct）

```bash
# 1) 路由评测集 + 规则版/LLM 版分类准确率 + L1-L4 端到端 ReAct
python data/scripts/build_routing_eval.py            # 生成 100 条 routing.jsonl
python -m sports_agent.eval.experiment_w6             # 产物：benchmarks/results/w6/

# 2) data_server 四个 SQL 工具直调（不启动 MCP 进程，直查 PostgreSQL）
python -c "from sports_agent.data.queries import query_standings; print(query_standings('E0','2425')[:3])"

# 3) 单元测试（不依赖 Ollama；DB 不可用时设 SKIP_DB_TESTS=1 跳过 data 工具用例）
pytest tests/test_agent_w6.py -q
```

### Spring AI BFF（W8）

Spring Boot 3.4 + Spring AI 1.0 + JDK 21，最小版企业服务层。

```bash
cd spring-service && mvn spring-boot:run
# 健康检查（公开）：curl http://localhost:8080/api/health
# 工具列表（需 API Key）：curl -H 'X-API-Key: demo-key-001' http://localhost:8080/api/tools
# 预测账本：curl -H 'X-API-Key: demo-key-001' 'http://localhost:8080/api/predictions?summary=true'
# 分析透传：curl -X POST -H 'X-API-Key: demo-key-001' -H 'Content-Type: application/json' \
#   -d '{"query":"英超积分榜"}' http://localhost:8080/api/analyze
# Actuator：curl http://localhost:8080/actuator/health
```

| 模块 | 实现 |
|---|---|
| 对外 API | `/api/analyze`（透传 FastAPI）、`/api/predict`、`/api/predictions`（pred_ledger）、`/api/tools`（MCP） |
| 鉴权 | API Key（`X-API-Key` 头），Stateless |
| 限流 | Bucket4j 令牌桶，20 req/min/IP |
| MCP Client | `spring-ai-starter-mcp-client` Streamable HTTP 连三个 Python Server |
| 可观测 | Actuator（health/metrics），HikariPool 连 PostgreSQL |

## 数据纪律（README 数字规则）

README 与简历中所有性能/质量数字必须来自 `benchmarks/results/` 中的实验产物（原始 CSV/JSON + 生成图表的脚本），禁止手填目标值；没有数据的位置写"未测"。引用论文数字须标注来源。

## 十周路线图（12–15h/周）

- [x] W1 数据管道 + EDA（19,763 场已入库，详见 `data/processed/eda_report.txt`）
- [x] W2 walk-forward 脚手架 + 赔率去水 baseline（LogLoss/Brier/RPS，详见 `benchmarks/results/w2/`）
- [x] W3 XGBoost 主模型 + 温度/isotonic 校准 + flat-stake 投注回测（详见 `benchmarks/results/w3/`）
- [x] W4 蒙特卡洛 + 本地推理双档 + FastAPI（详见 `benchmarks/results/w4/`）
- [x] W5 bge-m3 + pgvector RAG（详见 `benchmarks/results/w5/`）（recall@5 / 溯源）
- [x] W6 三个 MCP Server + Planner + ReAct 端到端（详见 `benchmarks/results/w6/`）（规则版路由准确率、混淆矩阵、L1-L4 端到端 trace）
- [ ] W7 云 GPU 实测 vLLM AWQ/GPTQ/FP8 + SGLang RadixAttention（预算 ¥50）
- [x] W8 Spring AI BFF（鉴权 + 限流 + MCP client + 可观测，5 项烟测通过）
- [x] W9 Streamlit trace 可视化 + pred_ledger 回填与看板
- [ ] W10 README 实测数据、演示视频、测试收口

## 目录结构

```
config/models.yaml        推理后端注册表与层级预算
                          （Qwen3 默认思考模式，需 max_tokens≥2048 让其思考完后输出答案/tool_calls）
db/init/                  PostgreSQL 初始化 SQL
data/{raw,processed,master,scripts}   数据域
src/sports_agent/
  inference/  OpenAI 兼容客户端（Qwen3 关闭思考）
  agent/      state / planner / graph
  data/ ml/ eval/ rag/ api/
mcp_servers/  data / prediction / knowledge 三个 MCP Server
benchmarks/   评测集、实验脚本、results 原始数据
spring-service/  Spring AI 最小 BFF
frontend/     Streamlit
```
