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
- JDK 17+、Maven（仅 spring-service）

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

### Spring AI BFF（W8，最小版）

先启动上面三个 MCP Server（BFF 启动时会初始化连接），再：

```bash
cd spring-service && mvn spring-boot:run
# http://localhost:8080/api/health （basic auth: demo / demo123）
# http://localhost:8080/api/tools  → 列出经 MCP 发现的全部工具
```

## 数据纪律（README 数字规则）

README 与简历中所有性能/质量数字必须来自 `benchmarks/results/` 中的实验产物（原始 CSV/JSON + 生成图表的脚本），禁止手填目标值；没有数据的位置写"未测"。引用论文数字须标注来源。

## 十周路线图（12–15h/周）

- [x] W1 数据管道 + EDA（19,763 场已入库，详见 `data/processed/eda_report.txt`）
- [x] W2 walk-forward 脚手架 + 赔率去水 baseline（LogLoss/Brier/RPS，详见 `benchmarks/results/w2/`）
- [x] W3 XGBoost 主模型 + 温度/isotonic 校准 + flat-stake 投注回测（详见 `benchmarks/results/w3/`）
- [x] W4 蒙特卡洛 + 本地推理双档 + FastAPI（详见 `benchmarks/results/w4/`）
- [ ] W5 bge-m3 + pgvector RAG（recall@5 / 溯源）
- [ ] W6 三个 MCP Server + Planner + ReAct 端到端
- [ ] W7 云 GPU 实测 vLLM AWQ/GPTQ/FP8 + SGLang RadixAttention（预算 ¥50）
- [ ] W8 Spring AI 最小 BFF（鉴权 + MCP client 调通）
- [ ] W9 Streamlit trace 可视化 + 预测账本看板
- [ ] W10 README 实测数据、演示视频、测试收口

## 目录结构

```
config/models.yaml        推理后端注册表与层级预算
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
