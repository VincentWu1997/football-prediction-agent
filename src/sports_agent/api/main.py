"""FastAPI 网关（W4）。

- GET  /health            健康检查（不依赖外部服务）
- GET  /inference/status  fast/precise 双档后端探测（不触发推理）
- POST /analyze           分析请求；dry_run=true 只跑 Planner
- POST /predict           单场预测：DC 概率 + 蒙特卡洛比分仿真
- POST /simulate/season   L4 赛季剩余赛程蒙特卡洛（夺冠/前四/降级）

W6 计划：/analyze 升级为 ReAct 循环经 MCP 调用 /predict 背后的同一能力层。
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sports_agent.agent.graph import build_graph
from sports_agent.agent.planner import Planner
from sports_agent.inference.client import InferenceClient
from sports_agent.ml.predict_service import predict_match, simulate_season

app = FastAPI(title="sports-agent runtime", version="0.2.0")


class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="用户问题，如：详细分析曼联对利物浦")
    dry_run: bool = Field(False, description="True 时只返回 Planner 分层结果，不调用 LLM")


class PredictRequest(BaseModel):
    home: str = Field(..., examples=["Manchester United"])
    away: str = Field(..., examples=["Liverpool"])
    league: str | None = Field(None, description="省略时按两队共同联赛自动推断")
    as_of: str | None = Field(None, description="YYYY-MM-DD；默认用数据中最新日期")
    n_sims: int = Field(20_000, ge=1_000, le=200_000)


class SeasonSimRequest(BaseModel):
    league: str = Field(..., examples=["E0"])
    as_of: str = Field(..., description="赛季进行中的某天 YYYY-MM-DD，之后视为剩余赛程")
    n_runs: int = Field(20_000, ge=1_000, le=200_000)


_planner = Planner()
_inference = InferenceClient()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/inference/status")
def inference_status() -> dict:
    """fast/precise 双档探测；Ollama 未启动时 reachable=false，不报错。"""
    return _inference.status()


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    plan = _planner.plan(req.query)
    if req.dry_run:
        return {"query": req.query, "dry_run": True, "plan": plan.to_dict()}

    # 图在首次真实请求时惰性编译；Planner 仍用上面的单例以共享配置
    graph = build_graph(planner=_planner, inference=_inference)
    result = graph.invoke({"query": req.query})
    return {
        "query": req.query,
        "level": result.get("level"),
        "route": result.get("route"),
        "answer": result.get("final_answer"),
        "inference": result.get("inference"),
        "trace": result.get("trace", []),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        return predict_match(
            req.home, req.away, league=req.league, as_of=req.as_of, n_sims=req.n_sims
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/simulate/season")
def simulate_season_endpoint(req: SeasonSimRequest) -> dict:
    try:
        return simulate_season(req.league, req.as_of, n_runs=req.n_runs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
