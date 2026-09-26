"""W4 API 与预测服务集成测试（不依赖 Ollama）。"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sports_agent.api.main import app

# predict 相关端点读 data/processed/matches.csv；该文件被 gitignore，
# CI 干净 checkout 中不存在，此时只跳过依赖数据的用例。
HAS_DATA = (Path(__file__).parent.parent / "data/processed/matches.csv").exists()
skip_no_data = pytest.mark.skipif(not HAS_DATA, reason="需要 data/processed/matches.csv")

client = TestClient(app)


def test_health_and_dry_run() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/analyze", json={"query": "详细分析曼联对利物浦", "dry_run": True})
    body = r.json()
    assert r.status_code == 200
    assert body["plan"]["level"] == "L3"


def test_inference_status_reports_unreachable_gracefully() -> None:
    r = client.get("/inference/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"fast", "precise"}
    for route in body.values():
        assert isinstance(route["reachable"], bool)


@skip_no_data
def test_predict_match_real_data() -> None:
    r = client.post(
        "/predict",
        json={"home": "Manchester United", "away": "Liverpool", "n_sims": 5_000},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["league"] == "E0"
    probs = body["analytic_probs"]
    # 返回值按 4 位小数四舍五入，和的偏差上限约 1.5e-4
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    mc = body["monte_carlo"]
    assert abs(sum(mc["probs"].values()) - 1.0) < 1e-3
    assert len(mc["top_scores"]) == 5


@skip_no_data
def test_predict_unknown_team_returns_422() -> None:
    r = client.post("/predict", json={"home": "不存在的队", "away": "Liverpool"})
    assert r.status_code == 422


@skip_no_data
def test_simulate_season_real_data() -> None:
    r = client.post(
        "/simulate/season", json={"league": "E0", "as_of": "2026-01-10", "n_runs": 2_000}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["season"] == "2526"
    assert body["n_remaining"] > 0
    assert abs(sum(body["title"].values()) - 1.0) < 1e-6
    assert len(body["actual_final_order"]) == 20  # 英超 20 队
    # 冠军概率最高的应是真实前四球队之一（粗校验）
    top_team = max(body["title"], key=body["title"].get)
    assert top_team in body["actual_final_order"][:4]
