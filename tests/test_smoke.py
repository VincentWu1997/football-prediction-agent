"""不依赖外部服务的冒烟测试：配置加载与 Planner 规则分层。"""

import pytest

from sports_agent.agent.planner import Planner
from sports_agent.inference.registry import ModelRegistry


@pytest.fixture(scope="module")
def planner() -> Planner:
    return Planner()


def test_registry_backends_and_levels() -> None:
    registry = ModelRegistry()
    fast = registry.backend("fast")
    assert fast.model.startswith("qwen3")
    assert registry.level("L1").route == "fast"
    assert registry.level("L3").route == "precise"
    assert registry.level("L4").allow_simulation is True
    # RAG 配置：bge-m3 1024 维，需与 db/init SQL 的 vector(1024) 一致
    assert registry.rag_config["dimensions"] == 1024


@pytest.mark.parametrize(
    "query,expected",
    [
        ("英超积分榜", "L1"),
        ("详细分析曼联对利物浦，考虑伤停", "L3"),
        ("模拟本赛季英超前四", "L4"),
        ("曼联对利物浦谁赢", "L2"),
    ],
)
def test_planner_routing(planner: Planner, query: str, expected: str) -> None:
    plan = planner.plan(query)
    assert plan.level == expected
    assert plan.max_tool_iters > 0
