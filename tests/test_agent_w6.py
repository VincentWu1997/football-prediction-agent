"""W6 烟测：data tools + planner 路由 + graph 编译 + mcp_tools 双模式。

不依赖 Ollama / 不依赖运行中的 MCP Server：
- data queries 直接打 PostgreSQL（sports-agent-pg 容器）；
- planner 跑规则版（不调 LLM）；
- mcp_tools local 模式加载 7 个工具；
- graph 仅 compile 不 invoke（invoke 需 Ollama，留给 experiment_w6）。
"""

import json
import os

import pytest

# 数据库不可用时跳过 data 工具测试
HAS_DB = os.environ.get("SKIP_DB_TESTS") is None

skip_no_db = pytest.mark.skipif(not HAS_DB, reason="需要 PostgreSQL（sports-agent-pg 容器）")


@skip_no_db
def test_query_standings_top_of_e0_2425() -> None:
    from sports_agent.data.queries import query_standings

    rows = query_standings("E0", "2425")
    assert len(rows) >= 10  # 英超至少 10 队
    top = rows[0]
    assert top["rank"] == 1
    assert top["points"] >= 70
    assert top["played"] == 38
    assert set(top.keys()) == {
        "rank",
        "team",
        "played",
        "won",
        "drawn",
        "lost",
        "gf",
        "ga",
        "gd",
        "points",
    }


@skip_no_db
def test_query_recent_form_liverpool() -> None:
    from sports_agent.data.queries import query_recent_form

    rows = query_recent_form("Liverpool", n=5)
    assert len(rows) == 5
    for r in rows:
        assert r["venue"] in ("H", "A")
        assert r["result"] in ("H", "D", "A")
        assert isinstance(r["gf"], int)


@skip_no_db
def test_query_h2h_man_united_vs_liverpool() -> None:
    from sports_agent.data.queries import query_h2h

    rows = query_h2h("Man United", "Liverpool", n=5)
    assert len(rows) > 0
    for r in rows:
        assert r["home"] in ("Man United", "Liverpool")
        assert r["away"] in ("Man United", "Liverpool")


@skip_no_db
def test_query_odds_found_and_not_found() -> None:
    from sports_agent.data.queries import query_odds

    found = query_odds("Man United", "Liverpool")
    assert found["found"] is True
    assert found["bet365"]["h"] > 1.0
    # 皇马对曼联（不同联赛，应在 DB 中无共同交手）
    not_found = query_odds("Man United", "Real Madrid")
    assert not_found["found"] is False


def test_planner_rule_all_levels() -> None:
    from sports_agent.agent.planner import Planner

    p = Planner()
    cases = [
        ("英超积分榜", "L1"),
        ("英超最近比赛几点开始", "L1"),
        ("Man United对Liverpool谁会赢", "L2"),
        ("详细分析Arsenal对Chelsea，考虑伤停", "L3"),
        ("模拟本赛季英超前四", "L4"),
        ("按剩余赛程模拟降级", "L4"),
    ]
    for q, expected in cases:
        plan = p.plan(q)
        assert plan.level == expected, (
            f"query={q} expected={expected} got={plan.level} reason={plan.reason}"
        )
        assert plan.route in ("fast", "precise")
        assert plan.max_tool_iters > 0


def test_local_tools_loaded_seven() -> None:
    from sports_agent.agent.mcp_tools import local_tools, tool_schemas_for_openai

    tools = local_tools()
    assert len(tools) == 7
    names = {t.name for t in tools}
    expected = {
        "query_standings",
        "query_recent_form",
        "query_h2h",
        "query_odds",
        "predict_match",
        "simulate_season",
        "search_knowledge",
    }
    assert names == expected
    schemas = tool_schemas_for_openai(tools)
    assert len(schemas) == 7
    assert all(s["type"] == "function" for s in schemas)
    assert all("name" in s["function"] for s in schemas)


def test_local_tool_invokes_query_recent_form() -> None:
    """local 模式工具能被 LangChain 调用（输入 dict，输出 dict/list）。"""
    if not HAS_DB:
        pytest.skip("需要 PostgreSQL")
    from sports_agent.agent.mcp_tools import local_tools

    tools = {t.name: t for t in local_tools()}
    out = tools["query_recent_form"].invoke({"team": "Arsenal", "n": 3})
    assert out["n_returned"] == 3
    assert all("opponent" in m for m in out["matches"])


def test_graph_compiles() -> None:
    """图能编译，不调用 Ollama。"""
    from sports_agent.agent.graph import build_graph

    g = build_graph()
    assert g is not None
    # 节点应至少包含 planner / react
    nodes = getattr(g, "nodes", {})
    assert any("planner" in str(n) for n in nodes) or "planner" in nodes


def test_routing_eval_set_balanced_100() -> None:
    """路由评测集 100 条，L1-L4 各 25。"""
    from collections import Counter

    from sports_agent.settings import REPO_ROOT

    path = REPO_ROOT / "benchmarks" / "eval_sets" / "routing.jsonl"
    if not path.exists():
        pytest.skip("routing.jsonl 未生成（先运行 data/scripts/build_routing_eval.py）")
    pairs = [json.loads(line) for line in path.open(encoding="utf-8")]
    assert len(pairs) == 100
    counts = Counter(p["level"] for p in pairs)
    for lv in ("L1", "L2", "L3", "L4"):
        assert counts[lv] == 25
