"""统一工具加载：将三个 MCP Server 的工具暴露为 LangChain BaseTool。

设计（方案 §6.3）：
- 生产模式 (mode="mcp")：通过 fastmcp Client 以 Streamable HTTP 连接
  data(8001) / prediction(8002) / knowledge(8003) 三个 Server，跨进程跨语言；
- 本地/测试模式 (mode="local")：直接 import Python 能力层同进程调用，免去启 3 个 Server。
两种模式返回的 BaseTool 同名同 schema，ReAct 循环与 ToolNode 无感知切换。

W8 Spring AI 走同一组 MCP Server，跨语言价值闭环。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

# 三 MCP Server 的默认 HTTP 端点
DEFAULT_MCP_URLS = {
    "data": "http://localhost:8001/mcp",
    "prediction": "http://localhost:8002/mcp",
    "knowledge": "http://localhost:8003/mcp",
}


# ============ 本地直调实现（mode="local"，ReAct/测试共用） ============


def _local_query_standings(league: str, season: str | None = None) -> dict:
    from sports_agent.data.queries import latest_season, query_standings

    resolved = season or latest_season(league)
    rows = query_standings(league, resolved)
    return {"league": league, "season": resolved, "n_teams": len(rows), "standings": rows}


def _local_query_recent_form(team: str, n: int = 5) -> dict:
    from sports_agent.data.queries import query_recent_form

    n = max(1, min(int(n), 30))
    rows = query_recent_form(team, n=n)
    return {"team": team, "n_requested": n, "n_returned": len(rows), "matches": rows}


def _local_query_h2h(team_a: str, team_b: str, n: int = 10) -> dict:
    from sports_agent.data.queries import query_h2h

    n = max(1, min(int(n), 50))
    rows = query_h2h(team_a, team_b, n=n)
    return {
        "team_a": team_a,
        "team_b": team_b,
        "n_requested": n,
        "n_returned": len(rows),
        "matches": rows,
    }


def _local_query_odds(
    home: str, away: str, league: str | None = None, season: str | None = None
) -> dict:
    from sports_agent.data.queries import query_odds

    return query_odds(home, away, league=league, season=season)


def _local_predict_match(
    home: str, away: str, league: str | None = None, n_sims: int = 20_000
) -> dict:
    from sports_agent.ml.predict_service import predict_match

    return predict_match(home, away, league=league, n_sims=n_sims)


def _local_simulate_season(league: str, as_of: str, n_runs: int = 20_000) -> dict:
    from sports_agent.ml.predict_service import simulate_season

    return simulate_season(league, as_of, n_runs=n_runs)


def _local_search_knowledge(
    query: str, k: int = 5, league: str | None = None, team: str | None = None
) -> list[dict]:
    from sports_agent.rag.retrieve import search_knowledge

    chunks = search_knowledge(query, k=k, league=league, team=team)
    return [
        {
            "content": c.content,
            "doc_source": c.doc_source,
            "doc_type": c.doc_type,
            "league": c.league,
            "team": c.team,
            "doc_date": c.doc_date,
            "score": round(c.score, 4),
        }
        for c in chunks
    ]


# ============ 工具元数据：name / description / 入参 schema ============

# 每条 = (tool_name, callable_local, description, args_schema_dict)
# args_schema 用 JSON Schema dict（OpenAI tools 格式可直接转），StructuredTool 由 pydantic 推断
_TOOL_SPECS: list[tuple[str, Callable, str, type]] = []


def _spec(name: str, fn: Callable, desc: str):
    _TOOL_SPECS.append((name, fn, desc, None))
    return fn


_spec(
    "query_standings",
    _local_query_standings,
    "查询某联赛某赛季积分榜。league 为 football-data.co.uk 代码（E0/SP1/D1/I1/F1），"
    "season 为 4 位起始年份+2 位终止年份（如 2425），省略时自动使用最新赛季（推荐）。"
    "返回 {rank, team, played, won, drawn, lost, gf, ga, gd, points} 列表。",
)
_spec(
    "query_recent_form",
    _local_query_recent_form,
    "查询某队最近 N 场战绩（含主客，按日期倒序）。"
    "team 为官方队名（如 Manchester United），n 默认 5。",
)
_spec(
    "query_h2h",
    _local_query_h2h,
    "查询两队近 N 次交手记录（任一主客组合，按日期倒序）。"
    "team_a/team_b 为官方队名，n 默认 10。",
)
_spec(
    "query_odds",
    _local_query_odds,
    "查询两队最近一次交锋的临场赔率（Bet365 + Pinnacle + 市场平均）。"
    "无记录返回 {found: false}。",
)
_spec(
    "predict_match",
    _local_predict_match,
    "预测单场胜负平：基于 Dixon-Coles 概率 + 蒙特卡洛比分仿真。"
    "home/away 为官方队名，league 可省略（自动推断），n_sims 默认 20000。"
    "返回 analytic_probs / monte_carlo 概率与置信区间。",
)
_spec(
    "simulate_season",
    _local_simulate_season,
    "L4：对某联赛赛季剩余赛程做蒙特卡洛（夺冠/前四/降级概率）。"
    "league 为代码，as_of 为 YYYY-MM-DD（赛季进行中某天），n_runs 默认 20000。",
)
_spec(
    "search_knowledge",
    _local_search_knowledge,
    "检索伤停新闻/战术报告/赛前采访（RAG：bge-m3 + pgvector HNSW）。"
    "返回文本块列表，每条含 doc_source（强制溯源）。league/team 可选过滤。",
)


# ============ Public API ============


def local_tools() -> list[StructuredTool]:
    """本地直调模式：3 个 Server 的 7 个工具全部走 Python import 同进程执行。"""
    tools: list[StructuredTool] = []
    for name, fn, desc, _ in _TOOL_SPECS:
        tools.append(StructuredTool.from_function(func=fn, name=name, description=desc))
    return tools


def _run_async(coro):
    """在同步上下文跑 async fastmcp Client。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已有事件循环（如 asyncio.run 内）→ 用 nest_asyncio 风格不可靠，直接 new loop
            import asyncio as _a

            new_loop = _a.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
    except RuntimeError:
        pass
    return asyncio.run(coro)


async def _mcp_call(server_url: str, tool_name: str, **args) -> Any:
    """以 fastmcp Client 调用远端 MCP Server 的 tool。"""
    from fastmcp import Client

    async with Client(server_url) as client:
        result = await client.call_tool(tool_name, args)
        # call_tool 返回 StructuredContent / list[Content]，统一抽取为 dict/list
        if hasattr(result, "structured_content") and result.structured_content is not None:
            return result.structured_content
        if isinstance(result, list) and result and hasattr(result[0], "text"):
            # 文本内容：尝试 json.loads 还原结构
            try:
                return json.loads(result[0].text)
            except (json.JSONDecodeError, TypeError):
                return [c.text for c in result if hasattr(c, "text")]
        if isinstance(result, list):
            return result
        return result


def _make_mcp_tool(server_key: str, server_url: str, tool_name: str, description: str):
    """构造一个调用远端 MCP tool 的 StructuredTool。

    入参 schema 从 _TOOL_SPECS 中同名本地函数推断，保持双模式签名一致。
    """
    local_spec = next((s for s in _TOOL_SPECS if s[0] == tool_name), None)
    if local_spec is None:
        raise ValueError(f"未注册工具: {tool_name}")
    _, local_fn, local_desc, _ = local_spec
    import inspect

    sig = inspect.signature(local_fn)

    def _call(**kwargs):
        return _run_async(_mcp_call(server_url, tool_name, **kwargs))

    # 复用本地函数签名，便于 StructuredTool 推断 args_schema
    _call.__signature__ = sig  # type: ignore[attr-defined]
    _call.__doc__ = description or local_desc
    return StructuredTool.from_function(
        func=_call, name=tool_name, description=description or local_desc
    )


def mcp_tools(urls: dict[str, str] | None = None, discover: bool = False) -> list[StructuredTool]:
    """MCP HTTP 模式：返回 7 个工具的远端调用包装。

    Args:
        urls: 三 Server 的端点（data/prediction/knowledge）。不传用 DEFAULT_MCP_URLS。
        discover: True 时先 list_tools 探测各 Server 真实工具集，仅注册可见工具；
                  False（默认）直接按 _TOOL_SPECS 静态注册，避免每次启动都阻塞探测。

    返回的工具 name/description 与 local_tools() 一致，ReAct 循环无感知。
    """
    urls = urls or DEFAULT_MCP_URLS
    # 工具 → 所在 Server 的映射
    tool_to_server = {
        "query_standings": "data",
        "query_recent_form": "data",
        "query_h2h": "data",
        "query_odds": "data",
        "predict_match": "prediction",
        "simulate_season": "prediction",
        "search_knowledge": "knowledge",
    }
    available: set[str] | None = None
    if discover:
        available = set()
        for server_key, url in urls.items():
            try:

                async def _list(u):
                    from fastmcp import Client

                    async with Client(u) as c:
                        return await c.list_tools()

                tools_list = _run_async(_list(url))
                available.update(t.name for t in tools_list)
            except Exception as e:
                print(f"[WARN] MCP Server {server_key} ({url}) 不可达: {e}")
        if not available:
            raise RuntimeError("所有 MCP Server 不可达，请先启动 data/prediction/knowledge")

    tools: list[StructuredTool] = []
    for name, _, desc, _ in _TOOL_SPECS:
        if available is not None and name not in available:
            continue
        server_key = tool_to_server.get(name)
        if server_key is None or server_key not in urls:
            continue
        # 给描述加 MCP 来源前缀，便于 trace 区分
        full_desc = f"[MCP:{server_key}] {desc}"
        tools.append(_make_mcp_tool(server_key, urls[server_key], name, full_desc))
    return tools


def load_tools(mode: str = "local", **kwargs) -> list[StructuredTool]:
    """根据模式加载工具集。

    mode="local"（默认）：同进程直调，无需启动 Server；适合测试 / 单机演示。
    mode="mcp"：通过 HTTP 调用远端 MCP Server，跨进程跨语言；需先 `fastmcp run` 3 个 Server。
    """
    if mode == "local":
        return local_tools()
    if mode == "mcp":
        return mcp_tools(**kwargs)
    raise ValueError(f"未知工具加载模式: {mode}（支持 local / mcp）")


def tool_schemas_for_openai(tools: list[StructuredTool]) -> list[dict]:
    """将 StructuredTool 转为 OpenAI /v1/chat/completions 的 tools 字段格式。

    ReAct 循环手动跑（不依赖 langchain_openai.ChatModel.bind_tools），
    故需要把 LangChain tool 的 args_schema 转成 OpenAI function spec。
    """
    out: list[dict] = []
    for t in tools:
        schema = (
            t.args_schema.model_json_schema()
            if t.args_schema
            else {"type": "object", "properties": {}}
        )
        # 去掉 pydantic 加的 title 字段（OpenAI 不需要，且会污染 prompt）
        props = schema.get("properties", {})
        for _k, v in props.items():
            v.pop("title", None)
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": schema.get("required", []),
                    },
                },
            }
        )
    return out
