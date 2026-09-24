"""LangGraph 编排：Planner + ReAct tool-calling 循环。

W6 实现（方案 §6.2）：
    入口 → planner → react(自主调工具) → END

ReAct 循环（react 节点内，使用 OpenAI 原生 tool_calls）：
    for i in 1..max_tool_iters:
        completion = llm.chat(route, messages, tools=schema_for(allowlist))
        if completion.tool_calls and iters_used < max_tool_iters:
            追加 assistant 消息（含 tool_calls）
            for tc in tool_calls:
                result = dispatch_tool(tc.name, tc.arguments)
                追加 tool 消息
                写 trace
            iters_used += 1
        else:
            final_answer = completion.content
            return
预算耗尽仍要调工具时，最后一轮不传 tools，强制模型基于已收集信息作答。

容错（§6.5）：
    - 工具调用失败：写 ToolMessage 标记失败原因，让 LLM 决定换路径
    - LLM 调用失败：返回错误消息，trace 中可见，不抛给上层
"""

from __future__ import annotations

import datetime
import json
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from sports_agent.agent.mcp_tools import load_tools, tool_schemas_for_openai
from sports_agent.agent.planner import Planner
from sports_agent.agent.state import AgentState, TraceStep
from sports_agent.inference.client import InferenceClient

_SYSTEM_PROMPT = (
    "你是足球赛事分析助手。只能基于工具返回的数据与知识库内容作答，"
    "不得编造伤停、比分或概率；引用事实时标注来源（doc_source 或比赛日期）。\n"
    "当前日期：{today}。数据库存有至 {latest_season_hint} 赛季的完整数据。\n"
    "当前任务层级：{level}，工具调用上限：{max_iters} 次。\n"
    "工具白名单：{allowlist}。\n"
    "强制规则：\n"
    "- 涉及单场胜负平预测（如'谁会赢'）时，必须先调用 predict_match 工具获取真实概率，再回答；\n"
    "- 调用 query_standings 时**省略 season 参数**（工具会自动使用最新赛季），"
    "除非用户明确要求历史赛季；禁止凭记忆猜赛季代码；\n"
    "- 需要伤停/战术/采访等非结构化信息时调用 search_knowledge；\n"
    "- L4 赛季模拟才允许 simulate_season；\n"
    "- 积分榜/战绩/赔率走 query_standings / query_recent_form / query_h2h / query_odds。\n"
    "作答时用中文，结构清晰；如果工具数据不足，明确说'信息不足'，不要补编。"
)

# 按层级的工具白名单（§2.2 任务分层）
_LEVEL_TOOLS: dict[str, list[str]] = {
    "L1": ["query_standings", "query_recent_form", "query_h2h", "query_odds"],
    "L2": [
        "query_standings",
        "query_recent_form",
        "query_h2h",
        "query_odds",
        "predict_match",
    ],
    "L3": [
        "query_standings",
        "query_recent_form",
        "query_h2h",
        "query_odds",
        "predict_match",
        "search_knowledge",
    ],
    "L4": [
        "query_standings",
        "query_recent_form",
        "query_h2h",
        "query_odds",
        "predict_match",
        "search_knowledge",
        "simulate_season",
    ],
}


def _season_hint() -> str:
    """从库中取最新赛季代码注入 prompt；DB 不可用时给通用提示。"""
    try:
        from sports_agent.data.queries import latest_season

        s = latest_season("E0")
        return s or "最新"
    except Exception:
        return "最新"


def _build_registry() -> tuple[Planner, InferenceClient, dict]:
    """构造图编译期对象；Tools 字典：name -> StructuredTool。"""
    planner = Planner()
    inference = InferenceClient()
    tools_list = load_tools(mode="local")
    tools_by_name = {t.name: t for t in tools_list}
    return planner, inference, tools_by_name


def _dispatch_tool(tools: dict, name: str, args_json: str) -> tuple[Any, int, str | None]:
    """执行单个 tool_call；返回 (result, latency_ms, error_msg)。

    出错时不抛，返回 error 让 LLM 决定换路径。
    """
    start = time.perf_counter()

    def _lat() -> int:
        return int((time.perf_counter() - start) * 1000)

    try:
        if name not in tools:
            return {"error": f"未注册工具 {name}（不在白名单内）"}, _lat(), "unknown_tool"
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as e:
            return {"error": f"参数 JSON 解析失败: {e}"}, _lat(), "bad_args"
        result = tools[name].invoke(args)
        return result, _lat(), None
    except Exception as e:
        return {"error": str(e)[:200]}, _lat(), type(e).__name__


def _build_react_node(planner, inference, tools_by_name):
    """返回 react 节点函数。"""

    def react_node(state: AgentState) -> AgentState:
        # 初始化消息
        messages: list[dict] = state.get("messages") or []
        if not messages:
            allowlist = _LEVEL_TOOLS.get(state["level"], [])
            messages = [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT.format(
                        today=datetime.date.today().isoformat(),
                        latest_season_hint=_season_hint(),
                        level=state["level"],
                        max_iters=state["max_tool_iters"],
                        allowlist=", ".join(allowlist),
                    ),
                },
                {"role": "user", "content": state["query"]},
            ]
        trace: list[TraceStep] = state.get("trace", [])
        iters_used: int = state.get("iters_used", 0)
        max_iters: int = state["max_tool_iters"]
        allowlist = _LEVEL_TOOLS.get(state["level"], [])
        # 当前轮可见工具（按白名单裁剪）
        active_tools = [tools_by_name[n] for n in allowlist if n in tools_by_name]
        schemas = tool_schemas_for_openai(active_tools)

        sources: list[dict] = []
        inference_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "n_calls": 0,
        }
        final_answer: str | None = None

        for step_idx in range(max_iters + 1):  # +1 让最后一轮可不传 tools 收尾
            budget_left = max_iters - iters_used
            # 最后一轮（预算耗尽）：不再传 tools，逼模型作答
            use_tools = budget_left > 0 and step_idx < max_iters
            try:
                result = inference.chat(
                    state["route"],
                    messages,
                    tools=schemas if use_tools else None,
                    max_retries=1,  # ReAct 循环里只重试一次，避免长超时拖死整轮
                )
            except Exception as e:
                trace.append(
                    TraceStep(
                        kind="llm",
                        name=state["route"],
                        latency_ms=0,
                        detail={"error": f"{type(e).__name__}: {str(e)[:200]}"},
                    )
                )
                final_answer = f"[推理失败] {type(e).__name__}: {str(e)[:200]}"
                break

            inference_total["prompt_tokens"] += result.prompt_tokens
            inference_total["completion_tokens"] += result.completion_tokens
            inference_total["latency_ms"] += result.latency_ms
            inference_total["n_calls"] += 1
            trace.append(
                TraceStep(
                    kind="llm",
                    name=result.model,
                    latency_ms=result.latency_ms,
                    detail={
                        "step": step_idx,
                        "finish_reason": result.finish_reason,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "tool_calls": [tc["name"] for tc in (result.tool_calls or [])],
                    },
                )
            )

            # 追加 assistant 消息（保留 tool_calls 给下一条 tool 消息配对）
            assistant_msg: dict = {"role": "assistant", "content": result.content or ""}
            if result.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in result.tool_calls
                ]
            messages.append(assistant_msg)

            if not result.tool_calls or not use_tools:
                # 模型主动停止 OR 已无预算：取 content 作终答
                final_answer = result.content or ""
                break

            # 执行 tool_calls
            for tc in result.tool_calls:
                tool_result, t_lat, err = _dispatch_tool(tools_by_name, tc["name"], tc["arguments"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                    }
                )
                trace.append(
                    TraceStep(
                        kind="tool",
                        name=tc["name"],
                        latency_ms=t_lat,
                        detail={
                            "args": _safe_short(tc["arguments"]),
                            "result_keys": (
                                list(tool_result.keys()) if isinstance(tool_result, dict) else None
                            ),
                            "error": err,
                        },
                    )
                )
                # 抽取来源（doc_source 或比赛信息）
                _collect_sources(sources, tc["name"], tool_result)
            iters_used += 1

        if final_answer is None:
            final_answer = "[Agent] 已达工具调用预算上限，未能给出最终答案。"

        return {
            "messages": messages,
            "iters_used": iters_used,
            "final_answer": final_answer,
            "sources": sources,
            "trace": trace,
            "inference": inference_total,
        }

    return react_node


def _safe_short(s: str, n: int = 200) -> str:
    """截断长字符串用于 trace 展示。"""
    if not s:
        return ""
    return s if len(s) <= n else s[:n] + "..."


def _collect_sources(sources: list[dict], tool_name: str, result: Any) -> None:
    """从工具结果中抽取溯源锚点。"""
    if not isinstance(result, dict):
        return
    if tool_name == "search_knowledge":
        items = result if isinstance(result, list) else result.get("matches", [])
        for c in items:
            if isinstance(c, dict) and c.get("doc_source"):
                sources.append(
                    {
                        "type": "knowledge",
                        "doc_source": c["doc_source"],
                        "score": c.get("score"),
                    }
                )
    elif tool_name == "predict_match":
        sources.append(
            {
                "type": "model",
                "model": result.get("model"),
                "as_of": result.get("as_of"),
            }
        )
    elif tool_name == "query_odds" and result.get("found"):
        sources.append(
            {
                "type": "odds",
                "date": result.get("date"),
                "league": result.get("league"),
            }
        )
    elif tool_name == "query_h2h":
        for m in result.get("matches", [])[:3]:
            sources.append({"type": "h2h", "date": m.get("date")})


def build_graph(
    planner: Planner | None = None,
    inference: InferenceClient | None = None,
    tools_by_name: dict | None = None,
):
    """编译 Agent 状态图。

    Args:
        planner: 自定义 Planner；不传则新建。
        inference: 自定义 InferenceClient；不传则新建。
        tools_by_name: 自定义工具字典；不传则用 load_tools(local) 默认集。

    dry_run（不进 ReAct）请用 Planner.plan() 直接调用，不走本图。
    """
    if planner is None or inference is None or tools_by_name is None:
        default_planner, default_inference, default_tools = _build_registry()
        planner = planner or default_planner
        inference = inference or default_inference
        tools_by_name = tools_by_name or default_tools

    def plan_node(state: AgentState) -> AgentState:
        plan = planner.plan(state["query"])
        trace: list[TraceStep] = state.get("trace", [])
        # §6.5 路由降级：precise 模型不可达时退到 fast（保留 level 以反映真实任务复杂度）
        route = plan.route
        downgrade_reason: str | None = None
        if route == "precise":
            try:
                avail = inference.available_routes()
                if "precise" not in avail:
                    route = "fast"
                    downgrade_reason = (
                        f"precise 不可用，降级到 fast（available={sorted(avail) or 'none'})"
                    )
            except Exception as e:  # 探测失败不阻塞，继续用 precise 尝试
                downgrade_reason = f"available_routes 探测异常: {type(e).__name__}"
        detail = plan.to_dict()
        if downgrade_reason:
            detail["route_final"] = route
            detail["downgrade_reason"] = downgrade_reason
        trace.append(
            TraceStep(
                kind="planner",
                name="rule_planner",
                latency_ms=getattr(planner, "_last_latency_ms", 0),
                detail=detail,
            )
        )
        return {
            "level": plan.level,
            "route": route,
            "max_tool_iters": plan.max_tool_iters,
            "allow_simulation": plan.allow_simulation,
            "tools_available": _LEVEL_TOOLS.get(plan.level, []),
            "trace": trace,
        }

    react_node = _build_react_node(planner, inference, tools_by_name)

    graph = StateGraph(AgentState)
    graph.add_node("planner", plan_node)
    graph.add_node("react", react_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "react")
    graph.add_edge("react", END)
    return graph.compile()
