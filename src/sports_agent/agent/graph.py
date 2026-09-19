"""LangGraph 编排。

当前（骨架，W4）：planner -> answer 两节点，answer 直接调用对应档位模型。
目标（W6）：answer 节点替换为 ReAct 循环：
    bind_tools(模型) -> 有 tool_calls? -> ToolNode(三个 MCP Server) -> 回边
                                      -> 否 -> schema 校验 -> END
    工具选择由模型自主完成，Planner 只通过 system prompt 注入层级预算。
"""

from langgraph.graph import END, START, StateGraph

from sports_agent.agent.planner import Planner
from sports_agent.agent.state import AgentState, TraceStep
from sports_agent.inference.client import InferenceClient

_SYSTEM_PROMPT = (
    "你是足球赛事分析助手。只能基于工具返回的数据与知识库内容作答，"
    "不得编造伤停、比分或概率；引用事实时标注来源。"
    "当前任务层级：{level}，工具调用上限：{max_iters} 次。"
)


def build_graph(
    planner: Planner | None = None,
    inference: InferenceClient | None = None,
):
    """编译 Agent 状态图。"""
    planner = planner or Planner()
    inference = inference or InferenceClient()

    def plan_node(state: AgentState) -> AgentState:
        plan = planner.plan(state["query"])
        trace: list[TraceStep] = state.get("trace", [])
        trace.append(
            TraceStep(
                kind="planner",
                name="rule_planner",
                latency_ms=getattr(planner, "_last_latency_ms", 0),
                detail=plan.to_dict(),
            )
        )
        return {
            "level": plan.level,
            "route": plan.route,
            "max_tool_iters": plan.max_tool_iters,
            "allow_simulation": plan.allow_simulation,
            "trace": trace,
        }

    def answer_node(state: AgentState) -> AgentState:
        # TODO(W6): 替换为 bind_tools + ToolNode + 条件回边的 ReAct 循环
        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(
                    level=state["level"], max_iters=state["max_tool_iters"]
                ),
            },
            {"role": "user", "content": state["query"]},
        ]
        result = inference.chat(state["route"], messages)
        trace: list[TraceStep] = state.get("trace", [])
        trace.append(
            TraceStep(
                kind="llm",
                name=result.model,
                latency_ms=result.latency_ms,
                detail={
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
            )
        )
        return {
            "final_answer": result.content,
            "inference": {
                "model": result.model,
                "latency_ms": result.latency_ms,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
            "trace": trace,
        }

    graph = StateGraph(AgentState)
    graph.add_node("planner", plan_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
