"""Agent 共享状态定义。"""

from typing import Any, TypedDict


class TraceStep(TypedDict):
    """单步执行记录，供前端渲染推理过程。"""

    kind: str  # planner / llm / tool
    name: str
    latency_ms: int
    detail: dict[str, Any]


class AgentState(TypedDict, total=False):
    """LangGraph 在节点间传递的状态。

    注意：当前为骨架版本，工具由固定 LLM 节点占位；
    W6 将把 answer 节点替换为 bind_tools + ToolNode + 回边的 ReAct 循环。
    """

    query: str
    level: str  # L1 / L2 / L3 / L4
    route: str  # fast / precise
    max_tool_iters: int
    allow_simulation: bool
    final_answer: str
    trace: list[TraceStep]
    inference: dict[str, Any]
