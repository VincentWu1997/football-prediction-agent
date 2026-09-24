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

    W6 ReAct 循环：planner 决定层级与预算，react 节点在预算内自主调工具。
    messages 为 OpenAI /v1/chat/completions 的完整对话（含 system / user / assistant / tool 消息）。
    """

    query: str
    level: str  # L1 / L2 / L3 / L4
    route: str  # fast / precise
    max_tool_iters: int
    allow_simulation: bool
    iters_used: int  # 已用工具迭代次数
    messages: list[dict]  # LLM 上下文（OpenAI 消息格式）
    tools_available: list[str]  # 当前轮可见工具白名单（按层级裁剪）
    final_answer: str
    sources: list[dict[str, Any]]  # 引用来源（doc_source / 比赛 id 等）
    trace: list[TraceStep]
    inference: dict[str, Any]  # 汇总推理消耗
