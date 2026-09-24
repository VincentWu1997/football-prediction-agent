"""任务规划器：判定 L1-L4 层级并产出推理/工具预算。

W6 提供：
- 规则版 Planner.plan()：关键词分类（基线，无需 LLM）；
- 模型版 Planner.plan_with_llm()：fast 模型 + few-shot JSON 输出，作为对照；
- experiment_w6 同时跑两版，报告准确率与混淆矩阵。

规则版要点（基于 W5 后 100 条评测集发现的误差修正）：
- "积分榜"/"赛程"/"第几"/"排名"/"比分" 等事实查询 → L1，优先级高于 L4；
- "模拟"/"前四"/"夺冠"/"降级"/"概率" 才走 L4（"赛季"单独不算 L4 触发）；
- 默认 L2（单场快速预测）。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from sports_agent.inference.registry import LevelBudget, ModelRegistry

# L1 事实查询关键词：强信号
_L1_KEYWORDS = (
    "积分榜",
    "排名",
    "积分",
    "赛程",
    "比分",
    "战绩",
    "什么时候",
    "几点",
    "第几",
    "目前积分",
    "上一场",
)
# L3 深度分析关键词（"对阵"不算 L3 强信号，避免误吞 L2 单场预测）
_L3_KEYWORDS = ("详细", "深度", "分析", "战术", "伤停", "阵容", "盘口", "赔率", "复盘")
# L4 必须命中"模拟/夺冠/前四/降级/概率"中至少一个，"赛季"单独不触发
_L4_KEYWORDS = ("模拟", "夺冠", "前四", "降级", "概率", "蒙特卡洛", "剩余赛程")
# L2 默认（单场快速预测）：包含"对/对阵/谁赢" 等
_L2_HINTS = ("对", "对阵", "谁赢", "胜负", "胜平负", "赛果")


@dataclass(frozen=True)
class Plan:
    """Planner 输出。"""

    level: str
    route: str
    max_tool_iters: int
    allow_simulation: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class Planner:
    """根据 query 与层级预算生成 Plan。"""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self._budgets = {lv: self.registry.level(lv) for lv in ("L1", "L2", "L3", "L4")}
        self._last_latency_ms: int = 0
        # LLM 版延迟记录（仅在调用 plan_with_llm 后填充）
        self._last_llm_latency_ms: int = 0
        self._llm_used: bool = False

    def _classify(self, query: str) -> tuple[str, str]:
        q = query.strip()
        # L3 强信号优先于 L1（避免"详细分析 ... 赛程密度"被赛程触发到 L1）
        l3_strong = [
            kw for kw in ("详细", "深度", "伤停", "战术", "盘口", "赔率", "复盘") if kw in q
        ]
        if l3_strong:
            return "L3", f"关键词命中 L3 强信号: {','.join(l3_strong)}"
        # L1 事实查询：积分榜/排名/赛程/比分/战绩等
        hit1 = [kw for kw in _L1_KEYWORDS if kw in q]
        # 但若同时有 L4 强信号（模拟/夺冠/前四/降级），跳过 L1
        hit4 = [kw for kw in _L4_KEYWORDS if kw in q]
        if hit1 and not hit4:
            return "L1", f"关键词命中 L1: {','.join(hit1)}"
        if hit4:
            return "L4", f"关键词命中 L4: {','.join(hit4)}"
        # L3 其他弱信号（分析/阵容/对阵 等）
        hit3 = [kw for kw in _L3_KEYWORDS if kw in q]
        if hit3:
            return "L3", f"关键词命中 L3: {','.join(hit3)}"
        # 默认按单场快速预测处理
        return "L2", "未命中 L1/L3/L4 特定关键词，默认单场快速预测"

    def plan(self, query: str) -> Plan:
        start = time.perf_counter()
        level, reason = self._classify(query)
        budget: LevelBudget = self._budgets[level]
        plan = Plan(
            level=level,
            route=budget.route,
            max_tool_iters=budget.max_tool_iters,
            allow_simulation=budget.allow_simulation,
            reason=reason,
        )
        self._last_latency_ms = int((time.perf_counter() - start) * 1000)
        self._llm_used = False
        return plan

    # ============ LLM 分类版（few-shot JSON，方案 §6.1）============

    _FEW_SHOT = """你是体育赛事 Agent 的任务路由器。把用户问题分为 4 个层级之一，输出 JSON：

L1：事实查询（积分榜、排名、赛程、比分、战绩、几点开始）
L2：单场快速预测（谁会赢、胜平负、胜负、赛果）
L3：深度分析（详细分析、战术层面、伤停影响、盘口赔率、阵容深度）
L4：赛季全局模拟（模拟赛季、前四概率、夺冠概率、降级概率、蒙特卡洛）

示例：
问："英超积分榜" → {"level":"L1","reason":"事实查询：积分榜"}
问："曼联对利物浦谁会赢" → {"level":"L2","reason":"单场胜负预测"}
问："详细分析曼联对利物浦，考虑伤停" → {"level":"L3","reason":"含'详细分析'+涉及单场+伤停"}
问："模拟本赛季英超前四归属概率" → {"level":"L4","reason":"含'模拟'+前四概率，赛季级任务"}

只输出 JSON 一行，不要额外文字，思考也不要输出。"""

    def plan_with_llm(self, query: str, inference_client) -> Plan:
        """用 fast 模型做 L1-L4 分类；inference_client 为 InferenceClient 实例。

        失败时降级到规则版（self.plan）。
        """
        start = time.perf_counter()
        try:
            result = inference_client.chat(
                "fast",
                [
                    {"role": "system", "content": self._FEW_SHOT},
                    {"role": "user", "content": query},
                ],
            )
        except Exception as e:
            self._last_llm_latency_ms = int((time.perf_counter() - start) * 1000)
            self._llm_used = False
            rule_plan = self.plan(query)
            rule_plan_dict = rule_plan.to_dict()
            rule_plan_dict["llm_fallback_reason"] = f"{type(e).__name__}: {str(e)[:120]}"
            return rule_plan
        self._last_llm_latency_ms = result.latency_ms
        self._llm_used = True
        # 解析 JSON（用 json-repair 兜底，模型偶尔会输出多余文字）
        try:
            from json_repair import repair_json

            data = json.loads(repair_json(result.content))
            level = data.get("level", "L2")
            reason = data.get("reason", "")
        except Exception:
            level, reason = "L2", f"LLM 输出解析失败，默认 L2: {result.content[:80]}"
        if level not in self._budgets:
            level, reason = "L2", f"LLM 输出非法 level={level}，降级 L2"
        budget: LevelBudget = self._budgets[level]
        return Plan(
            level=level,
            route=budget.route,
            max_tool_iters=budget.max_tool_iters,
            allow_simulation=budget.allow_simulation,
            reason=f"[LLM] {reason}",
        )
