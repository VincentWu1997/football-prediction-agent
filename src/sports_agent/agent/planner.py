"""任务规划器：判定 L1-L4 层级并产出推理/工具预算。

当前实现为规则冷启动版本；W6 将替换为 fast 模型 + few-shot 的分类，
并用 benchmarks/eval_sets/routing.jsonl 报告真实分类准确率（混淆矩阵）。
"""

import time
from dataclasses import asdict, dataclass

from sports_agent.inference.registry import LevelBudget, ModelRegistry

# 关键词按层级排列，顺序即优先级
_LEVEL_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("L4", ("模拟", "赛季", "夺冠", "降级", "前四", "概率排名", "剩余赛程")),
    ("L3", ("详细", "深度", "分析", "战术", "伤停", "阵容", "盘口", "赔率")),
    ("L1", ("积分榜", "排名", "赛程", "比分", "战绩", "什么时候", "几点")),
]


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

    def _classify(self, query: str) -> tuple[str, str]:
        q = query.strip()
        for level, keywords in _LEVEL_KEYWORDS:
            hit = [kw for kw in keywords if kw in q]
            if hit:
                return level, f"关键词命中 {level}: {','.join(hit)}"
        # 默认按单场快速预测处理
        return "L2", "未命中特定关键词，默认单场快速预测"

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
        plan_ms = int((time.perf_counter() - start) * 1000)
        # 规则版延迟极低；模型版需记录此值用于与规则版对比
        self._last_latency_ms = plan_ms
        return plan
