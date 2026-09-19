"""Flat-stake value betting 回测。

规则（显式约定，避免口径漂移）：
- 对每场比赛的 H/D/A 三个结果分别算 EV = p_model * decimal_odds - 1；
- 每场最多下一注：选 EV 最高且超过 edge 阈值的结果，固定 1 单位本金；
- 结算：命中收回 odds（净赚 odds-1），未中损失 1；
- 只在传入的有效赔率行上回测（调用方负责保证为赛前 closing odds 且非空）；
- 最大回撤按比赛时间排序后的累计盈亏曲线计算。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SIDES = ("H", "D", "A")


@dataclass(frozen=True)
class BacktestResult:
    bets: pd.DataFrame
    n_bets: int
    n_hits: int
    total_staked: float
    pnl: float
    roi: float
    max_drawdown: float

    def summary_dict(self) -> dict:
        return {
            "n_bets": self.n_bets,
            "hit_rate": round(self.n_hits / self.n_bets, 4) if self.n_bets else None,
            "avg_odds": round(float(self.bets["odds"].mean()), 3) if self.n_bets else None,
            "roi": round(self.roi, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "pnl": round(self.pnl, 2),
        }


def run_backtest(
    probs: np.ndarray,
    odds: np.ndarray,
    outcomes: pd.Series,
    dates: pd.Series,
    edge_threshold: float,
) -> BacktestResult:
    """probs/odds 形状均为 (n,3)，列顺序 H/D/A。"""
    ev = probs * odds - 1.0
    best_idx = ev.argmax(axis=1)
    best_ev = ev.max(axis=1)
    mask = best_ev > edge_threshold

    rows = []
    for i in np.flatnonzero(mask):
        side = SIDES[int(best_idx[i])]
        decimal_odds = float(odds[i, best_idx[i]])
        hit = outcomes.iloc[i] == side
        rows.append(
            {
                "match_date": dates.iloc[i],
                "side": side,
                "odds": decimal_odds,
                "model_prob": float(probs[i, best_idx[i]]),
                "edge": float(best_ev[i]),
                "hit": bool(hit),
                "pnl": decimal_odds - 1.0 if hit else -1.0,
            }
        )

    bets = pd.DataFrame(rows)
    n_bets = len(bets)
    if n_bets == 0:
        return BacktestResult(bets, 0, 0, 0.0, 0.0, 0.0, 0.0)

    bets = bets.sort_values("match_date").reset_index(drop=True)
    pnl = float(bets["pnl"].sum())
    cumulative = bets["pnl"].cumsum()
    running_max = cumulative.cummax().clip(lower=0)
    # 从峰值（或 0）跌到谷底的最大负值
    max_drawdown = float((cumulative - running_max).min())

    return BacktestResult(
        bets=bets,
        n_bets=n_bets,
        n_hits=int(bets["hit"].sum()),
        total_staked=float(n_bets),
        pnl=pnl,
        roi=pnl / n_bets,
        max_drawdown=max_drawdown,
    )
