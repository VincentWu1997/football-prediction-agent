"""ELO 评分模型（足球版）。

- 每个联赛独立一套评分（不做跨联赛映射，升降级球队回到基准分）；
- 主场优势以固定评分数 HFA 加到主队；
- 平局概率不来自 ELO 本身，用训练折拟合的高斯形曲线
  p_draw(dr) = a * exp(-b * dr^2)，再由期望得分拆出主胜/客胜。

所有赛前预测只使用当时已发生比赛更新后的评分（天然无未来信息）。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BASE_RATING = 1500.0
DEFAULT_K = 20.0
DEFAULT_HFA = 65.0  # 主场优势（评分点）


@dataclass(frozen=True)
class EloConfig:
    k: float = DEFAULT_K
    hfa: float = DEFAULT_HFA
    base_rating: float = BASE_RATING


def expected_home_score(dr: float) -> float:
    """主队期望得分（胜 1/平 0.5/负 0 的期望），dr 为含主场优势的评分差。"""
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


def fit_draw_curve(rating_diffs: np.ndarray, outcomes: pd.Series) -> tuple[float, float]:
    """在训练折上拟合 p_draw(dr)=a*exp(-b*dr^2)。返回 (a, b)。"""
    is_draw = (outcomes.to_numpy() == "D").astype(float)
    x = np.abs(rating_diffs)

    def model(xv: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.exp(-b * xv**2)

    # b 的初值按"评分差 ~150 时平局率减半"给量级，避免曲线拟合不收敛
    popt, _ = curve_fit(
        model,
        x,
        is_draw,
        p0=(0.30, np.log(2) / 150**2),
        bounds=([0.05, 1e-7], [0.5, 1e-4]),
        maxfev=5000,
    )
    return float(popt[0]), float(popt[1])


def split_probs(e_home: float, p_draw: float) -> tuple[float, float, float]:
    """期望得分 + 平局概率 -> (主胜, 平, 客胜)，保证非负并归一。"""
    p_h = e_home - p_draw / 2.0
    p_a = 1.0 - e_home - p_draw / 2.0
    p_h = max(p_h, 1e-4)
    p_a = max(p_a, 1e-4)
    p_draw = max(p_draw, 1e-4)
    total = p_h + p_draw + p_a
    return p_h / total, p_draw / total, p_a / total


class EloEngine:
    """单联赛顺序评分引擎：回放比赛并记录每场赛前评分差。"""

    def __init__(self, config: EloConfig | None = None) -> None:
        self.cfg = config or EloConfig()
        self.ratings: dict[str, float] = {}

    def _rating(self, team: str) -> float:
        return self.ratings.get(team, self.cfg.base_rating)

    def pre_match_diff(self, home: str, away: str) -> float:
        """赛前评分差（含主场优势）。"""
        return self._rating(home) - self._rating(away) + self.cfg.hfa

    def update(self, home: str, away: str, outcome: str) -> None:
        """按赛果更新双方评分。"""
        rh, ra = self._rating(home), self._rating(away)
        e_h = expected_home_score(rh - ra + self.cfg.hfa)
        s_h = {"H": 1.0, "D": 0.5, "A": 0.0}[outcome]
        self.ratings[home] = rh + self.cfg.k * (s_h - e_h)
        self.ratings[away] = ra + self.cfg.k * ((1.0 - s_h) - (1.0 - e_h))

    def replay(self, matches: pd.DataFrame) -> pd.DataFrame:
        """按时间顺序回放比赛，返回含赛前评分差 dr 的副本。

        matches 需按日期升序，列：match_date, home_team, away_team, full_time_res。
        """
        rows = []
        for row in matches.sort_values("match_date").itertuples(index=False):
            dr = self.pre_match_diff(row.home_team, row.away_team)
            rows.append(dr)
            self.update(row.home_team, row.away_team, row.full_time_res)
        out = matches.copy()
        out["elo_dr"] = rows
        return out
