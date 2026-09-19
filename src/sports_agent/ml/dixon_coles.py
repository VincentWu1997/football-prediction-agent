"""Dixon-Coles (1997) 双变量 Poisson 模型。

- 各队独立进攻/防守参数，λ_home = exp(μ + hfa + atk_h - def_a)；
- τ 低比分校正项修正独立 Poisson 对 0-0/1-0/0-1/1-1 的偏差；
- 历史比赛按时间指数衰减加权，半衰期约 1 年（xi=ln2/365≈0.0019/天）；
- 训练折未出现的升级球队以联赛平均水平参数（0,0）预测，不跨联赛借参。

似然计算全程向量化，单次拟合为秒级。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

XI_PER_DAY = np.log(2) / 365.0  # 半衰期约 365 天
_MAX_GOALS = 10


@dataclass
class DixonColesModel:
    """拟合后的 DC 模型。"""

    attack: dict[str, float]
    defense: dict[str, float]
    intercept: float
    home_adv: float
    rho: float

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        """返回 (λ_主队进球, λ_客队进球)；未见过的球队按平均参数。"""
        log_lh = (
            self.intercept
            + self.home_adv
            + self.attack.get(home, 0.0)
            - self.defense.get(away, 0.0)
        )
        log_la = (
            self.intercept
            + self.attack.get(away, 0.0)
            - self.defense.get(home, 0.0)
        )
        return float(np.exp(log_lh)), float(np.exp(log_la))

    def score_grid(
        self, home: str, away: str, max_goals: int = _MAX_GOALS
    ) -> tuple[np.ndarray, float, float]:
        """返回归一化的比分概率矩阵 (行=主队进球) 与 (λ主, λ客)。"""
        lh, la = self.lambdas(home, away)
        goals = np.arange(max_goals + 1)
        # Poisson pmf（对数空间后还原，数值稳定）；不能用 +=，列向量无法原地广播为方阵
        log_pmf_h = goals[:, None] * np.log(lh) - lh - gammaln(goals[:, None] + 1)
        log_pmf_a = goals[None, :] * np.log(la) - la - gammaln(goals[None, :] + 1)
        grid = np.exp(log_pmf_h + log_pmf_a)
        grid *= _tau_matrix(goals, lh, la, self.rho)
        grid /= grid.sum()
        return grid, lh, la

    def probs(
        self, home: str, away: str, max_goals: int = _MAX_GOALS
    ) -> tuple[float, float, float]:
        """返回 (主胜, 平, 客胜) 概率，按 0..max_goals 比分网格求和并归一。"""
        grid, _, _ = self.score_grid(home, away, max_goals)

        wins = np.tril_indices(max_goals + 1, k=-1)  # x > y（主队进球行）
        losses = np.triu_indices(max_goals + 1, k=1)
        p_h = float(grid[wins].sum())
        p_a = float(grid[losses].sum())
        p_d = float(np.trace(grid))
        return p_h, p_d, p_a


def _tau_matrix(goals: np.ndarray, lh: float, la: float, rho: float) -> np.ndarray:
    """Dixon-Coles τ 校正矩阵（行=主队进球 x，列=客队进球 y）。"""
    x = goals[:, None]
    y = goals[None, :]
    tau = np.ones((len(goals), len(goals)))
    tau[(x == 0) & (y == 0)] = 1.0 - lh * la * rho
    tau[(x == 0) & (y == 1)] = 1.0 + lh * rho
    tau[(x == 1) & (y == 0)] = 1.0 + la * rho
    tau[(x == 1) & (y == 1)] = 1.0 - rho
    return np.clip(tau, 1e-10, None)


def fit_dixon_coles(train: pd.DataFrame, xi: float = XI_PER_DAY) -> DixonColesModel:
    """在训练集（同一联赛、按时间加权）上拟合 DC。

    train 列：match_date, home_team, away_team, fthg, ftag。
    """
    teams = sorted(set(train["home_team"]) | set(train["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hidx = train["home_team"].map(idx).to_numpy()
    aidx = train["away_team"].map(idx).to_numpy()
    gh = train["fthg"].to_numpy(dtype=float)
    ga = train["ftag"].to_numpy(dtype=float)

    ref_date = train["match_date"].max()
    days_old = (ref_date - train["match_date"]).dt.days.to_numpy(dtype=float)
    weights = np.exp(-xi * days_old)

    avg_goal = np.average(np.concatenate([gh, ga]), weights=np.concatenate([weights, weights]))

    def neg_log_likelihood(params: np.ndarray) -> float:
        atk = params[:n]
        dfn = params[n : 2 * n]
        mu, hfa, rho = params[-3:]

        log_lh = mu + hfa + atk[hidx] - dfn[aidx]
        log_la = mu + atk[aidx] - dfn[hidx]
        lh, la = np.exp(log_lh), np.exp(log_la)

        ll_pois = (
            gh * log_lh - lh - gammaln(gh + 1) + ga * log_la - la - gammaln(ga + 1)
        )

        tau = np.ones_like(lh)
        m00 = (gh == 0) & (ga == 0)
        m01 = (gh == 0) & (ga == 1)
        m10 = (gh == 1) & (ga == 0)
        m11 = (gh == 1) & (ga == 1)
        tau[m00] = 1.0 - lh[m00] * la[m00] * rho
        tau[m01] = 1.0 + lh[m01] * rho
        tau[m10] = 1.0 + la[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-10, None)

        return float(-np.sum(weights * (ll_pois + np.log(tau))))

    # 可识别性约束：进攻参数之和为 0
    constraints = [{"type": "eq", "fun": lambda p: p[:n].sum()}]
    bounds = [(-2.0, 2.0)] * (2 * n) + [
        (-1.0, 2.0),   # μ：平均进球的对数
        (-0.5, 0.8),   # 主场优势
        (-0.2, 0.2),   # ρ 低比分校正
    ]
    p0 = np.concatenate(
        [
            np.zeros(n),
            np.zeros(n),
            [np.log(max(avg_goal, 0.5)), 0.25, -0.06],
        ]
    )

    result = minimize(
        neg_log_likelihood,
        p0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9, "disp": False},
    )
    if not result.success:
        # 数值失败不静默：抛出让实验层感知（避免产出未拟合完成的假模型）
        raise RuntimeError(f"Dixon-Coles 拟合失败: {result.message}")

    params = result.x
    return DixonColesModel(
        attack={t: float(params[i]) for t, i in idx.items()},
        defense={t: float(params[n + i]) for t, i in idx.items()},
        intercept=float(params[-3]),
        home_adv=float(params[-2]),
        rho=float(params[-1]),
    )
