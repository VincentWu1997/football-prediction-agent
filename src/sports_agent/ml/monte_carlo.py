"""蒙特卡洛比分仿真：从 Dixon-Coles 比分网格精确采样。

- 单场：直接对 DC 网格（含 τ 校正）按概率抽样，得到比分分布、
  胜平负概率置信区间、大小球与 BTTS；
- 赛季：对"剩余赛程"逐场抽样积分，聚合排名分布（夺冠/前四/降级）。
  积分相同按 净胜球→进球数 顺序模拟排位；18 队联赛按降级 2 队处理。
"""

from dataclasses import dataclass

import numpy as np

from sports_agent.ml.dixon_coles import DixonColesModel

_MAX_GOALS = 10


def _outcome_points(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """比分矩阵 (n,2) -> 主/客积分增量数组 (n,)。"""
    home_pts = np.where(scores[:, 0] > scores[:, 1], 3.0, 0.0)
    home_pts[scores[:, 0] == scores[:, 1]] = 1.0
    away_pts = np.where(scores[:, 1] > scores[:, 0], 3.0, 0.0)
    away_pts[scores[:, 1] == scores[:, 0]] = 1.0
    return home_pts, away_pts


def _probs_with_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    se = np.sqrt(max(p * (1 - p), 1e-12) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


@dataclass(frozen=True)
class MatchSimulation:
    probs: dict[str, float]  # H/D/A 点估计
    cis: dict[str, tuple[float, float]]
    top_scores: list[dict]  # [{score, prob}] 降序前 N
    over_25: float
    btts: float
    n_sims: int


def simulate_match(
    model: DixonColesModel, home: str, away: str, n_sims: int = 20_000, seed: int | None = None
) -> MatchSimulation:
    """单场比分仿真（网格精确抽样）。"""
    rng = np.random.default_rng(seed)
    grid, _, _ = model.score_grid(home, away)
    draws = rng.choice(grid.size, size=n_sims, p=grid.ravel())
    scores = np.column_stack(np.unravel_index(draws, grid.shape))

    n_h = int((scores[:, 0] > scores[:, 1]).sum())
    n_d = int((scores[:, 0] == scores[:, 1]).sum())
    n_a = n_sims - n_h - n_d
    probs = {"H": n_h / n_sims, "D": n_d / n_sims, "A": n_a / n_sims}
    cis = {k: _probs_with_ci(v, n_sims) for k, v in probs.items()}

    unique, counts = np.unique(scores, axis=0, return_counts=True)
    order = np.argsort(-counts)[:5]
    top_scores = [
        {"score": f"{int(unique[i][0])}-{int(unique[i][1])}", "prob": float(counts[i] / n_sims)}
        for i in order
    ]
    total = scores.sum(axis=1)
    return MatchSimulation(
        probs=probs,
        cis=cis,
        top_scores=top_scores,
        over_25=float((total > 2.5).mean()),
        btts=float(((scores[:, 0] > 0) & (scores[:, 1] > 0)).mean()),
        n_sims=n_sims,
    )


@dataclass(frozen=True)
class SeasonSimulation:
    teams: list[str]
    n_runs: int
    title: dict[str, float]
    top4: dict[str, float]
    relegation: dict[str, float]
    expected_points: dict[str, float]


def simulate_season_fixtures(
    model: DixonColesModel,
    teams: list[str],
    current_points: dict[str, float],
    current_gd: dict[str, float],
    current_gf: dict[str, float],
    fixtures: list[tuple[str, str]],
    n_runs: int = 20_000,
    seed: int | None = None,
) -> SeasonSimulation:
    """对剩余赛程逐场抽样积分并聚合排名分布。

    teams 为全部参赛队（含已赛/未赛）；current_* 为已赛场次的累计统计；
    fixtures 为 (主队, 客队) 剩余赛程列表。
    """
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    pts = np.tile(
        np.array([current_points.get(t, 0.0) for t in teams], dtype=float), (n_runs, 1)
    )
    gd = np.tile(np.array([current_gd.get(t, 0.0) for t in teams], dtype=float), (n_runs, 1))
    gf = np.tile(np.array([current_gf.get(t, 0.0) for t in teams], dtype=float), (n_runs, 1))

    for home, away in fixtures:
        grid, _, _ = model.score_grid(home, away)
        draws = rng.choice(grid.size, size=n_runs, p=grid.ravel())
        scores = np.column_stack(np.unravel_index(draws, grid.shape))
        hp, ap = _outcome_points(scores)
        hi, ai = idx[home], idx[away]
        pts[:, hi] += hp
        pts[:, ai] += ap
        gd[:, hi] += scores[:, 0] - scores[:, 1]
        gd[:, ai] += scores[:, 1] - scores[:, 0]
        gf[:, hi] += scores[:, 0]
        gf[:, ai] += scores[:, 1]

    # 排名：积分 -> GD -> GF 降序；残余并列用微小噪声稳定打破（对概率影响可忽略）
    noise = rng.uniform(0, 0.01, size=(n_runs, n_teams))
    order = np.lexsort((noise, -gf, -gd, -pts), axis=1)

    title = np.zeros(n_teams)
    top4 = np.zeros(n_teams)
    rel_n = 3 if n_teams >= 20 else 2
    releg = np.zeros(n_teams)
    for rank_pos in range(n_teams):
        team_idx = order[:, rank_pos]
        if rank_pos == 0:
            np.add.at(title, team_idx, 1)
        if rank_pos < 4:
            np.add.at(top4, team_idx, 1)
        if rank_pos >= n_teams - rel_n:
            np.add.at(releg, team_idx, 1)

    return SeasonSimulation(
        teams=teams,
        n_runs=n_runs,
        title={t: float(title[idx[t]] / n_runs) for t in teams},
        top4={t: float(top4[idx[t]] / n_runs) for t in teams},
        relegation={t: float(releg[idx[t]] / n_runs) for t in teams},
        expected_points={t: float(pts[:, idx[t]].mean()) for t in teams},
    )
