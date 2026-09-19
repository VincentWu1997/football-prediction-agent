"""预测服务核心：MCP Server B 与 FastAPI 共用的 ML 能力层。

数据源：data/processed/matches.csv（W1 清洗产物，单一事实来源）。
纪律：
- DC 模型只用 as_of 之前的比赛拟合，并按 (league, as_of) 缓存；
- 未见过 as_of 之前的任何比赛的球队会被拒绝（不给中性参数预测，避免误导）；
- 赛季模拟的"剩余赛程"来自真实赛程表（matches 中的未来场次），
  不虚构赛程；数据已含实际最终排名，供回看校验。
"""

from functools import lru_cache

import pandas as pd

from sports_agent.ml.dixon_coles import DixonColesModel, fit_dixon_coles
from sports_agent.ml.monte_carlo import (
    MatchSimulation,
    SeasonSimulation,
    simulate_match,
    simulate_season_fixtures,
)
from sports_agent.settings import REPO_ROOT

MATCHES_CSV = REPO_ROOT / "data" / "processed" / "matches.csv"

# 常用全称别名 -> football-data.co.uk 官方队名（数据源口径）
_ALIASES = {
    "manchester united": "Man United",
    "manchester city": "Man City",
    "newcastle united": "Newcastle",
    "tottenham hotspur": "Tottenham",
    "west ham united": "West Ham",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton": "Wolves",
    "nottingham forest": "Nott'm Forest",
    "west bromwich albion": "West Brom",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "norwich city": "Norwich",
    "cardiff city": "Cardiff",
    "hull city": "Hull",
    "stoke city": "Stoke",
    "swansea city": "Swansea",
    "sheffield united": "Sheffield United",
    "aston villa": "Aston Villa",
    "crystal palace": "Crystal Palace",
}


@lru_cache(maxsize=1)
def _all_team_names() -> frozenset[str]:
    df = load_matches()
    return frozenset(df["home_team"]) | frozenset(df["away_team"])


def resolve_team(name: str) -> str:
    """队名规范化：官方名 -> 原样；别名表 -> 映射；否则 difflib 近似匹配。"""
    if name in _all_team_names():
        return name
    if name.lower() in _ALIASES:
        return _ALIASES[name.lower()]
    import difflib

    close = difflib.get_close_matches(name, _all_team_names(), n=1, cutoff=0.8)
    if close:
        return close[0]
    raise ValueError(f"未知球队: {name}（请使用官方队名，如 Man United）")


@lru_cache(maxsize=1)
def load_matches() -> pd.DataFrame:
    df = pd.read_csv(MATCHES_CSV, parse_dates=["match_date"])
    df["season"] = df["season"].astype(str)
    return df.sort_values("match_date").reset_index(drop=True)


@lru_cache(maxsize=64)
def _fit_cached(league: str, as_of: str) -> DixonColesModel:
    df = load_matches()
    train = df[(df["league"] == league) & (df["match_date"] < as_of)]
    if train.empty:
        raise ValueError(f"联赛 {league} 在 {as_of} 之前没有比赛数据")
    return fit_dixon_coles(train)


def _resolve_league(home: str, away: str, as_of: str | None) -> str:
    """按两队共同出现的联赛推断（最近 3 个赛季窗口）。"""
    df = load_matches()
    sub = df if as_of is None else df[df["match_date"] < pd.Timestamp(as_of)]
    if sub.empty:
        raise ValueError(f"{as_of} 之前没有比赛数据")
    recent_seasons = sorted(sub["season"].unique())[-3:]
    recent = sub[sub["season"].isin(recent_seasons)]
    counts: dict[str, int] = {}
    for lg, g in recent.groupby("league"):
        teams = set(g["home_team"]) | set(g["away_team"])
        if home in teams and away in teams:
            counts[lg] = len(g)
    if not counts:
        raise ValueError(
            f"最近三个赛季未在同一联赛找到 {home} 与 {away}；请显式传入 league"
        )
    return max(counts, key=lambda lg: counts[lg])


def predict_match(
    home: str,
    away: str,
    league: str | None = None,
    as_of: str | None = None,
    n_sims: int = 20_000,
    seed: int | None = 42,
) -> dict:
    """单场预测：DC 概率 + 蒙特卡洛比分仿真。队名自动规范化。"""
    home, away = resolve_team(home), resolve_team(away)
    as_of_str = as_of or str(load_matches()["match_date"].max().date())
    league = league or _resolve_league(home, away, as_of_str)
    model = _fit_cached(league, as_of_str)

    df = load_matches()
    history = df[
        (df["league"] == league) & (df["match_date"] < pd.Timestamp(as_of_str))
    ]
    known = set(history["home_team"]) | set(history["away_team"])
    unknown = [t for t in (home, away) if t not in known]
    if unknown:
        raise ValueError(f"球队在 {as_of_str} 前的 {league} 数据中不存在: {unknown}")

    sim: MatchSimulation = simulate_match(model, home, away, n_sims=n_sims, seed=seed)
    p_h, p_d, p_a = model.probs(home, away)
    return {
        "league": league,
        "as_of": as_of_str,
        "home": home,
        "away": away,
        "model": "dixon_coles",
        "analytic_probs": {"H": round(p_h, 4), "D": round(p_d, 4), "A": round(p_a, 4)},
        "monte_carlo": {
            "n_sims": sim.n_sims,
            "probs": {k: round(v, 4) for k, v in sim.probs.items()},
            "ci_95": {k: [round(x, 4) for x in v] for k, v in sim.cis.items()},
            "top_scores": sim.top_scores,
            "over_2_5": round(sim.over_25, 4),
            "btts": round(sim.btts, 4),
        },
    }


def season_state(league: str, as_of: str) -> dict:
    """切分某联赛某赛季：as_of 前为已赛（积分榜），之后为剩余赛程。"""
    df = load_matches()
    lg = df[df["league"] == league]
    season = lg[lg["match_date"] < pd.Timestamp(as_of)]["season"].max()
    if season is None:
        raise ValueError(f"{league} 在 {as_of} 前没有比赛")
    in_season = lg[lg["season"] == season]
    played = in_season[in_season["match_date"] < pd.Timestamp(as_of)]
    remaining = in_season[in_season["match_date"] >= pd.Timestamp(as_of)]

    teams = sorted(set(in_season["home_team"]) | set(in_season["away_team"]))
    points, gd, gf = {}, {}, {}
    for row in played.itertuples(index=False):
        points[row.home_team] = points.get(row.home_team, 0) + (
            3 if row.full_time_res == "H" else (1 if row.full_time_res == "D" else 0)
        )
        points[row.away_team] = points.get(row.away_team, 0) + (
            3 if row.full_time_res == "A" else (1 if row.full_time_res == "D" else 0)
        )
        gd[row.home_team] = gd.get(row.home_team, 0) + row.fthg - row.ftag
        gd[row.away_team] = gd.get(row.away_team, 0) + row.ftag - row.fthg
        gf[row.home_team] = gf.get(row.home_team, 0) + row.fthg
        gf[row.away_team] = gf.get(row.away_team, 0) + row.ftag

    return {
        "league": league,
        "season": season,
        "as_of": as_of,
        "teams": teams,
        "n_played": len(played),
        "n_remaining": len(remaining),
        "current_points": points,
        "current_gd": gd,
        "current_gf": gf,
        "fixtures": list(zip(remaining["home_team"], remaining["away_team"], strict=True)),
        "actual_final_order": _actual_final_order(in_season),
    }


def _actual_final_order(in_season: pd.DataFrame) -> list[str]:
    """真实最终排名（积分/GD/GF），用于模拟结果的回看校验。"""
    pts, gd, gf = {}, {}, {}
    for row in in_season.itertuples(index=False):
        pts[row.home_team] = pts.get(row.home_team, 0) + (
            3 if row.full_time_res == "H" else (1 if row.full_time_res == "D" else 0)
        )
        pts[row.away_team] = pts.get(row.away_team, 0) + (
            3 if row.full_time_res == "A" else (1 if row.full_time_res == "D" else 0)
        )
        gd[row.home_team] = gd.get(row.home_team, 0) + row.fthg - row.ftag
        gd[row.away_team] = gd.get(row.away_team, 0) + row.ftag - row.fthg
        gf[row.home_team] = gf.get(row.home_team, 0) + row.fthg
        gf[row.away_team] = gf.get(row.away_team, 0) + row.ftag
    table = pd.DataFrame({"pts": pts, "gd": gd, "gf": gf})
    return table.sort_values(["pts", "gd", "gf"], ascending=False).index.tolist()


def simulate_season(
    league: str, as_of: str, n_runs: int = 20_000, seed: int | None = 42
) -> dict:
    """L4：赛季剩余赛程蒙特卡洛（DC 只用 as_of 前数据拟合）。"""
    state = season_state(league, as_of)
    model = _fit_cached(league, as_of)
    sim: SeasonSimulation = simulate_season_fixtures(
        model,
        teams=state["teams"],
        current_points=state["current_points"],
        current_gd=state["current_gd"],
        current_gf=state["current_gf"],
        fixtures=state["fixtures"],
        n_runs=n_runs,
        seed=seed,
    )
    actual = state["actual_final_order"]
    return {
        "league": state["league"],
        "season": state["season"],
        "as_of": state["as_of"],
        "teams": state["teams"],
        "n_runs": sim.n_runs,
        "n_played": state["n_played"],
        "n_remaining": state["n_remaining"],
        "title": sim.title,
        "top4": sim.top4,
        "relegation": sim.relegation,
        "expected_points": sim.expected_points,
        "actual_final_order": actual,
    }
