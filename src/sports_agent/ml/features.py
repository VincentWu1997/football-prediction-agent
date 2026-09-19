"""W3 特征工程：所有特征严格只使用赛前信息。

防泄漏要点：
- 球队滚动状态先在"球队-比赛"长表上 shift(1) 去掉当场，再 rolling，
  因此第 N 场的状态只含第 1..N-1 场；
- H2H 同理在 (球队, 对手) 组内 shift(1)；
- rest_days 为距本队上一场的天数，首场为 NaN（XGBoost 原生处理缺失）；
- DC λ 与 ELO 评分差由调用方在折外用"仅训练数据拟合"的模型填入；
- 赔率特征：Pinnacle 去水概率优先，缺失时回退 B365，ps_available 标识来源
  （显式指示特征，不是常数占位）。
"""

import numpy as np
import pandas as pd

from sports_agent.eval.odds import devig
from sports_agent.ml.dixon_coles import DixonColesModel

RECENT_N = 5  # 近 N 场状态窗口
H2H_N = 5
MIN_PERIODS = 3  # 滚动窗口内最少样本数，不足则 NaN

FEATURE_COLUMNS = [
    "elo_dr",
    "dc_log_lh",
    "dc_log_la",
    "odds_h",
    "odds_d",
    "odds_a",
    "odds_overround",
    "ps_available",
    "home_pts5",
    "home_gf5",
    "home_ga5",
    "away_pts5",
    "away_gf5",
    "away_ga5",
    "home_rest_days",
    "away_rest_days",
    "home_h2h_pts5",
]
LABEL_MAP = {"H": 0, "D": 1, "A": 2}


def _team_long(df: pd.DataFrame) -> pd.DataFrame:
    """每场拆成主/客两行的球队视角长表。"""
    home = df[
        ["match_date", "home_team", "away_team", "fthg", "ftag", "full_time_res"]
    ].rename(columns={"home_team": "team", "away_team": "opponent", "fthg": "gf", "ftag": "ga"})
    home["is_home"] = 1
    home["points"] = home["full_time_res"].map({"H": 3, "D": 1, "A": 0})

    away = df[
        ["match_date", "home_team", "away_team", "fthg", "ftag", "full_time_res"]
    ].rename(columns={"away_team": "team", "home_team": "opponent", "ftag": "gf", "fthg": "ga"})
    away["is_home"] = 0
    away["points"] = away["full_time_res"].map({"A": 3, "D": 1, "H": 0})

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "match_date"]).reset_index(drop=True)
    return long


def _rolling_team_features(long: pd.DataFrame) -> pd.DataFrame:
    """在长表上计算每队赛前滚动状态与休息天数。

    必须用 groupby(...).transform(shift+rolling)：若先 shift 再在全局
    Series 上 rolling，窗口会在球队边界混入相邻球队的历史比赛（数据泄漏）。
    """
    long = long.sort_values(["team", "match_date"]).reset_index(drop=True)

    def _recent(series_name: str) -> pd.Series:
        return long.groupby("team")[series_name].transform(
            lambda s: s.shift(1).rolling(RECENT_N, min_periods=MIN_PERIODS).mean()
        )

    long["pts_recent"] = _recent("points")
    long["gf_recent"] = _recent("gf")
    long["ga_recent"] = _recent("ga")
    long["rest_days"] = long.groupby("team")["match_date"].transform(
        lambda s: (s - s.shift(1)).dt.days
    )

    # H2H：固定同一对手，组内按时间 shift 后滚动；transform 自动对齐回原行序
    h2h_sorted = long.sort_values(["team", "opponent", "match_date"])
    long["h2h_pts"] = (
        h2h_sorted.groupby(["team", "opponent"])["points"]
        .transform(lambda s: s.shift(1).rolling(H2H_N, min_periods=2).mean())
    )
    return long


def build_features(df: pd.DataFrame, dc: DixonColesModel | None = None) -> pd.DataFrame:
    """为单联赛比赛表生成特征列。df 需含 elo_dr；dc 为 None 时 λ 特征留 NaN。

    保留 df 的原始行索引（仅按日期重排），便于调用方与 walk-forward 切片对齐。
    """
    out = df.sort_values("match_date").copy()
    # 后续 left merge 会把索引重置为 RangeIndex；left merge 保持左表行数与行顺序，
    # 故在返回前把原始索引（walk-forward 切片依赖它对齐）原样装回
    original_index = out.index

    # --- 赔率特征：PS 优先，B365 回退 ---
    ps = devig(out["psh"], out["psd"], out["psa"])
    b365 = devig(out["b365h"], out["b365d"], out["b365a"])
    ps_ok = ps.notna().all(axis=1)
    # DataFrame.where 的 Series 条件按行广播到 H/D/A 三列
    probs = ps.where(ps_ok, b365)
    out["odds_h"] = probs["H"]
    out["odds_d"] = probs["D"]
    out["odds_a"] = probs["A"]
    out["ps_available"] = ps_ok.astype(int)

    # 市场 overround（毛利水平）：用实际采用的那组赔率
    ps_odds = out[["psh", "psd", "psa"]].to_numpy()
    b365_odds = out[["b365h", "b365d", "b365a"]].to_numpy()
    odds_used = np.where(ps_ok.to_numpy()[:, None], ps_odds, b365_odds)
    reciprocal_sum = (
        1.0 / pd.DataFrame(odds_used, columns=["h", "d", "a"])
    ).sum(axis=1)
    out["odds_overround"] = reciprocal_sum - 1.0

    # --- 滚动状态 ---
    long = _rolling_team_features(_team_long(out))
    state_cols = ["pts_recent", "gf_recent", "ga_recent", "rest_days", "h2h_pts"]
    home_state = long[long["is_home"] == 1][
        ["match_date", "team", "opponent", *state_cols]
    ].rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "pts_recent": "home_pts5",
            "gf_recent": "home_gf5",
            "ga_recent": "home_ga5",
            "rest_days": "home_rest_days",
            "h2h_pts": "home_h2h_pts5",
        }
    )
    away_state = long[long["is_home"] == 0][
        ["match_date", "team", "opponent", "pts_recent", "gf_recent", "ga_recent", "rest_days"]
    ].rename(
        columns={
            "team": "away_team",
            "opponent": "home_team",
            "pts_recent": "away_pts5",
            "gf_recent": "away_gf5",
            "ga_recent": "away_ga5",
            "rest_days": "away_rest_days",
        }
    )

    keys = ["match_date", "home_team", "away_team"]
    out = out.merge(home_state, on=keys, how="left")
    out = out.merge(away_state, on=keys, how="left")
    out.index = original_index

    attach_dc_lambdas(out, dc)
    out["label"] = out["full_time_res"].map(LABEL_MAP)
    return out


def attach_dc_lambdas(frame: pd.DataFrame, dc: DixonColesModel | None) -> pd.DataFrame:
    """用给定（折外、仅训练数据拟合的）DC 模型填入 λ 特征；None 时留 NaN。"""
    if dc is None:
        frame["dc_log_lh"] = np.nan
        frame["dc_log_la"] = np.nan
        return frame

    lambdas = [
        dc.lambdas(h, a)
        for h, a in zip(frame["home_team"], frame["away_team"], strict=True)
    ]
    frame["dc_log_lh"] = np.log([lh for lh, _ in lambdas])
    frame["dc_log_la"] = np.log([la for _, la in lambdas])
    return frame
