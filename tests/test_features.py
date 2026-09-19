"""特征工程防泄漏与赔率回退测试。"""

import numpy as np
import pandas as pd

from sports_agent.ml.features import build_features


def _match(date: str, home: str, away: str, res: str, hg: int, ag: int) -> dict:
    return {
        "match_date": pd.Timestamp(date),
        "league": "T1",
        "season": "2324",
        "home_team": home,
        "away_team": away,
        "fthg": hg,
        "ftag": ag,
        "full_time_res": res,
        "elo_dr": 100.0,
        "psh": 2.0, "psd": 3.3, "psa": 4.0,
        "b365h": 2.0, "b365d": 3.3, "b365a": 4.0,
    }


def _mini_league() -> pd.DataFrame:
    # A 对 B 连赛 6 场，A 全胜；之后 C 首次出场
    rows = [
        _match("2024-08-01", "A", "B", "H", 2, 0),
        _match("2024-08-08", "B", "A", "A", 0, 1),
        _match("2024-08-15", "A", "B", "H", 3, 1),
        _match("2024-08-22", "B", "A", "A", 1, 2),
        _match("2024-08-29", "A", "B", "H", 1, 0),
        _match("2024-09-05", "C", "A", "H", 2, 1),  # C 队首场
    ]
    return pd.DataFrame(rows)


def test_first_matches_are_nan_until_enough_history() -> None:
    feat = build_features(_mini_league())
    first = feat.iloc[0]
    assert pd.isna(first["home_pts5"])  # A 首场无历史
    assert pd.isna(first["away_pts5"])  # B 首场无历史


def test_rolling_state_uses_only_prior_matches() -> None:
    feat = build_features(_mini_league())
    # 第 5 场（索引 4）A 主场：A 前 4 场全胜，进球 2/1/3/2
    fifth = feat.iloc[4]
    assert fifth["home_pts5"] == 3.0
    # 赛前 gf 均值 = 2.0；若泄漏当场（A 只进 1 球）会变成 (8+1)/5=1.8
    assert fifth["home_gf5"] == 2.0


def test_new_team_does_not_inherit_other_teams_history() -> None:
    feat = build_features(_mini_league())
    c_match = feat[feat["home_team"] == "C"].iloc[0]
    assert pd.isna(c_match["home_pts5"])  # C 首场不能混入 A/B 的比赛
    # 对手 A 此时已赛 5 场（含与 C 当场之前的 5 场）
    assert c_match["away_pts5"] == 3.0


def test_rest_days_from_previous_match_only() -> None:
    feat = build_features(_mini_league())
    # 第一场两队均无 rest；第二场两队都在首日刚出过场，间隔都是 7 天
    second = feat.iloc[1]
    assert second["home_rest_days"] == 7
    assert second["away_rest_days"] == 7
    assert pd.isna(feat.iloc[0]["home_rest_days"])


def test_b365_fallback_when_pinnacle_missing() -> None:
    df = _mini_league()
    df.loc[0, ["psh", "psd", "psa"]] = np.nan
    feat = build_features(df)
    assert feat.loc[0, "ps_available"] == 0
    # 回退到 B365 2.0/3.3/4.0 的去水主胜概率
    expected_h = 0.5 / (0.5 + 1 / 3.3 + 0.25)
    assert abs(feat.loc[0, "odds_h"] - expected_h) < 1e-9
    assert feat.loc[1, "ps_available"] == 1
