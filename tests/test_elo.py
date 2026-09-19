"""ELO 引擎基本性质测试。"""

import numpy as np
import pandas as pd

from sports_agent.ml.elo import (
    EloEngine,
    expected_home_score,
    fit_draw_curve,
    split_probs,
)


def test_expected_score_monotonic() -> None:
    assert expected_home_score(200) > 0.5
    assert expected_home_score(-200) < 0.5
    assert abs(expected_home_score(0) - 0.5) < 1e-9


def test_split_probs_normalized() -> None:
    p = split_probs(0.6, 0.28)
    assert abs(sum(p) - 1.0) < 1e-9
    assert all(x > 0 for x in p)


def test_upset_moves_ratings_more_than_expected_win() -> None:
    eng = EloEngine()
    eng.ratings["Strong"] = 1800
    eng.ratings["Weak"] = 1400
    strong_before, weak_before = 1800, 1400

    # 弱队主场爆冷击败强队
    eng.update("Weak", "Strong", "H")
    assert eng.ratings["Weak"] > weak_before
    assert eng.ratings["Strong"] < strong_before
    gain = eng.ratings["Weak"] - weak_before
    assert gain > 5  # 爆冷得分显著（K=20 量级）


def test_replay_uses_only_past_matches() -> None:
    # 赛后的评分差不应出现在该场的赛前 dr 上：第二场 dr 必须反映第一场结果
    matches = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2024-08-01", "2024-08-08"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "full_time_res": ["H", "H"],
        }
    )
    eng = EloEngine()
    out = eng.replay(matches)
    assert out["elo_dr"].iloc[0] == 65.0  # 双方均为基准分时只有主场优势
    assert out["elo_dr"].iloc[1] > out["elo_dr"].iloc[0]


def test_draw_curve_fit_peak_and_decay() -> None:
    rng = np.random.default_rng(0)
    dr = rng.uniform(-400, 400, size=2000)
    # 构造平局率 0.30*exp(-b dr^2) 的伯努利结局
    b = np.log(2) / 200**2
    p_draw = 0.30 * np.exp(-b * dr**2)
    outcomes = pd.Series(np.where(rng.uniform(size=2000) < p_draw, "D", "H"))
    a_hat, b_hat = fit_draw_curve(dr, outcomes)
    assert abs(a_hat - 0.30) < 0.03
    assert abs(b_hat - b) / b < 0.25
