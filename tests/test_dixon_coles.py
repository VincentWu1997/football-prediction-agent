"""Dixon-Coles 拟合与比分网格性质测试。"""

import numpy as np
import pandas as pd

from sports_agent.ml.dixon_coles import DixonColesModel, fit_dixon_coles


def test_probs_sum_to_one_and_home_advantage() -> None:
    # 同实力球队 + 正主场优势 -> 主胜概率应高于客胜
    model = DixonColesModel(
        attack={"A": 0.0, "B": 0.0},
        defense={"A": 0.0, "B": 0.0},
        intercept=np.log(1.4),
        home_adv=0.3,
        rho=-0.05,
    )
    p_h, p_d, p_a = model.probs("A", "B")
    assert abs(p_h + p_d + p_a - 1.0) < 1e-9
    assert p_h > p_a


def test_unseen_team_uses_neutral_params() -> None:
    model = DixonColesModel(
        attack={"A": 0.5}, defense={"A": -0.2}, intercept=np.log(1.4), home_adv=0.2, rho=0.0
    )
    p_h, _, p_a = model.probs("A", "NewlyPromoted")
    assert 0.0 < p_a < p_h < 1.0  # A 仍占优，但不报错


def _synthetic_league(seed: int = 42, n_matches: int = 1200) -> pd.DataFrame:
    """4 队合成联赛：强弱分明、有主场优势，Poisson 生成比分。"""
    rng = np.random.default_rng(seed)
    teams = ["Strong", "Mid1", "Mid2", "Weak"]
    strengths = {"Strong": 0.45, "Mid1": 0.1, "Mid2": -0.1, "Weak": -0.45}
    rows = []
    base_date = pd.Timestamp("2024-08-01")
    for i in range(n_matches):
        h, a = rng.choice(teams, size=2, replace=False)
        lh = np.exp(0.3 + 0.35 + strengths[h] - strengths[a])
        la = np.exp(0.3 + strengths[a] - strengths[h])
        rows.append(
            {
                "match_date": base_date + pd.Timedelta(days=i % 300),
                "home_team": h,
                "away_team": a,
                "fthg": rng.poisson(lh),
                "ftag": rng.poisson(la),
            }
        )
    return pd.DataFrame(rows)


def test_fit_recovers_attack_ranking() -> None:
    train = _synthetic_league()
    model = fit_dixon_coles(train, xi=0.0)  # 合成数据无时间趋势，不衰减
    assert model.attack["Strong"] > model.attack["Weak"]
    assert model.home_adv > 0
    assert -0.2 < model.rho < 0.2

    # 强强对话主胜概率应高于弱队主场打强队
    p_strong_home, _, _ = model.probs("Strong", "Weak")
    p_weak_home, _, _ = model.probs("Weak", "Strong")
    assert p_strong_home > p_weak_home
