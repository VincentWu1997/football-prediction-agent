"""蒙特卡洛仿真性质测试。"""

import numpy as np

from sports_agent.ml.dixon_coles import DixonColesModel
from sports_agent.ml.monte_carlo import simulate_match, simulate_season_fixtures


def _model() -> DixonColesModel:
    return DixonColesModel(
        attack={"A": 0.3, "B": -0.3, "C": 0.0, "D": 0.0},
        defense={"A": -0.2, "B": 0.2, "C": 0.0, "D": 0.0},
        intercept=np.log(1.3),
        home_adv=0.25,
        rho=-0.05,
    )


def test_simulate_match_matches_analytic_probs() -> None:
    model = _model()
    sim = simulate_match(model, "A", "B", n_sims=40_000, seed=7)
    p_h, p_d, p_a = model.probs("A", "B")
    assert abs(sim.probs["H"] - p_h) < 0.01
    assert abs(sim.probs["D"] - p_d) < 0.01
    assert abs(sim.probs["A"] - p_a) < 0.01


def test_simulate_match_summary_fields() -> None:
    sim = simulate_match(_model(), "A", "B", n_sims=5_000, seed=7)
    assert abs(sum(sim.probs.values()) - 1.0) < 1e-9
    assert all(
        lo <= p <= hi
        for p, (lo, hi) in zip(sim.probs.values(), sim.cis.values(), strict=True)
    )
    # top_scores 降序且概率和不超过 1
    probs = [s["prob"] for s in sim.top_scores]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0 + 1e-9
    assert 0.0 <= sim.over_25 <= 1.0
    assert 0.0 <= sim.btts <= 1.0


def test_season_simulation_rank_probabilities() -> None:
    teams = ["A", "B", "C", "D"]
    points = {"A": 40, "B": 35, "C": 30, "D": 20}
    fixtures = [("A", "D"), ("B", "C"), ("D", "A"), ("C", "B")]
    sim = simulate_season_fixtures(
        _model(),
        teams=teams,
        current_points=points,
        current_gd={t: 0 for t in teams},
        current_gf={t: 0 for t in teams},
        fixtures=fixtures,
        n_runs=8_000,
        seed=7,
    )
    # 冠军/前四概率归一（每 run 恰好一个冠军）
    assert abs(sum(sim.title.values()) - 1.0) < 1e-9
    assert abs(sum(sim.top4.values()) - 4.0) < 1e-9
    # 4 队联赛 rel_n=2
    assert abs(sum(sim.relegation.values()) - 2.0) < 1e-9
    # A 领先 5 分且仅剩对 D 的两回合，夺冠概率应显著最高
    assert max(sim.title, key=sim.title.get) == "A"
    assert sim.title["A"] > 0.5
    # D 实力最弱且积分垫底，降级概率最高
    assert max(sim.relegation, key=sim.relegation.get) == "D"
    # 期望积分应高于当前积分（还有比赛可打，两场对 D 期望约 +3）
    assert sim.expected_points["A"] > points["A"] + 2
