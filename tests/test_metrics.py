"""指标手算例验证。"""

import numpy as np
import pandas as pd

from sports_agent.eval.metrics import accuracy, brier_score, log_loss, rps
from sports_agent.eval.odds import devig

# 单样本：预测 (0.5, 0.3, 0.2)，实际客胜 A
PROBS = np.array([[0.5, 0.3, 0.2]])
OUTCOMES = pd.Series(["A"])


def test_logloss_hand_computed() -> None:
    # -ln(0.2)
    assert abs(log_loss(PROBS, OUTCOMES) - 1.6094379) < 1e-6


def test_brier_hand_computed() -> None:
    # 多类 Brier 标准定义：Σ_k (p_k-y_k)^2（不除以类别数）
    assert abs(brier_score(PROBS, OUTCOMES) - 0.98) < 1e-9


def test_rps_hand_computed() -> None:
    # cumP=[0.5,0.8], cumY=[0,0] -> (0.25+0.64)/2
    assert abs(rps(PROBS, OUTCOMES) - 0.445) < 1e-9


def test_accuracy_argmax() -> None:
    assert accuracy(PROBS, OUTCOMES) == 0.0
    assert accuracy(np.array([[0.1, 0.1, 0.8]]), pd.Series(["A"])) == 1.0


def test_perfect_prediction_scores_best() -> None:
    perfect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    outcomes = pd.Series(["H", "D"])
    assert log_loss(perfect, outcomes) < 1e-9
    assert brier_score(perfect, outcomes) < 1e-9
    assert rps(perfect, outcomes) < 1e-9


def test_devig_sums_to_one_and_strips_margin() -> None:
    probs = devig(pd.Series([2.0]), pd.Series([3.5]), pd.Series([4.0]))
    row = probs.iloc[0]
    assert abs(row.sum() - 1.0) < 1e-9
    # 原始 overround：0.5+0.2857+0.25=1.0357，归一后主胜概率应低于原始隐含 0.5
    assert row["H"] < 0.5
