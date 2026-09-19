"""回测结算手算例与边界。"""

import numpy as np
import pandas as pd

from sports_agent.ml.backtest import run_backtest

PROBS = np.array(
    [
        [0.60, 0.30, 0.10],  # EV(H)=0.2, EV(D)=0.05, EV(A)=-0.6
        [0.40, 0.30, 0.30],  # EV(H)=-0.2, EV(D)=-0.1, EV(A)=0.2
    ]
)
ODDS = np.array([[2.0, 3.5, 4.0], [2.0, 3.0, 4.0]])
OUTCOMES = pd.Series(["H", "H"])  # 第一场主胜命中，第二场客胜注单未中
DATES = pd.Series(pd.to_datetime(["2024-08-01", "2024-08-08"]))


def test_flat_stake_settlement() -> None:
    r = run_backtest(PROBS, ODDS, OUTCOMES, DATES, edge_threshold=0.0)
    assert r.n_bets == 2
    assert r.n_hits == 1
    # 第一场净赚 2.0-1=1.0；第二场输 1.0
    assert r.pnl == 0.0
    assert r.roi == 0.0
    assert list(r.bets["side"]) == ["H", "A"]


def test_edge_threshold_filters_bets() -> None:
    # 两笔最大 EV 都只有 0.2，阈值 0.25 时无注单
    r = run_backtest(PROBS, ODDS, OUTCOMES, DATES, edge_threshold=0.25)
    assert r.n_bets == 0
    assert r.roi == 0.0


def test_max_drawdown_tracked() -> None:
    # 强自信主胜：三场都选 H；赛果为 H/D/D -> 资金曲线 +1, 0, -1
    probs = np.array([[0.9, 0.05, 0.05]] * 3)
    odds = np.array([[2.0, 3.5, 4.0], [1.5, 4.0, 8.0], [1.2, 6.0, 15.0]])
    outcomes = pd.Series(["H", "D", "D"])
    dates = pd.Series(pd.to_datetime(["2024-08-01", "2024-08-08", "2024-08-15"]))
    r = run_backtest(probs, odds, outcomes, dates, edge_threshold=0.0)
    assert list(r.bets["side"]) == ["H", "H", "H"]
    # 资金曲线 +1, 0, -1：从峰值 1 到谷底 -1 的回撤为 2 个单位
    assert r.pnl == -1.0
    assert r.max_drawdown == -2.0
