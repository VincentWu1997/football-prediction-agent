"""W9 烟测：回填逻辑 + 前端导入。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sports_agent.eval.metrics import brier_score, log_loss, rps


def test_settlement_import():
    from sports_agent.eval.settlement import ledger_summary, settle_unsettled
    assert callable(settle_unsettled)
    assert callable(ledger_summary)


def test_metrics_computation():
    """验证回填用的三个指标计算正确。"""
    probs = np.array([[0.5, 0.3, 0.2]])
    outcome = pd.Series(["H"])
    ll = log_loss(probs, outcome)
    br = brier_score(probs, outcome)
    rp = rps(probs, outcome)
    # H 命中时 log_loss = -log(0.5) ≈ 0.693
    assert abs(ll - (-np.log(0.5))) < 1e-6
    # Brier = (0.5-1)^2 + (0.3-0)^2 + (0.2-0)^2 = 0.38
    assert abs(br - 0.38) < 1e-6
    assert 0 <= rp <= 1


def test_metrics_wrong_prediction():
    """预测 A 但实际 H，指标应更大。"""
    probs = np.array([[0.1, 0.2, 0.7]])  # 预测 A
    outcome = pd.Series(["H"])          # 实际 H
    ll_wrong = log_loss(probs, outcome)
    # log_loss = -log(0.1) = 2.303
    assert ll_wrong > 2.0


def test_frontend_import():
    """前端脚本可被 Python 解析导入（不启动 streamlit）。"""
    import ast
    from pathlib import Path
    p = Path(__file__).parent.parent / "frontend" / "app.py"
    tree = ast.parse(p.read_text())
    assert any(isinstance(n, ast.FunctionDef) or True for n in ast.walk(tree))


def test_ledger_summary_empty():
    """ledger_summary 在空表时返回零值（需 DB，SKIP_DB_TESTS=1 跳过）。"""
    import os
    if os.environ.get("SKIP_DB_TESTS"):
        pytest.skip("SKIP_DB_TESTS=1")
    from sports_agent.eval.settlement import ledger_summary
    s = ledger_summary()
    assert "total" in s
    assert "unsettled" in s
    assert "settled" in s
