"""XGBoost 训练与校准冒烟测试（小样本合成数据）。"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sports_agent.ml.xgb_model import train_calibrated_xgb

FEATURES = ["elo_dr", "odds_h", "odds_d", "odds_a"]


def _synthetic(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dr = rng.normal(0, 150, n)
    # 标签随评分差倾向主胜/客胜
    p_h = 1 / (1 + np.exp(-dr / 200))
    label = (rng.uniform(size=n) > p_h).astype(int)
    label = np.where(rng.uniform(size=n) < 0.2, 2, label)  # 少量客胜
    return pd.DataFrame(
        {
            "match_date": pd.date_range("2020-08-01", periods=n, freq="3D"),
            "elo_dr": dr,
            "odds_h": p_h,
            "odds_d": rng.uniform(0.2, 0.35, n),
            "odds_a": 1 - p_h - 0.25,
            "label": label,
        }
    )


def test_train_and_calibrated_output_is_valid() -> None:
    train = _synthetic()
    fitted = train_calibrated_xgb(train, FEATURES)
    X = train[FEATURES].iloc[:20]
    versions = (
        fitted.predict_raw(X),
        fitted.predict_temperature(X),
        fitted.predict_isotonic(X),
    )
    for probs in versions:
        assert probs.shape == (20, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert ((probs >= 0) & (probs <= 1)).all()

    # 温度 T 只压缩置信度：temp 版的最大概率不应超过 raw
    raw = fitted.predict_raw(X)
    temp = fitted.predict_temperature(X)
    assert fitted.temperature >= 1.0 - 1e-9
    assert temp.max() <= raw.max() + 1e-6

    # isotonic 是逐类单调映射，与 raw 的秩相关为正
    X200 = train[FEATURES].iloc[:200]
    raw200 = fitted.predict_raw(X200)
    iso = fitted.predict_isotonic(X200)
    corr, _ = spearmanr(raw200.ravel(), iso.ravel())
    assert corr > 0.9
