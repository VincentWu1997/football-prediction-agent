"""概率预测评估指标。

所有函数输入对齐到 1X2 三类顺序：(主胜 H, 平 D, 客胜 A)。
- Log Loss：概率质量，对错误自信的预测惩罚大
- Brier Score：多类概率均方误差
- RPS（Ranked Probability Score）：有序结局的标准指标，奖励"接近"的概率分布
"""

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")
_EPS = 1e-12


def _to_arrays(probs: np.ndarray, outcomes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    # 固定类别顺序，保证某些样本缺失某类结果时列仍对齐到 H/D/A
    cats = outcomes.astype("category")
    cats = cats.cat.set_categories(list(OUTCOMES))
    y = pd.get_dummies(cats).to_numpy(float)
    if p.shape != y.shape:
        raise ValueError(f"概率形状 {p.shape} 与结局形状 {y.shape} 不一致")
    return p, y


def log_loss(probs: np.ndarray, outcomes: pd.Series) -> float:
    """多类 LogLoss = -mean(log p_actual)。"""
    p, y = _to_arrays(probs, outcomes)
    p_safe = np.clip(p, _EPS, 1.0)
    return float(-(y * np.log(p_safe)).sum(axis=1).mean())


def brier_score(probs: np.ndarray, outcomes: pd.Series) -> float:
    """多类 Brier = mean Σ_k (p_k - y_k)^2。"""
    p, y = _to_arrays(probs, outcomes)
    return float(((p - y) ** 2).sum(axis=1).mean())


def rps(probs: np.ndarray, outcomes: pd.Series) -> float:
    """Ranked Probability Score（三类，除以 r-1=2）。

    RPS = mean [ 1/(r-1) Σ_{k=1..r-1} (cumP_k - cumY_k)^2 ]
    值越低越好，完美预测为 0。
    """
    p, y = _to_arrays(probs, outcomes)
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_y = np.cumsum(y, axis=1)[:, :-1]
    return float(((cum_p - cum_y) ** 2).sum(axis=1).mean() / (p.shape[1] - 1))


def accuracy(probs: np.ndarray, outcomes: pd.Series) -> float:
    """argmax 命中率（仅作辅助展示）。"""
    p, y = _to_arrays(probs, outcomes)
    return float((p.argmax(axis=1) == y.argmax(axis=1)).mean())


ALL_METRICS = {
    "log_loss": log_loss,
    "brier": brier_score,
    "rps": rps,
    "accuracy": accuracy,
}
