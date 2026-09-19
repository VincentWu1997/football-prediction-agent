"""XGBoost 多分类（softprob）+ 两种折内校准。

防泄漏：
- 校准器只在训练折内部的时序后段（cal 段）拟合，绝不触碰测试赛季；
- 流程：训练折按时间 80/20 切 base/cal → base 训 XGB → cal 拟合校准器 →
  用全训练折重新拟合 XGB（更多数据），校准器沿用；
- 超参在代码中固定（浅层 + 强正则），W3 不做面向测试集的调参。

W3 实测（每联赛每折 cal 段仅数百场）：逐类 isotonic 过拟合，校准后
LogLoss 反而变差；单参数温度缩放（T≥1）更稳健，故两者都保留并用数据说话。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from sports_agent.ml._omp import preload_libomp

preload_libomp()

# 浅层保守参数：每联赛训练样本仅数千场
XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 20,
    "reg_lambda": 2.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_jobs": -1,
    "random_state": 42,
}
CAL_FRACTION = 0.2  # 训练折末尾 20% 作为校准段


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """在 cal 段上以最小化 LogLoss 拟合单一温度 T（T≥1 只做置信度压缩）。"""

    def nll(log_t: float) -> float:
        probs = np.clip(_softmax(logits / np.exp(log_t)), 1e-12, 1.0)
        return float(-np.log(probs[np.arange(len(labels)), labels]).mean())

    result = minimize_scalar(nll, bounds=(0.0, np.log(8.0)), method="bounded")
    return float(np.exp(result.x))


@dataclass
class CalibratedXGB:
    model: XGBClassifier
    calibrators: list[IsotonicRegression]
    temperature: float

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_temperature(self, X: pd.DataFrame) -> np.ndarray:
        """logits / T 后 softmax（单参数，小样本下不易过拟合）。"""
        logits = self.model.predict(X, output_margin=True)
        return _softmax(np.asarray(logits) / self.temperature)

    def predict_isotonic(self, X: pd.DataFrame) -> np.ndarray:
        """softprob → 逐类 isotonic → 截断 + L1 归一。"""
        raw = self.predict_raw(X)
        cal = np.column_stack(
            [iso.predict(raw[:, k]) for k, iso in enumerate(self.calibrators)]
        )
        cal = np.clip(cal, 1e-6, 1.0)
        return cal / cal.sum(axis=1, keepdims=True)


def train_calibrated_xgb(train_df: pd.DataFrame, features: list[str]) -> CalibratedXGB:
    """训练折内完成 XGB 拟合、温度缩放与 isotonic 校准。"""
    ordered = train_df.sort_values("match_date")
    split_idx = int(len(ordered) * (1 - CAL_FRACTION))
    base, cal = ordered.iloc[:split_idx], ordered.iloc[split_idx:]

    # 1) base 段训练，cal 段拟合两种校准器
    base_model = XGBClassifier(**XGB_PARAMS)
    base_model.fit(base[features], base["label"])
    cal_raw = base_model.predict_proba(cal[features])
    cal_logits = np.asarray(base_model.predict(cal[features], output_margin=True))
    temperature = _fit_temperature(cal_logits, cal["label"].to_numpy())

    calibrators = [
        IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        for _ in range(3)
    ]
    for k, iso in enumerate(calibrators):
        y_k = (cal["label"].to_numpy() == k).astype(float)
        iso.fit(cal_raw[:, k], y_k)

    # 2) 全训练折重训主模型（校准器保持来自独立 cal 段）
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(ordered[features], ordered["label"])
    return CalibratedXGB(
        model=model, calibrators=calibrators, temperature=temperature
    )
