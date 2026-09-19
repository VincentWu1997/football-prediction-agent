"""赔率去水（de-vig）：把含博彩公司毛利的赔率转为公平概率。

方法：比例归一化（multiplicative / overround normalization）
    implied_k = 1/odds_k；p_k = implied_k / Σ implied
该方法保持各结果隐含概率的相对比例。W3 可对照 Shin / power 方法做敏感性分析。
"""

import numpy as np
import pandas as pd


def devig(odds_h: pd.Series, odds_d: pd.Series, odds_a: pd.Series) -> pd.DataFrame:
    """三列赔率 -> H/D/A 公平概率 DataFrame。任一列缺失则该行结果为 NaN。"""
    odds = pd.concat([odds_h, odds_d, odds_a], axis=1).astype(float)
    implied = 1.0 / odds
    sums = implied.sum(axis=1)
    probs = implied.div(sums, axis=0)
    probs.columns = ["H", "D", "A"]
    # 赔率非法（<=1 等异常）时标记为缺失
    bad = (odds <= 1.0).any(axis=1) | (sums <= 1.0)
    probs.loc[bad, :] = np.nan
    return probs
