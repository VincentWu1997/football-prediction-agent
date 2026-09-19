"""预测模型层（W2-W3/W4）。

规划模块：
- elo.py            ELO 评分（主场优势、差异化 K 值）
- dixon_coles.py    双变量 Poisson，输出比分分布与 1X2/大小球概率
- features.py       滚动状态、赛程密度、赔率漂移等特征
- xgb_model.py      XGBoost 主模型（walk-forward 训练）
- simulate.py       L4 剩余赛程蒙特卡洛模拟
"""
