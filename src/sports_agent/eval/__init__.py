"""评估层（W2）。

- 按日期 walk-forward，禁止随机切分
- 指标：LogLoss / Brier / RPS（有序结局）/ Accuracy / 校准(ECE)
- Baseline：赔率去水隐含概率（首选 Pinnacle closing）
- betting.py：价值投注回测（ROI / 命中率 / 最大回撤，必须带样本量）
"""
