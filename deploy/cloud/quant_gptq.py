"""E2：用 GPTQModel 自量化 Qwen3-8B 为 4-bit GPTQ。

方案 §5.3 明确要求 GPTQ 量化脚本入仓，与官方 AWQ/FP8 checkpoint 做三方对照。

云机内运行：
    HF_ENDPOINT=https://hf-mirror.com python deploy/cloud/quant_gptq.py

产物：./Qwen3-8B-GPTQ/（含 config.json / 模型权重 / tokenizer），
之后用 `bash deploy/cloud/10_serve.sh gptq` 启动。

说明：
- 校准数据用少量中文通用 + 体育领域样本（GPTQ 对校准集不敏感，128 条足够）；
- bits=4 / group_size=128 / desc_act=True，与 AWQ 的 4bit/128group 对齐口径；
- 量化是一次性操作（~10-15 分钟 on 4090），产物落盘后 serve 复用。
"""

from __future__ import annotations

import os

MODEL_ID = "Qwen/Qwen3-8B"
OUT_DIR = os.environ.get("GPTQ_DIR", "./Qwen3-8B-GPTQ")

# GPTQ 校准样本：混合通用中文与足球领域（无需外部数据集，直接内嵌，保证可复现）
_CALIBRATION = [
    "请分析这场足球比赛的胜负平概率。",
    "曼城主场对阵利物浦，双方近期状态如何？",
    "英超积分榜上，哪些球队有希望进入前四？",
    "详细说明 Dixon-Coles 模型如何估计进球期望。",
    "伤停信息对单场预测结果有什么影响？",
    "蒙特卡洛模拟一万次后，各队夺冠概率是多少？",
    "请给出主胜、平局、客胜的概率分布。",
    "拜仁慕尼黑在德甲的统治力来自哪些方面？",
] * 16  # 128 条


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from gptqmodel import GPTQModel, QuantizeConfig

    print(f"加载模型: {MODEL_ID}")
    print(f"输出目录: {OUT_DIR}")
    print(f"校准样本: {len(_CALIBRATION)} 条")

    # 4bit / group 128 / desc_act=True：与 AWQ 4bit 口径对齐
    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
        damp_percent=0.1,
    )

    # 量化（GPTQModel 内部自动加载 tokenizer，传校准文本即可）
    model = GPTQModel.load(MODEL_ID, quant_config)
    model.quantize(_CALIBRATION)

    print(f"保存量化模型到 {OUT_DIR}")
    model.save(OUT_DIR)
    print("完成。下一步：bash deploy/cloud/10_serve.sh gptq")


if __name__ == "__main__":
    main()
