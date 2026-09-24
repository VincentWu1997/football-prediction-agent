"""E4：构造前缀复用负载（方案 §5.4 RadixAttention 主场）。

生成 50 个 ShareGPT JSONL 请求，共享同一段固定长前缀（球队背景资料 system+context），
只有末尾提问不同——公共前缀缓存收益最大的场景。vLLM 与 SGLang 的
benchmark_serving 都支持 ShareGPT loader，保证两侧用同一负载公平对比。

输出：benchmarks/results/w7/prefix_workload.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 固定长公共前缀：模拟球队背景资料（system + context），约 300+ 字
_PREFIX = (
    "你是足球赛事分析助手。以下是利物浦足球俱乐部 2023-24 赛季的背景资料，请据此回答问题。"
    "利物浦本赛季在英超表现强势，防守端由门将阿利松领衔，范戴克与科纳特组成中卫搭档；"
    "中场在麦卡利斯特和索博斯洛伊加盟后攻守更加均衡，边路萨拉赫与努涅斯持续制造威胁。"
    "球队采用 4-3-3 阵型，强调高位压迫与快速转换，主场安菲尔德氛围热烈。"
    "近期赛程密集，需兼顾英超与欧战，教练会在部分场次轮换阵容。"
    "以下问题均基于上述资料作答。背景资料结束。\n\n"
)

_QUESTIONS = [
    "利物浦的主力门将是谁？",
    "中卫搭档是哪两位球员？",
    "中场新援有哪些？",
    "边路进攻核心是谁？",
    "球队常用什么阵型？",
    "球队的主场叫什么？",
    "球队的战术风格是怎样的？",
    "为什么近期需要轮换阵容？",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", default="benchmarks/results/w7/prefix_workload.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            q = _QUESTIONS[i % len(_QUESTIONS)] + f"（问题 {i + 1}）"
            row = {"conversations": [{"from": "human", "value": _PREFIX + q}]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.n} prefix-sharing ShareGPT rows -> {out}")


if __name__ == "__main__":
    main()
