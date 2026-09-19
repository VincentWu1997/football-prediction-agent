#!/usr/bin/env python3
"""W1 EDA：基于 processed/matches.csv 输出文本报告（图表留待 W2）。

用法：python data/scripts/eda.py
输出：data/processed/eda_report.txt（同时打印到 stdout）
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "data" / "processed" / "matches.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "eda_report.txt"

LEAGUE_NAMES = {"E0": "英超", "SP1": "西甲", "D1": "德甲", "I1": "意甲", "F1": "法甲"}
FIRMS = {
    "b365": ("b365h", "b365d", "b365a"),
    "ps": ("psh", "psd", "psa"),
    "avg": ("avgh", "avgd", "avga"),
}


def overround(df: pd.DataFrame, cols: tuple[str, str, str]) -> float:
    """平均返还水位倒数之和：1.0 为公平盘，>1.0 的部分即博彩公司毛利。"""
    h, d, a = cols
    sub = df[list(cols)].dropna()
    return float((1 / sub[h] + 1 / sub[d] + 1 / sub[a]).mean())


def main() -> None:
    df = pd.read_csv(CSV_PATH, parse_dates=["match_date"])
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text)
        lines.append(text)

    p("=" * 64)
    p("W1 EDA 报告（足球赛果与赔率数据）")
    p(f"样本：{len(df):,} 场 | {df['match_date'].min().date()} ~ {df['match_date'].max().date()}")
    p("=" * 64)

    # 1) 每联赛×赛季场次数（检查完整性）
    p("\n[1] 各联赛×赛季已完赛场次")
    pivot = df.pivot_table(
        index="league", columns="season", values="fthg", aggfunc="count", fill_value=0
    )
    p(pivot.to_string())

    # 2) 赛果分布
    p("\n[2] 主胜/平/客胜分布（%）")
    overall = df["full_time_res"].value_counts(normalize=True).reindex(["H", "D", "A"]) * 100
    p(f"全样本：H {overall['H']:.1f} / D {overall['D']:.1f} / A {overall['A']:.1f}")
    for lg, name in LEAGUE_NAMES.items():
        sub = df[df["league"] == lg]
        r = sub["full_time_res"].value_counts(normalize=True).reindex(["H", "D", "A"]) * 100
        p(f"  {lg} {name}: H {r['H']:.1f} / D {r['D']:.1f} / A {r['A']:.1f}  (n={len(sub)})")

    # 3) 进球与大小球
    p("\n[3] 进球与大小球")
    total_goals = df["fthg"] + df["ftag"]
    p(
        f"场均总进球 {total_goals.mean():.3f}（主 {df['fthg'].mean():.3f} / "
        f"客 {df['ftag'].mean():.3f}）"
    )
    p(f"大 2.5 球占比 {(total_goals > 2.5).mean() * 100:.1f}%")

    # 4) 赔率覆盖率
    p("\n[4] 赔率覆盖率（非空占比）")
    for firm, cols in FIRMS.items():
        cover = df[list(cols)].dropna().shape[0] / len(df) * 100
        p(f"  {firm:4s}: {cover:5.1f}%")

    # 5) 博彩公司毛利（overround）
    p("\n[5] 平均 overround（1/赔率 之和；越低越接近公平盘）")
    for firm, cols in FIRMS.items():
        ovr = overround(df, cols)
        p(f"  {firm:4s}: {ovr:.4f}（理论毛利率 {(ovr - 1) * 100:.2f}%）")

    # 6) 去水隐含概率 vs 实际频率（W2 baseline 的预检验：校准粗看）
    p("\n[6] Pinnacle 去水隐含概率 vs 实际频率（全样本粗检，W2 按赛季细化）")
    sub = df[["psh", "psd", "psa", "full_time_res"]].dropna()
    inv = 1 / sub[["psh", "psd", "psa"]]
    fair = inv.div(inv.sum(axis=1), axis=0)
    actual = pd.get_dummies(sub["full_time_res"])[["H", "D", "A"]].mean()
    for col, label, res in [("psh", "主胜", "H"), ("psd", "平", "D"), ("psa", "客胜", "A")]:
        p(
            f"  {label}: 去水隐含 {fair[col].mean() * 100:5.2f}% / "
            f"实际 {actual[res] * 100:5.2f}% / 差值 {(fair[col].mean() - actual[res]) * 100:+.2f}pp"
        )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入：{OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
