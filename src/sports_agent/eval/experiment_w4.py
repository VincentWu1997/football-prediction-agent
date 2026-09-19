"""W4 实验：五联赛 2526 赛季中途（as_of=2026-01-10）剩余赛程蒙特卡洛。

运行：python -m sports_agent.eval.experiment_w4
产物（benchmarks/results/w4/）：
- season_simulation_<league>.csv  各队夺冠/前四/降级概率 + 期望积分 + 真实最终排名
- single_match_demo.json          单场预测（DC 概率 + 比分分布）示例

注意：真实最终排名在 as_of 时点不可知，仅用于事后回看校验模拟质量。
"""

import json
import sys
import time

import pandas as pd

from sports_agent.ml.predict_service import load_matches, predict_match, simulate_season
from sports_agent.settings import REPO_ROOT

OUT_DIR = REPO_ROOT / "benchmarks" / "results" / "w4"
AS_OF = "2026-01-10"
N_RUNS = 20_000
LEAGUES = ("E0", "SP1", "D1", "I1", "F1")


def run() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for league in LEAGUES:
        result = simulate_season(league, AS_OF, n_runs=N_RUNS, seed=42)
        actual_rank = {t: i + 1 for i, t in enumerate(result["actual_final_order"])}
        table = pd.DataFrame(
            {
                "team": result["teams"],
                "title_prob": [result["title"][t] for t in result["teams"]],
                "top4_prob": [result["top4"][t] for t in result["teams"]],
                "relegation_prob": [result["relegation"][t] for t in result["teams"]],
                "expected_points": [
                    round(result["expected_points"][t], 1) for t in result["teams"]
                ],
                "actual_final_rank": [actual_rank[t] for t in result["teams"]],
            }
        ).sort_values("title_prob", ascending=False)
        path = OUT_DIR / f"season_simulation_{league}.csv"
        table.to_csv(path, index=False)

        # 模拟冠军 vs 真实冠军；模拟降级 3 队 vs 真实降级 3 队（事后校验）
        sim_champion = table.iloc[0]["team"]
        real_champion = result["actual_final_order"][0]
        sim_relegated = set(table.nlargest(3, "relegation_prob")["team"])
        real_relegated = set(result["actual_final_order"][-3:])
        print(
            f"[{league}] 剩余 {result['n_remaining']} 场 | "
            f"模拟冠军 {sim_champion} vs 真实冠军 {real_champion} | "
            f"降级命中 {len(sim_relegated & real_relegated)}/3"
        )

    demo = predict_match("Manchester United", "Liverpool", n_sims=50_000)
    (OUT_DIR / "single_match_demo.json").write_text(
        json.dumps(demo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_matches = len(load_matches())
    print(f"\n数据总场次 {total_matches:,}；单场示例已写入 single_match_demo.json")
    print(f"耗时 {time.perf_counter() - t0:.1f}s；产物：{OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(run())
