"""W2 实验：walk-forward 下 ELO / Dixon-Coles / 赔率去水 baseline 对照。

运行：
    python -m sports_agent.eval.experiment

产物（全部基于赛前可得信息，赛后结果只用于评分）：
- benchmarks/results/w2/predictions.csv      每场×每模型的概率明细
- benchmarks/results/w2/model_comparison.csv 模型×联赛的 LogLoss/Brier/RPS/Accuracy
评估集合为三类预测都可用的交集（主要受 Pinnacle 覆盖率约束）。
"""

import sys
import time

import numpy as np
import pandas as pd

from sports_agent.eval.metrics import ALL_METRICS
from sports_agent.eval.odds import devig
from sports_agent.eval.walkforward import walk_forward_folds
from sports_agent.ml.dixon_coles import fit_dixon_coles
from sports_agent.ml.elo import (
    EloEngine,
    expected_home_score,
    fit_draw_curve,
    split_probs,
)
from sports_agent.settings import REPO_ROOT

MATCHES_CSV = REPO_ROOT / "data" / "processed" / "matches.csv"
OUT_DIR = REPO_ROOT / "benchmarks" / "results" / "w2"

MODELS = ("odds_ps", "elo", "dixon_coles")
LEAGUE_NAMES = {"E0": "英超", "SP1": "西甲", "D1": "德甲", "I1": "意甲", "F1": "法甲"}


def _predict_elo(test: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """用赛前评分差与训练折平局曲线产出 ELO 概率。"""
    probs = [
        split_probs(expected_home_score(dr), a * np.exp(-b * dr**2))
        for dr in test["elo_dr"].to_numpy()
    ]
    return pd.DataFrame(probs, columns=["H", "D", "A"], index=test.index)


def run() -> None:
    t0 = time.perf_counter()
    df = pd.read_csv(MATCHES_CSV, parse_dates=["match_date"])
    df["season"] = df["season"].astype(str)

    pred_frames: list[pd.DataFrame] = []
    fit_stats: list[dict] = []

    for league in LEAGUE_NAMES:
        lg = df[df["league"] == league].sort_values("match_date").copy()

        # ELO：全历史顺序回放一次，每场得到赛前评分差（无未来信息）
        elo = EloEngine()
        lg = elo.replay(lg)

        for test_season, train, test in walk_forward_folds(lg):
            # ELO 平局曲线只用训练折拟合
            a, b = fit_draw_curve(train["elo_dr"].to_numpy(), train["full_time_res"])
            elo_probs = _predict_elo(test, a, b)

            # Dixon-Coles：每个折叠重新拟合（训练样本按时间衰减加权）
            t_fit = time.perf_counter()
            dc = fit_dixon_coles(train)
            fit_seconds = time.perf_counter() - t_fit
            dc_probs = pd.DataFrame(
                [
                    dc.probs(h, a_)
                    for h, a_ in zip(test["home_team"], test["away_team"], strict=True)
                ],
                columns=["H", "D", "A"],
                index=test.index,
            )
            fit_stats.append(
                {"league": league, "season": test_season, "dc_fit_seconds": round(fit_seconds, 2)}
            )

            odds_probs = devig(test["psh"], test["psd"], test["psa"])

            meta = test[
                ["league", "season", "match_date", "home_team", "away_team", "full_time_res"]
            ]
            for model_name, probs in (
                ("odds_ps", odds_probs),
                ("elo", elo_probs),
                ("dixon_coles", dc_probs),
            ):
                chunk = meta.copy()
                chunk["model"] = model_name
                chunk[["prob_h", "prob_d", "prob_a"]] = probs[["H", "D", "A"]].to_numpy()
                pred_frames.append(chunk)

    predictions = pd.concat(pred_frames, ignore_index=True)

    # 三模型都可用的比赛（Pinnacle 赔率非空是主要约束）
    wide = predictions.pivot_table(
        index=["league", "season", "match_date", "home_team", "away_team", "full_time_res"],
        columns="model",
        values=["prob_h", "prob_d", "prob_a"],
    )
    valid = wide.dropna().reset_index()
    n_dropped = len(predictions) // len(MODELS) - len(valid)
    print(f"评估比赛数：{len(valid)}（因赔率缺失剔除 {n_dropped} 场）")

    rows = []
    for scope_name, scope_df in [
        ("OVERALL", valid),
        *[(lg, g) for lg, g in valid.groupby("league")],
    ]:
        for model in MODELS:
            probs = scope_df[[("prob_h", model), ("prob_d", model), ("prob_a", model)]].to_numpy()
            outcomes = scope_df["full_time_res"]
            row = {"scope": scope_name, "model": model, "n": len(scope_df)}
            row.update({name: fn(probs, outcomes) for name, fn in ALL_METRICS.items()})
            rows.append(row)

    comparison = pd.DataFrame(rows)
    for col in ("log_loss", "brier", "rps"):
        comparison[col] = comparison[col].round(4)
    comparison["accuracy"] = comparison["accuracy"].round(4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_out = predictions.dropna(subset=["prob_h", "prob_d", "prob_a"])
    predictions_out.to_csv(OUT_DIR / "predictions.csv", index=False)
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(fit_stats).to_csv(OUT_DIR / "dc_fit_times.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== 模型对照（OVERALL，越低越好，accuracy 越高越好）===")
    print(comparison[comparison["scope"] == "OVERALL"].to_string(index=False))
    print("\n=== 分联赛 LogLoss ===")
    pivot = comparison[comparison["scope"] != "OVERALL"].pivot(
        index="scope", columns="model", values="log_loss"
    )
    print(pivot.to_string())
    print(f"\n耗时 {time.perf_counter() - t0:.1f}s；产物目录：{OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(run())
