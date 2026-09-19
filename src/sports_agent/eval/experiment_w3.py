"""W3 实验：XGBoost（raw + isotonic 校准）对照 W2 三模型 + 投注回测。

运行：python -m sports_agent.eval.experiment_w3
产物（benchmarks/results/w3/）：
- predictions.csv          逐场×模型概率（含 xgb_raw / xgb_calibrated）
- model_comparison.csv     LogLoss/Brier/RPS/Accuracy（评估集=PS 赔率非空）
- bets.csv                 回测注单明细
- backtest_summary.csv     flat-stake value betting：ROI/注数/命中率/最大回撤
"""

import sys
import time

import numpy as np
import pandas as pd

from sports_agent.eval.metrics import ALL_METRICS
from sports_agent.eval.odds import devig
from sports_agent.eval.walkforward import walk_forward_folds
from sports_agent.ml.backtest import run_backtest
from sports_agent.ml.dixon_coles import fit_dixon_coles
from sports_agent.ml.elo import (
    EloEngine,
    expected_home_score,
    fit_draw_curve,
    split_probs,
)
from sports_agent.ml.features import FEATURE_COLUMNS, attach_dc_lambdas, build_features
from sports_agent.ml.xgb_model import train_calibrated_xgb
from sports_agent.settings import REPO_ROOT

MATCHES_CSV = REPO_ROOT / "data" / "processed" / "matches.csv"
OUT_DIR = REPO_ROOT / "benchmarks" / "results" / "w3"

METRIC_MODELS = (
    "odds_ps",
    "elo",
    "dixon_coles",
    "xgb_raw",
    "xgb_temp",
    "xgb_isotonic",
)
# 回测只对未经 isotonic 扭曲的 raw / temp 进行
BACKTEST_MODELS = ("xgb_raw", "xgb_temp")
EDGE_THRESHOLDS = (0.0, 0.02, 0.05, 0.10)
LEAGUE_NAMES = {"E0": "英超", "SP1": "西甲", "D1": "德甲", "I1": "意甲", "F1": "法甲"}


def _elo_probs(test: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
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
    bet_frames: list[pd.DataFrame] = []
    odds_records: list[pd.DataFrame] = []  # 全部 PS 非空比赛（favorite 对照用）

    for league in LEAGUE_NAMES:
        lg = df[df["league"] == league].sort_values("match_date").copy()
        elo = EloEngine()
        lg = elo.replay(lg)
        # 滚动/赔率/H2H 特征联赛级一次性算完（全部只使用赛前信息），DC λ 分折填充
        features_all = build_features(lg, dc=None)

        for test_season, train_raw, test_raw in walk_forward_folds(lg):
            train_idx, test_idx = train_raw.index, test_raw.index
            a, b = fit_draw_curve(
                train_raw["elo_dr"].to_numpy(), train_raw["full_time_res"]
            )
            dc = fit_dixon_coles(train_raw)

            feat_train = attach_dc_lambdas(features_all.loc[train_idx].copy(), dc)
            feat_test = attach_dc_lambdas(features_all.loc[test_idx].copy(), dc)

            # 五模型概率
            probs_by_model: dict[str, pd.DataFrame] = {
                "odds_ps": devig(test_raw["psh"], test_raw["psd"], test_raw["psa"]),
                "elo": _elo_probs(test_raw, a, b),
                "dixon_coles": pd.DataFrame(
                    [
                        dc.probs(h, aw)
                        for h, aw in zip(
                            test_raw["home_team"], test_raw["away_team"], strict=True
                        )
                    ],
                    columns=["H", "D", "A"],
                    index=test_raw.index,
                ),
            }

            fitted = train_calibrated_xgb(feat_train, FEATURE_COLUMNS)
            xgb_versions = {
                "xgb_raw": fitted.predict_raw,
                "xgb_temp": fitted.predict_temperature,
                "xgb_isotonic": fitted.predict_isotonic,
            }
            for model_name, predict_fn in xgb_versions.items():
                probs_by_model[model_name] = pd.DataFrame(
                    predict_fn(feat_test[FEATURE_COLUMNS]),
                    columns=["H", "D", "A"],
                    index=test_idx,
                )

            meta = test_raw[
                ["league", "season", "match_date", "home_team", "away_team", "full_time_res"]
            ]
            for model_name, probs in probs_by_model.items():
                chunk = meta.copy()
                chunk["model"] = model_name
                chunk[["prob_h", "prob_d", "prob_a"]] = probs[["H", "D", "A"]].to_numpy()
                pred_frames.append(chunk)

            # 回测只在 PS 非空的比赛上（用 PS closing odds 结算）
            ps = test_raw[["psh", "psd", "psa"]].to_numpy(float)
            valid_ps = ~np.isnan(ps).any(axis=1)
            valid_test_index = test_raw.index[valid_ps]
            outcomes = test_raw["full_time_res"].loc[valid_test_index]
            dates = test_raw["match_date"].loc[valid_test_index]
            valid_odds = ps[valid_ps]

            odds_chunk = pd.DataFrame(valid_odds, columns=["psh", "psd", "psa"])
            odds_chunk["league"] = league
            odds_chunk["season"] = test_season
            odds_chunk["match_date"] = dates.to_numpy()
            odds_chunk["full_time_res"] = outcomes.to_numpy()
            odds_records.append(odds_chunk)

            for bt_model in BACKTEST_MODELS:
                model_p = probs_by_model[bt_model].loc[valid_test_index].to_numpy()
                for edge in EDGE_THRESHOLDS:
                    result = run_backtest(
                        model_p, valid_odds, outcomes, dates, edge_threshold=edge
                    )
                    if result.n_bets:
                        bets = result.bets
                        bets["league"] = league
                        bets["season"] = test_season
                        bets["model"] = bt_model
                        bets["edge_threshold"] = edge
                        bet_frames.append(bets)

    predictions = pd.concat(pred_frames, ignore_index=True)

    # 五模型共有样本（PS 非空决定）
    wide = predictions.pivot_table(
        index=["league", "season", "match_date", "home_team", "away_team", "full_time_res"],
        columns="model",
        values=["prob_h", "prob_d", "prob_a"],
    ).dropna()
    valid = wide.reset_index()
    print(f"评估比赛数：{len(valid)}")

    rows = []
    scopes = [("OVERALL", valid), *list(valid.groupby("league"))]
    for scope_name, scope_df in scopes:
        for model in METRIC_MODELS:
            probs = scope_df[[("prob_h", model), ("prob_d", model), ("prob_a", model)]].to_numpy()
            row = {"scope": scope_name, "model": model, "n": len(scope_df)}
            outcomes = scope_df["full_time_res"]
            row.update({name: fn(probs, outcomes) for name, fn in ALL_METRICS.items()})
            rows.append(row)
    comparison = pd.DataFrame(rows)
    for col in ("log_loss", "brier", "rps", "accuracy"):
        comparison[col] = comparison[col].round(4)

    # 注单汇总
    bets_all = pd.concat(bet_frames, ignore_index=True) if bet_frames else pd.DataFrame()
    bt_rows = []
    group_cols = ["model", "league", "season", "edge_threshold"]

    def _summarize(scope: str, g: pd.DataFrame) -> dict:
        g_sorted = g.sort_values("match_date")
        pnl = g_sorted["pnl"].sum()
        cum = g_sorted["pnl"].cumsum()
        dd = float((cum - cum.cummax().clip(lower=0)).min())
        return {
            "scope": scope,
            "model": g["model"].iloc[0],
            "edge": g["edge_threshold"].iloc[0],
            "n_bets": len(g),
            "bet_pct": round(len(g) / len(valid) * 100, 1),
            "hit_rate": round(g["hit"].mean(), 4),
            "avg_odds": round(g["odds"].mean(), 3),
            "roi": round(pnl / len(g), 4),
            "pnl": round(pnl, 1),
            "max_drawdown": round(dd, 1),
        }

    if not bets_all.empty:
        for keys, g in bets_all.groupby(group_cols):
            model, league, season, edge = keys
            bt_rows.append(_summarize(f"{model}:{league}/{season}", g))
        for (_model, _edge), g in bets_all.groupby(["model", "edge_threshold"]):
            bt_rows.append(_summarize("OVERALL", g))

    # Favorite 对照：每场固定押赔率最低的结果（期望长期亏损约一个 overround）
    fav_df = pd.concat(odds_records, ignore_index=True)
    fav_odds = fav_df[["psh", "psd", "psa"]].to_numpy()
    fav_side_idx = fav_odds.argmin(axis=1)
    fav_sides = np.array(["H", "D", "A"])[fav_side_idx]
    fav_hit = (fav_df["full_time_res"].to_numpy() == fav_sides)
    fav_taken_odds = fav_odds[np.arange(len(fav_df)), fav_side_idx]
    fav_bets = pd.DataFrame(
        {
            "match_date": fav_df["match_date"],
            "league": fav_df["league"],
            "season": fav_df["season"],
            "side": fav_sides,
            "odds": fav_taken_odds,
            "hit": fav_hit,
            "pnl": np.where(fav_hit, fav_taken_odds - 1.0, -1.0),
            "model": "favorite",
            "edge_threshold": -1,
        }
    )
    for league, g in fav_bets.groupby("league"):
        bt_rows.append(_summarize(f"favorite:{league}", g))
    bt_rows.append(_summarize("OVERALL_FAVORITE", fav_bets))
    backtest_summary = pd.DataFrame(bt_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    bets_all.to_csv(OUT_DIR / "bets.csv", index=False)
    backtest_summary.to_csv(OUT_DIR / "backtest_summary.csv", index=False)

    pd.set_option("display.width", 180)
    print("\n=== 模型对照（OVERALL）===")
    print(comparison[comparison["scope"] == "OVERALL"].to_string(index=False))
    print("\n=== 投注回测（flat 1u / PS closing odds，仅 OVERALL 行）===")
    if not backtest_summary.empty:
        print(
            backtest_summary[backtest_summary["scope"].isin(["OVERALL", "OVERALL_FAVORITE"])]
            .sort_values(["model", "edge"])
            .to_string(index=False)
        )
    print(f"\n耗时 {time.perf_counter() - t0:.1f}s；产物：{OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(run())
