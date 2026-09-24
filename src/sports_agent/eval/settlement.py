"""pred_ledger 回填与结算。

从 f_matches 查实际结果，计算 log_loss/brier/rps，回写 pred_ledger。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from sports_agent.eval.metrics import brier_score, log_loss, rps
from sports_agent.settings import settings

# 1X2 → 数值索引（H=0, D=1, A=2）
_OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}


def _get_engine() -> Engine:
    """从 settings 获取 SQLAlchemy engine。"""
    return create_engine(settings.database_url)


def settle_unsettled(engine: Engine | None = None) -> dict:
    """回填所有未结算的 pred_ledger 记录。

    流程：
    1. 查 unsettled 记录，通过 match_id 关联 f_matches 拿 actual_outcome；
    2. 对每条算 log_loss / brier / rps；
    3. 批量 UPDATE。

    Returns:
        {"settled": N, "skipped_no_result": N, "errors": N}
    """
    eng = engine or _get_engine()

    # 查未结算且有 match_id 的记录
    rows = pd.read_sql(text("""
        SELECT pl.prediction_id, pl.match_id,
               pl.prob_home, pl.prob_draw, pl.prob_away,
               fm.full_time_res
        FROM pred_ledger pl
        LEFT JOIN f_matches fm ON pl.match_id = fm.match_id
        WHERE pl.settled_at IS NULL
          AND pl.match_id IS NOT NULL
    """), eng)

    settled = 0
    skipped = 0
    errors = 0

    for _, row in rows.iterrows():
        outcome = row["full_time_res"]
        pid = row["prediction_id"]

        if pd.isna(outcome) or outcome not in _OUTCOME_MAP:
            skipped += 1
            continue

        probs = np.array([[row["prob_home"], row["prob_draw"], row["prob_away"]]])
        outcomes = pd.Series([outcome])

        try:
            ll = float(log_loss(probs, outcomes))
            br = float(brier_score(probs, outcomes))
            rp = float(rps(probs, outcomes))
        except Exception:
            errors += 1
            continue

        with eng.begin() as conn:
            conn.execute(text("""
                UPDATE pred_ledger
                SET actual_outcome = :outcome,
                    log_loss = :ll,
                    brier = :br,
                    rps = :rp,
                    settled_at = now()
                WHERE prediction_id = :pid
            """), {
                "outcome": outcome,
                "ll": ll,
                "br": br,
                "rp": rp,
                "pid": pid,
            })
        settled += 1

    return {"settled": settled, "skipped_no_result": skipped, "errors": errors}


def ledger_summary(engine: Engine | None = None) -> dict:
    """pred_ledger 指标汇总（前端看板用）。"""
    eng = engine or _get_engine()

    with eng.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM pred_ledger")).scalar() or 0
        unsettled = conn.execute(text(
            "SELECT COUNT(*) FROM pred_ledger WHERE settled_at IS NULL"
        )).scalar() or 0

        settled_row = conn.execute(text("""
            SELECT COUNT(*) AS n,
                   AVG(log_loss) AS avg_ll,
                   AVG(brier)    AS avg_br,
                   AVG(rps)      AS avg_rps
            FROM pred_ledger WHERE settled_at IS NOT NULL
        """)).mappings().first()

        # 准确率（argmax 对比 actual_outcome）
        acc_row = conn.execute(text("""
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE actual_outcome =
                       CASE WHEN prob_home >= prob_draw AND prob_home >= prob_away THEN 'H'
                             WHEN prob_away >= prob_home AND prob_away >= prob_draw THEN 'A'
                             ELSE 'D' END) AS correct
            FROM pred_ledger WHERE settled_at IS NOT NULL
        """)).mappings().first()

    return {
        "total": total,
        "unsettled": unsettled,
        "settled": settled_row["n"] if settled_row else 0,
        "avg_log_loss": _safe_float(settled_row, "avg_ll"),
        "avg_brier": _safe_float(settled_row, "avg_br"),
        "avg_rps": _safe_float(settled_row, "avg_rps"),
        "accuracy": _safe_accuracy(acc_row),
    }


def _safe_float(row, key: str) -> float | None:
    if row and row.get(key):
        return float(row[key])
    return None


def _safe_accuracy(row) -> float | None:
    if row and row.get("total", 0) > 0:
        return row["correct"] / row["total"]
    return None
