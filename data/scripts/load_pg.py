#!/usr/bin/env python3
"""W1 幂等入库：data/processed/matches.csv -> PostgreSQL f_matches / f_odds。

幂等保证（可安全重复执行）：
- f_matches：业务唯一约束冲突时 DO NOTHING；
- f_odds：以 match_id 为主键，冲突时更新全部赔率列（重跑即刷新数据）。

用法：
    python data/scripts/load_pg.py
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sports_agent.settings import settings  # noqa: E402

PROCESSED_CSV = REPO_ROOT / "data" / "processed" / "matches.csv"
MATCH_UNIQUE = "f_matches_league_season_match_date_home_team_away_team_key"
ODDS_COLUMNS = [
    "b365h",
    "b365d",
    "b365a",
    "psh",
    "psd",
    "psa",
    "avgh",
    "avgd",
    "avga",
]


def _to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> dict 列表。

    关键：NaN 必须显式转 None，否则 psycopg2 会把 float('nan') 适配成 PG 浮点 NaN
    （在 PostgreSQL 中 NaN 不是 NULL，会污染覆盖率统计与下游计算）。
    """
    match_df = df.copy()
    match_df["match_date"] = pd.to_datetime(match_df["match_date"]).dt.date
    object_df = match_df.astype(object).where(match_df.notna(), None)
    return object_df.to_dict("records")


def main() -> None:
    if not PROCESSED_CSV.exists():
        raise SystemExit("缺少 processed/matches.csv，请先运行 normalize.py")

    matches = pd.read_csv(PROCESSED_CSV)
    engine = create_engine(settings.database_url, future=True)
    metadata = MetaData()
    t_matches = Table("f_matches", metadata, autoload_with=engine)
    t_odds = Table("f_odds", metadata, autoload_with=engine)

    with engine.begin() as conn:
        # 1) 比赛：冲突忽略
        match_cols = [
            c.name
            for c in t_matches.columns
            if c.name in matches.columns and c.name != "match_id"
        ]
        records = _to_records(matches[match_cols])
        stmt = insert(t_matches).on_conflict_do_nothing(constraint=MATCH_UNIQUE)
        inserted = conn.execute(stmt, records).rowcount

        # 2) 取业务键 -> match_id 映射（整表约 2 万行，一次拉回）
        id_rows = conn.execute(
            select(
                t_matches.c.match_id,
                t_matches.c.league,
                t_matches.c.season,
                t_matches.c.match_date,
                t_matches.c.home_team,
                t_matches.c.away_team,
            )
        ).all()
        id_map = {
            (r.league, r.season, r.match_date, r.home_team, r.away_team): r.match_id
            for r in id_rows
        }

        # 3) 赔率：冲突更新
        odds_records = []
        for rec in _to_records(matches):
            # season 在库中为 TEXT，CSV 读入为 int，统一转 str 后再组键
            key = (
                rec["league"],
                str(rec["season"]),
                rec["match_date"],
                rec["home_team"],
                rec["away_team"],
            )
            odds_records.append(
                {"match_id": id_map[key], **{c: rec.get(c) for c in ODDS_COLUMNS}}
            )
        odds_stmt = insert(t_odds).on_conflict_do_update(
            constraint="f_odds_pkey",
            set_={c: getattr(insert(t_odds).excluded, c) for c in ODDS_COLUMNS},
        )
        odds_rows = conn.execute(odds_stmt, odds_records).rowcount

        total_matches = conn.execute(select(func.count()).select_from(t_matches)).scalar() or 0
        total_odds = conn.execute(select(func.count()).select_from(t_odds)).scalar() or 0

    print(f"本次新插入比赛：{inserted:,}；写入/更新赔率：{odds_rows:,}")
    print(f"库内累计：f_matches {total_matches:,} 行 / f_odds {total_odds:,} 行")


if __name__ == "__main__":
    main()
