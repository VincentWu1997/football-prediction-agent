#!/usr/bin/env python3
"""W1 数据清洗：data/raw/*.csv -> data/processed/matches.csv + master/teams.csv。

已确认的数据事实（2026-09 实测）：
- 2021 赛季后的文件带 UTF-8 BOM，必须用 utf-8-sig 读取；
- PSH/PSD/PSA（Pinnacle）1516 赛季起齐全；AvgH/D/A 仅 2019 赛季后存在，缺失记 NaN；
- 五大联赛 10 年内无"同队异写"（Ajaccio 与 Ajaccio GFCO 是两支不同球队，不合并），
  故不引入别名表，以 (league, team_name) 作为稳定球队主键。

用法：
    python data/scripts/normalize.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MASTER_DIR = REPO_ROOT / "data" / "master"

# 原始列 -> 规范列名；核心列缺一不可，赔率列允许整列缺失
CORE_COLUMNS = {
    "Date": "match_date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "fthg",
    "FTAG": "ftag",
    "FTR": "full_time_res",
}
ODDS_COLUMNS = {
    "B365H": "b365h",
    "B365D": "b365d",
    "B365A": "b365a",
    "PSH": "psh",
    "PSD": "psd",
    "PSA": "psa",
    "AvgH": "avgh",
    "AvgD": "avgd",
    "AvgA": "avga",
}
ODDS_TARGETS = list(ODDS_COLUMNS.values())
OUTPUT_COLUMNS = [
    "league",
    "season",
    "match_date",
    "home_team",
    "away_team",
    "fthg",
    "ftag",
    "full_time_res",
    *ODDS_TARGETS,
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """读取单个原始 CSV，兼容 BOM 与各赛季列差异。"""
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def normalize_frame(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """单文件清洗：选列、丢弃未完赛、解析日期、类型规范化。"""
    missing_core = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing_core:
        raise ValueError(f"{league}_{season} 缺少核心列: {missing_core}")

    present_odds = [c for c in ODDS_COLUMNS if c in df.columns]
    df = df[list(CORE_COLUMNS) + present_odds].rename(
        columns={**CORE_COLUMNS, **ODDS_COLUMNS}
    )

    # 缺失的赔率列补齐为 NaN，保证输出 schema 稳定
    for col in ODDS_TARGETS:
        if col not in df.columns:
            df[col] = pd.NA

    # 队名去首尾空白
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    # 只保留已完赛：有进球数且赛果为 H/D/A（过滤延期/空行，2526 赛季存在未赛场次）
    df = df[df["full_time_res"].isin(["H", "D", "A"])]
    df = df.dropna(subset=["fthg", "ftag", "home_team", "away_team"])
    df["fthg"] = df["fthg"].astype(int)
    df["ftag"] = df["ftag"].astype(int)

    # 日期：dd/mm/yy 与 dd/mm/yyyy 混用；dayfirst 统一解析
    df["match_date"] = pd.to_datetime(df["match_date"], dayfirst=True, format="mixed")

    # 赔率转浮点，非法值转 NaN
    for col in ODDS_TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["league"] = league
    df["season"] = season
    df = df[OUTPUT_COLUMNS].drop_duplicates(
        subset=["league", "season", "match_date", "home_team", "away_team"]
    )
    return df.reset_index(drop=True)


def main() -> None:
    frames: list[pd.DataFrame] = []
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit("data/raw 下没有 CSV，请先运行 fetch_csv.py")

    for path in files:
        league, season = path.stem.split("_")
        frames.append(normalize_frame(load_raw(path), league, season))

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values(
        ["league", "season", "match_date"]
    ).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(PROCESSED_DIR / "matches.csv", index=False)

    # 球队主数据：(联赛, 队名) 去重
    teams = (
        pd.concat(
            [
                matches[["league", "home_team"]].rename(columns={"home_team": "team"}),
                matches[["league", "away_team"]].rename(columns={"away_team": "team"}),
            ]
        )
        .drop_duplicates()
        .sort_values(["league", "team"])
        .reset_index(drop=True)
    )
    teams.to_csv(MASTER_DIR / "teams.csv", index=False)

    print(f"已完赛比赛：{len(matches):,} 场")
    print(f"球队数：{len(teams)}（五大联赛合计）")
    print(f"日期范围：{matches['match_date'].min().date()} ~ {matches['match_date'].max().date()}")
    print(f"输出：{PROCESSED_DIR.relative_to(REPO_ROOT)}/matches.csv")


if __name__ == "__main__":
    main()
