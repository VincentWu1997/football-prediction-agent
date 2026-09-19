#!/usr/bin/env python3
"""W1 数据管道第一步：从 football-data.co.uk 下载赛季 CSV 到 data/raw/。

数据说明见 https://www.football-data.co.uk/data.php ：
CSV 含赛果与多家博彩公司赔率（含临场 closing odds），免费无需 key。

用法：
    python data/scripts/fetch_csv.py --leagues E0 SP1 D1 I1 F1 --seasons 2324 2425
    python data/scripts/fetch_csv.py --leagues E0 --seasons 2425 --force
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# 常用联赛代码，完整清单见 https://www.football-data.co.uk/data.php
LEAGUES = {
    "E0": "英超",
    "E1": "英冠",
    "SP1": "西甲",
    "D1": "德甲",
    "I1": "意甲",
    "F1": "法甲",
}


def download(league: str, season: str, *, force: bool = False) -> str:
    """下载单个 联赛×赛季 CSV。返回状态：ok / skip / fail。"""
    if league not in LEAGUES:
        raise ValueError(f"未知联赛代码 {league}，可选：{', '.join(LEAGUES)}")

    out_path = RAW_DIR / f"{league}_{season}.csv"
    if out_path.exists() and not force:
        print(f"SKIP  {league} {season}（已存在，--force 可覆盖）")
        return "skip"

    url = f"{BASE_URL}/{season}/{league}.csv"
    print(f"GET   {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read()
    except urllib.error.HTTPError as e:
        print(f"FAIL  {league} {season}: HTTP {e.code}", file=sys.stderr)
        return "fail"

    # 该站 CSV 偶尔有少量非法字符，忽略即可，normalize 阶段再做清洗
    out_path.write_bytes(content)
    print(f"OK    {out_path.relative_to(REPO_ROOT)} ({len(content)} bytes)")
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 football-data.co.uk 赛季 CSV")
    parser.add_argument("--leagues", nargs="+", required=True, choices=sorted(LEAGUES))
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="赛季代码，如 2324 表示 2023/24 赛季",
    )
    parser.add_argument("--force", action="store_true", help="覆盖已存在文件")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    for season in args.seasons:
        for league in args.leagues:
            counts[download(league, season, force=args.force)] += 1

    total = len(args.leagues) * len(args.seasons)
    print(
        f"\n完成：{counts['ok']} 新下载 / {counts['skip']} 已存在跳过 / "
        f"{counts['fail']} 失败，共 {total}"
    )
    return 1 if counts["fail"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
