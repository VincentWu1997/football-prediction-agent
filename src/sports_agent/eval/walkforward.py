"""Walk-forward 折叠：每个测试赛季之前的全部赛季作为训练集。

禁止随机切分——特征（ELO 评分、DC 参数、赔率）的计算时点必须早于比赛开球。
赛季代码为 YYyy 字符串，字典序即时间序（"1516" < ... < "2526"）。
"""

from collections.abc import Iterator

import pandas as pd

# 至少留 4 个完整赛季（1516-1819）作为首个训练窗
TEST_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]


def walk_forward_folds(
    df: pd.DataFrame, test_seasons: list[str] | None = None
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """产出 (test_season, train_df, test_df)；按赛季扩展窗口。"""
    seasons_test = test_seasons or TEST_SEASONS
    all_seasons = sorted(df["season"].astype(str).unique())

    for test_season in seasons_test:
        train_seasons = [s for s in all_seasons if s < test_season]
        if len(train_seasons) < 4:
            continue
        train = df[df["season"].astype(str).isin(train_seasons)]
        test = df[df["season"].astype(str) == test_season]
        if not test.empty:
            yield test_season, train, test
