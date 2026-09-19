"""normalize 清洗逻辑测试：未完赛过滤、日期解析、缺失赔率列兼容。"""

import pandas as pd
from normalize import normalize_frame


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["16/08/24", "17/08/24", "18/08/24"],
            "HomeTeam": ["Arsenal", "Chelsea", "Liverpool"],
            "AwayTeam": ["Wolves", "Palace", "Spurs"],
            "FTHG": [2, 1, None],
            "FTAG": [0, 1, None],
            "FTR": ["H", "D", None],  # 第三行未开赛
            "B365H": [1.8, 2.1, 2.0],
            "B365D": [3.5, 3.2, 3.3],
            "B365A": [4.2, 3.4, 3.5],
            # 无 PSH/AvgH，模拟 2019 前赛季
        }
    )


def test_drops_unplayed_and_parses_date() -> None:
    out = normalize_frame(_raw(), "E0", "2425")
    assert len(out) == 2  # 未开赛行被剔除
    assert list(out["full_time_res"]) == ["H", "D"]
    assert out["match_date"].iloc[0] == pd.Timestamp("2024-08-16")


def test_missing_odds_columns_become_nan() -> None:
    out = normalize_frame(_raw(), "E0", "2425")
    assert {"psh", "avgh"}.issubset(out.columns)
    assert out["psh"].isna().all()
    # 存在的 B365 正常保留为浮点
    assert out["b365h"].iloc[0] == 1.8


def test_team_whitespace_stripped_and_goals_int() -> None:
    df = _raw()
    df.loc[0, "HomeTeam"] = "  Arsenal "
    out = normalize_frame(df, "E0", "2425")
    assert out["home_team"].iloc[0] == "Arsenal"
    assert out["fthg"].dtype == "int64"
