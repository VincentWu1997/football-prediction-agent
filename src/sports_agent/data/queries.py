"""结构化足球数据查询：直接走 PostgreSQL f_matches + f_odds。

设计原则（方案 §3.2）：
- 积分榜/战绩/赔率这类结构化数据一律走 SQL，不进向量库；
- 队名规范化复用 predict_service.resolve_team，保证与预测口径一致；
- 未命中返回空列表 / None，绝不编造数字。
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sports_agent.data.db import get_engine
from sports_agent.ml.predict_service import resolve_team


def _engine() -> Engine:
    return get_engine()


def latest_season(league: str) -> str | None:
    """库中该联赛最新赛季代码（如 2526）。LLM 不知道"现在"是何时，
    不传 season 时自动取最新，避免默认用到旧赛季数据。"""
    sql = text("SELECT MAX(season) FROM f_matches WHERE league = :league")
    with _engine().connect() as conn:
        return conn.execute(sql, {"league": league}).scalar()


def query_standings(league: str, season: str | None = None) -> list[dict]:
    """某联赛某赛季积分榜：按 积分/GD/GF 排序。

    league 为 football-data.co.uk 代码（E0/SP1/D1/I1/F1）；
    season 为 4 位起始年份+末两位终止年份，如 2425；
    season 省略时自动取库中最新赛季。
    """
    if not season:
        season = latest_season(league)
        if season is None:
            return []
    sql = text(
        """
        WITH sides AS (
            SELECT home_team AS team, fthg AS gf, ftag AS ga, full_time_res AS res, 'H' AS venue
            FROM f_matches
            WHERE league = :league AND season = :season AND full_time_res IS NOT NULL
            UNION ALL
            SELECT away_team AS team, ftag AS gf, fthg AS ga, full_time_res AS res, 'A' AS venue
            FROM f_matches
            WHERE league = :league AND season = :season AND full_time_res IS NOT NULL
        )
        SELECT team,
               COUNT(*) AS played,
               SUM(CASE WHEN (res='H' AND venue='H')
                    OR (res='A' AND venue='A') THEN 1 ELSE 0 END) AS won,
               SUM(CASE WHEN res='D' THEN 1 ELSE 0 END) AS drawn,
               SUM(CASE WHEN (res='A' AND venue='H')
                    OR (res='H' AND venue='A') THEN 1 ELSE 0 END) AS lost,
               SUM(gf) AS gf,
               SUM(ga) AS ga,
               SUM(gf - ga) AS gd,
               SUM(CASE WHEN (res='H' AND venue='H') OR (res='A' AND venue='A') THEN 3
                        WHEN res='D' THEN 1 ELSE 0 END) AS points
        FROM sides
        GROUP BY team
        ORDER BY points DESC, gd DESC, gf DESC, team
        """
    )
    with _engine().connect() as conn:
        rows = conn.execute(sql, {"league": league, "season": season}).all()
    if not rows:
        return []
    return [
        {
            "rank": i + 1,
            "team": r.team,
            "played": int(r.played),
            "won": int(r.won),
            "drawn": int(r.drawn),
            "lost": int(r.lost),
            "gf": int(r.gf),
            "ga": int(r.ga),
            "gd": int(r.gd),
            "points": int(r.points),
        }
        for i, r in enumerate(rows)
    ]


def query_recent_form(team: str, n: int = 5) -> list[dict]:
    """某队最近 N 场（含主客），按比赛日期倒序。"""
    canonical = resolve_team(team)
    sql = text(
        """
        SELECT match_date, home_team, away_team, fthg, ftag, full_time_res
        FROM f_matches
        WHERE (home_team = :team OR away_team = :team)
          AND full_time_res IS NOT NULL
        ORDER BY match_date DESC
        LIMIT :n
        """
    )
    with _engine().connect() as conn:
        rows = conn.execute(sql, {"team": canonical, "n": n}).all()
    out: list[dict] = []
    for r in rows:
        is_home = r.home_team == canonical
        gf = r.fthg if is_home else r.ftag
        ga = r.ftag if is_home else r.fthg
        res_letter = "H" if gf > ga else ("D" if gf == ga else "A")
        out.append(
            {
                "date": str(r.match_date),
                "venue": "H" if is_home else "A",
                "opponent": r.away_team if is_home else r.home_team,
                "gf": int(gf),
                "ga": int(ga),
                "result": res_letter,
            }
        )
    return out


def query_h2h(team_a: str, team_b: str, n: int = 10) -> list[dict]:
    """两队近 N 次交手（任一主客组合），按比赛日期倒序。"""
    a = resolve_team(team_a)
    b = resolve_team(team_b)
    sql = text(
        """
        SELECT match_date, home_team, away_team, fthg, ftag, full_time_res, season
        FROM f_matches
        WHERE ((home_team = :a AND away_team = :b)
               OR (home_team = :b AND away_team = :a))
          AND full_time_res IS NOT NULL
        ORDER BY match_date DESC
        LIMIT :n
        """
    )
    with _engine().connect() as conn:
        rows = conn.execute(sql, {"a": a, "b": b, "n": n}).all()
    return [
        {
            "date": str(r.match_date),
            "season": r.season,
            "home": r.home_team,
            "away": r.away_team,
            "fthg": int(r.fthg),
            "ftag": int(r.ftag),
            "result": r.full_time_res,
        }
        for r in rows
    ]


def query_odds(home: str, away: str, league: str | None = None, season: str | None = None) -> dict:
    """两队最近一次交锋的赔率（Bet365 + Pinnacle + 市场平均，临场 closing）。

    无任何记录时返回 {"found": False}，不返回伪赔率。
    """
    a = resolve_team(home)
    b = resolve_team(away)
    sql = text(
        """
        SELECT m.match_date, m.season, m.league,
               m.home_team AS home_team, m.away_team AS away_team,
               o.b365h, o.b365d, o.b365a,
               o.psh, o.psd, o.psa,
               o.avgh, o.avgd, o.avga
        FROM f_matches m
        LEFT JOIN f_odds o ON o.match_id = m.match_id
        WHERE ((m.home_team = :a AND m.away_team = :b)
               OR (m.home_team = :b AND m.away_team = :a))
          AND (:league IS NULL OR m.league = :league)
          AND (:season IS NULL OR m.season = :season)
        ORDER BY m.match_date DESC
        LIMIT 1
        """
    )
    with _engine().connect() as conn:
        row = conn.execute(sql, {"a": a, "b": b, "league": league, "season": season}).first()
    if row is None:
        return {"found": False, "home": a, "away": b}

    def num(x):
        return float(x) if x is not None else None

    return {
        "found": True,
        "date": str(row.match_date),
        "season": row.season,
        "league": row.league,
        "home": row.home_team if row.home_team == a else b,
        "away": row.away_team if row.away_team == b else a,
        "bet365": {"h": num(row.b365h), "d": num(row.b365d), "a": num(row.b365a)},
        "pinnacle": {"h": num(row.psh), "d": num(row.psd), "a": num(row.psa)},
        "market_avg": {"h": num(row.avgh), "d": num(row.avgd), "a": num(row.avga)},
    }
