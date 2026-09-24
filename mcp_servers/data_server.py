"""MCP Server A：结构化足球数据查询（W6 已接入 PostgreSQL f_matches + f_odds）。

运行：
    fastmcp run mcp_servers/data_server.py --transport http --port 8001

实现说明：能力逻辑在 sports_agent.data.queries，MCP 只是薄封装（与 prediction_server 同构）；
队名规范化复用 predict_service.resolve_team，与预测口径一致。
"""

from fastmcp import FastMCP

from sports_agent.data.queries import (
    query_h2h,
    query_odds,
    query_recent_form,
    query_standings,
)

mcp = FastMCP("data-server")


@mcp.tool
def server_status() -> dict:
    """服务健康与工具清单。"""
    return {
        "server": "data-server",
        "tools": ["query_standings", "query_recent_form", "query_h2h", "query_odds"],
        "store": "PostgreSQL f_matches + f_odds",
        "status": "ready",
    }


@mcp.tool
def query_standings_tool(league: str, season: str | None = None) -> dict:
    """某联赛某赛季积分榜。

    league 为 football-data.co.uk 代码：E0(英超) / SP1(西甲) / D1(德甲) / I1(意甲) / F1(法甲)。
    season 为 4 位起始年份+2 位终止年份，如 2425 表示 2024-25 赛季。
    season 省略时自动取库中最新赛季（推荐，避免过时数据）。
    返回 {rank, team, played, won, drawn, lost, gf, ga, gd, points} 列表。
    """
    from sports_agent.data.queries import latest_season
    resolved = season or latest_season(league)
    rows = query_standings(league, resolved)
    return {"league": league, "season": resolved,
            "n_teams": len(rows), "standings": rows}


@mcp.tool
def query_recent_form_tool(team: str, n: int = 5) -> dict:
    """某队最近 N 场比赛战绩（含主客），按比赛日期倒序。

    team 为 football-data.co.uk 官方队名（如 Manchester United / Liverpool），
    支持常见别名与近似匹配；n 默认 5，最大 30。
    """
    n = max(1, min(int(n), 30))
    rows = query_recent_form(team, n=n)
    return {"team": team, "n_requested": n, "n_returned": len(rows), "matches": rows}


@mcp.tool
def query_h2h_tool(team_a: str, team_b: str, n: int = 10) -> dict:
    """两队近 N 次交手记录（任一主客组合），按比赛日期倒序。

    team_a / team_b 为官方队名；返回最近一次到最早的交手列表，
    每条含 date/season/home/away/fthg/ftag/result。
    """
    n = max(1, min(int(n), 50))
    rows = query_h2h(team_a, team_b, n=n)
    return {
        "team_a": team_a,
        "team_b": team_b,
        "n_requested": n,
        "n_returned": len(rows),
        "matches": rows,
    }


@mcp.tool
def query_odds_tool(
    home: str, away: str, league: str | None = None, season: str | None = None
) -> dict:
    """两队最近一次交锋的赔率（Bet365 + Pinnacle + 市场平均，临场 closing）。

    无记录时返回 {"found": false}，不返回伪赔率。
    league/season 可选过滤；不传时取两队最近一次交手。
    """
    return query_odds(home, away, league=league, season=season)


if __name__ == "__main__":
    mcp.run(transport="http", port=8001)
