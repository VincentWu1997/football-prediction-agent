"""MCP Server A：结构化足球数据查询（W6 实现真实 SQL）。

运行：
    fastmcp run mcp_servers/data_server.py --transport http --port 8001
"""

from fastmcp import FastMCP

mcp = FastMCP("data-server")


@mcp.tool
def server_status() -> dict:
    """服务健康与工具清单。"""
    return {
        "server": "data-server",
        "tools_planned": ["query_standings", "query_recent_form", "query_h2h", "query_odds"],
        "status": "skeleton",
    }


@mcp.tool
def query_standings(league: str, season: str) -> dict:
    """查询积分榜（W6 接入 PostgreSQL f_matches 表实时计算）。"""
    raise NotImplementedError("W6：基于 f_matches 聚合积分/净胜球，按联赛赛季返回榜单项")


if __name__ == "__main__":
    mcp.run(transport="http", port=8001)
