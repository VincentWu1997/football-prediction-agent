"""MCP Server B：ML 预测能力（W4 已接入 Dixon-Coles + 蒙特卡洛）。

运行：
    fastmcp run mcp_servers/prediction_server.py --transport http --port 8002

实现说明：能力逻辑在 sports_agent.ml.predict_service，MCP 只是薄封装；
W3 的 XGBoost 走离线 walk-forward 实验（eval.experiment_w3），不进在线服务。
"""

from fastmcp import FastMCP

from sports_agent.ml.predict_service import predict_match, simulate_season

mcp = FastMCP("prediction-server")


@mcp.tool
def server_status() -> dict:
    """服务健康与工具清单。"""
    return {
        "server": "prediction-server",
        "tools": ["predict_match", "simulate_season"],
        "models": ["dixon_coles + monte_carlo"],
        "status": "ready",
    }


@mcp.tool
def predict_match_tool(
    home: str, away: str, league: str | None = None, n_sims: int = 20_000
) -> dict:
    """预测单场：Dixon-Coles 胜平负概率 + 蒙特卡洛比分分布/置信区间。

    home/away 为官方队名（与 football-data.co.uk 一致，如 Manchester United）。
    league 可省略（自动按两队共同联赛推断，如 E0/SP1/D1/I1/F1）。
    """
    return predict_match(home, away, league=league, n_sims=n_sims)


@mcp.tool
def simulate_season_tool(
    league: str, as_of: str, n_runs: int = 20_000
) -> dict:
    """L4 赛季模拟：对 as_of 之后的剩余赛程做蒙特卡洛。

    返回各队夺冠/前四/降级概率、期望积分，以及真实最终排名（回看校验用）。
    as_of 格式 YYYY-MM-DD，需落在某赛季进行中。
    """
    return simulate_season(league, as_of, n_runs=n_runs)


if __name__ == "__main__":
    mcp.run(transport="http", port=8002)
