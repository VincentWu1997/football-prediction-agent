"""MCP Server B：ML 预测能力（W3 模型训练后接入）。

运行：
    fastmcp run mcp_servers/prediction_server.py --transport http --port 8002
"""

from fastmcp import FastMCP

mcp = FastMCP("prediction-server")


@mcp.tool
def server_status() -> dict:
    """服务健康与工具清单。"""
    return {
        "server": "prediction-server",
        "tools_planned": ["predict_match", "simulate_season"],
        "models_planned": ["elo", "dixon_coles", "xgboost"],
        "status": "skeleton",
    }


@mcp.tool
def predict_match(home: str, away: str, model_version: str = "latest") -> dict:
    """预测单场胜平负概率（W3：ELO/Dixon-Coles/XGBoost 集成结果）。"""
    raise NotImplementedError(
        "W3：返回三套模型概率 + 校准信息；禁止在模型就绪前返回任何虚构概率"
    )


@mcp.tool
def simulate_season(league: str, season: str, n_runs: int = 10_000) -> dict:
    """L4 剩余赛程蒙特卡洛模拟（W4）。"""
    raise NotImplementedError("W4：按 Dixon-Coles λ 逐轮抽样，输出前四/夺冠/降级概率")


if __name__ == "__main__":
    mcp.run(transport="http", port=8002)
