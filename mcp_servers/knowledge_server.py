"""MCP Server C：非结构化知识 RAG 检索（W5 已接入 pgvector + bge-m3）。

运行：
    fastmcp run mcp_servers/knowledge_server.py --transport http --port 8003

能力逻辑在 sports_agent.rag.retrieve，MCP 只是薄封装（与 prediction_server 同构）。
"""

from fastmcp import FastMCP

from sports_agent.rag.retrieve import search_knowledge as _search

mcp = FastMCP("knowledge-server")


@mcp.tool
def server_status() -> dict:
    """服务健康与工具清单。"""
    return {
        "server": "knowledge-server",
        "tools": ["search_knowledge"],
        "embedding": "bge-m3 via Ollama /v1/embeddings",
        "store": "pgvector HNSW cosine",
        "status": "ready",
    }


@mcp.tool
def search_knowledge(
    query: str, k: int = 5, league: str | None = None, team: str | None = None
) -> list[dict]:
    """检索伤停新闻/战术报告/赛前采访，返回文本块与来源。

    每条结果含 doc_source（强制溯源）；无结果返回空列表，不编造。
    league/team 可选过滤（如 E0 / Liverpool）。
    """
    chunks = _search(query, k=k, league=league, team=team)
    return [
        {
            "content": c.content,
            "doc_source": c.doc_source,
            "doc_type": c.doc_type,
            "league": c.league,
            "team": c.team,
            "doc_date": c.doc_date,
            "score": round(c.score, 4),
        }
        for c in chunks
    ]


if __name__ == "__main__":
    mcp.run(transport="http", port=8003)
