"""MCP Server C：非结构化知识 RAG 检索（W5 接入 pgvector）。

运行：
    fastmcp run mcp_servers/knowledge_server.py --transport http --port 8003
"""

from fastmcp import FastMCP

mcp = FastMCP("knowledge-server")


@mcp.tool
def server_status() -> dict:
    """服务健康与工具清单。"""
    return {
        "server": "knowledge-server",
        "tools_planned": ["search_knowledge"],
        "embedding": "bge-m3 via Ollama /v1/embeddings",
        "store": "pgvector HNSW cosine",
        "status": "skeleton",
    }


@mcp.tool
def search_knowledge(query: str, k: int = 5) -> list[dict]:
    """检索伤停新闻/战术报告/赛前采访，返回文本块与来源（W5）。"""
    raise NotImplementedError("W5：bge-m3 向量化 + v_chunks 检索，结果必须含 doc_source")


if __name__ == "__main__":
    mcp.run(transport="http", port=8003)
