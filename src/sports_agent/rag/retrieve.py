"""RAG 检索：HNSW cosine top-k + 元数据过滤 + 强制溯源。

search_knowledge 返回 RetrievedChunk 列表，每条含 doc_source（溯源锚点）。
元数据过滤走 SQL WHERE（不用向量后过滤），保证 top-k 内全符合过滤条件。
"""

from dataclasses import dataclass

from sqlalchemy import create_engine, text

from sports_agent.rag.embedding import EmbeddingClient, get_embedding_client
from sports_agent.settings import settings


@dataclass(frozen=True)
class RetrievedChunk:
    """检索结果：文本块 + 来源元数据 + 相似度分数。"""

    content: str
    doc_source: str
    doc_type: str
    league: str | None
    team: str | None
    doc_date: str | None
    score: float


def search_knowledge(
    query: str,
    *,
    k: int = 5,
    league: str | None = None,
    team: str | None = None,
    doc_type: str | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[RetrievedChunk]:
    """HNSW cosine top-k + 元数据过滤；强制返回 doc_source。

    无结果返回空列表，不编造。
    """
    client = embedding_client or get_embedding_client()
    query_vec = client.embed([query])[0]

    engine = create_engine(settings.database_url)
    vec_str = "[" + ",".join(f"{x:.8f}" for x in query_vec) + "]"

    # SQL WHERE 元数据过滤 + <=> cosine 距离排序（HNSW 自动启用）
    sql = text("""
        SELECT content, doc_source, doc_type, league, team, doc_date,
               1 - (embedding <=> :q_vec) AS score
        FROM v_chunks
        WHERE (:league IS NULL OR league = :league)
          AND (:team IS NULL OR team = :team)
          AND (:doc_type IS NULL OR doc_type = :doc_type)
        ORDER BY embedding <=> :q_vec
        LIMIT :k
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {
                "q_vec": vec_str,
                "league": league,
                "team": team,
                "doc_type": doc_type,
                "k": k,
            },
        ).all()

    return [
        RetrievedChunk(
            content=r.content,
            doc_source=r.doc_source,
            doc_type=r.doc_type,
            league=r.league,
            team=r.team,
            doc_date=str(r.doc_date) if r.doc_date else None,
            score=float(r.score),
        )
        for r in rows
    ]
