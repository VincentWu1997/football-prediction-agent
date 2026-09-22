"""RAG 摄入流水线：文档切片 + bge-m3 向量化 + 幂等入库。

- chunk_document：短文(≤max_chars)整篇返回，长文按句号切分带重叠；
- load_corpus_jsonl：读 data/rag_corpus/docs.jsonl -> DocRecord 列表；
- ingest_documents：批量 embedding + on_conflict_do_nothing 幂等入库。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import MetaData, Table, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from sports_agent.rag.embedding import EmbeddingClient
from sports_agent.settings import settings

# v_chunks 唯一约束名（db/init/02_rag_unique.sql）
_CHUNK_UNIQUE = "v_chunks_doc_source_content_key"

# 中文句子分隔符：。！？；（分号也常用于并列句）
_SENTENCE_END = re.compile(r"([。！？；])")


@dataclass(frozen=True)
class DocRecord:
    """单篇文档的元数据与全文。"""

    doc_id: str
    content: str
    doc_source: str
    doc_type: str
    league: str | None = None
    team: str | None = None
    doc_date: str | None = None


def load_corpus_jsonl(path: Path) -> list[DocRecord]:
    """读 data/rag_corpus/docs.jsonl -> DocRecord 列表。"""
    docs: list[DocRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                DocRecord(
                    doc_id=obj["doc_id"],
                    content=obj["content"],
                    doc_source=obj["doc_source"],
                    doc_type=obj["doc_type"],
                    league=obj.get("league"),
                    team=obj.get("team"),
                    doc_date=obj.get("doc_date"),
                )
            )
    return docs


def chunk_document(
    doc: DocRecord, *, max_chars: int = 500, overlap_chars: int = 50
) -> list[DocRecord]:
    """短文整篇返回；长文按句子切分带重叠；元数据原样继承。

    chunk 的 doc_source 与原 doc 相同（溯源锚点不变），content 为子串。
    """
    if len(doc.content) <= max_chars:
        return [doc]

    # 按句号切分，保留标点在句尾
    parts = _SENTENCE_END.split(doc.content)
    # split 会在分隔符前后产生空串和标点片段，需拼回
    sentences: list[str] = []
    for i in range(0, len(parts) - 1, 2):
        sent = parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i]
        if sent:
            sentences.append(sent)

    chunks: list[DocRecord] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(_replace_content(doc, current))
            # 重叠：保留末尾 overlap_chars 字符
            current = current[-overlap_chars:] + sent if overlap_chars > 0 else sent
        else:
            current += sent
    if current:
        chunks.append(_replace_content(doc, current))

    return chunks if chunks else [doc]


def _replace_content(doc: DocRecord, content: str) -> DocRecord:
    """复制 doc 的元数据，替换 content（其他字段 frozen 不可变）。"""
    return DocRecord(
        doc_id=doc.doc_id,
        content=content,
        doc_source=doc.doc_source,
        doc_type=doc.doc_type,
        league=doc.league,
        team=doc.team,
        doc_date=doc.doc_date,
    )


def ingest_documents(
    docs: list[DocRecord],
    embedding_client: EmbeddingClient,
    *,
    batch_size: int = 32,
) -> dict:
    """批量 embedding + 幂等入库 v_chunks。

    返回 {"n_input": ..., "n_chunks": ..., "n_inserted": ...}。
    幂等：on_conflict_do_nothing on UNIQUE(doc_source, content)。
    """
    # 1) 切片
    all_chunks: list[DocRecord] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    # 2) 批量 embedding + 入库
    engine = create_engine(settings.database_url)
    metadata = MetaData(schema="public")
    v_chunks = Table("v_chunks", metadata, autoload_with=engine)

    total_inserted = 0
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c.content for c in batch]
        vectors = embedding_client.embed(texts)

        records = [
            {
                "content": c.content,
                "doc_source": c.doc_source,
                "doc_type": c.doc_type,
                "league": c.league,
                "team": c.team,
                "doc_date": c.doc_date,
                "embedding": str(vec),  # pgvector 接受 '[0.1,0.2,...]' 文本
            }
            for c, vec in zip(batch, vectors, strict=True)
        ]
        stmt = pg_insert(v_chunks).on_conflict_do_nothing(constraint=_CHUNK_UNIQUE)
        with engine.begin() as conn:
            result = conn.execute(stmt, records)
            total_inserted += result.rowcount

    return {
        "n_input": len(docs),
        "n_chunks": len(all_chunks),
        "n_inserted": total_inserted,
    }


def count_chunks() -> int:
    """SELECT count(*) FROM v_chunks。"""
    engine = create_engine(settings.database_url)
    metadata = MetaData(schema="public")
    v_chunks = Table("v_chunks", metadata, autoload_with=engine)
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(v_chunks)).scalar() or 0
