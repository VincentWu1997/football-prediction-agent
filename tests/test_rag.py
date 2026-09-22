"""RAG 模块测试（不依赖 Ollama，用 MockEmbeddingClient）。"""

from sports_agent.rag.embedding import MockEmbeddingClient
from sports_agent.rag.eval import build_eval_pairs, mrr, recall_at_k
from sports_agent.rag.ingest import DocRecord, chunk_document


def test_mock_embedding_deterministic() -> None:
    c = MockEmbeddingClient(dim=1024)
    v1 = c.embed(["曼联伤停"])
    v2 = c.embed(["曼联伤停"])
    assert v1 == v2, "同一文本嵌入结果应一致"


def test_mock_embedding_different_texts() -> None:
    c = MockEmbeddingClient(dim=128)
    v1 = c.embed(["利物浦"])
    v2 = c.embed(["曼城"])
    assert v1 != v2, "不同文本嵌入结果应不同"


def test_mock_embedding_dimension_and_normalized() -> None:
    c = MockEmbeddingClient(dim=1024)
    v = c.embed(["test"])[0]
    assert len(v) == 1024
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6, f"L2 范数应为 1.0，实际 {norm}"


def test_mock_embedding_empty() -> None:
    assert MockEmbeddingClient().embed([]) == []


def test_chunk_short_doc_single_chunk() -> None:
    doc = DocRecord(
        doc_id="x",
        content="短文。" * 50,
        doc_source="s",
        doc_type="news",
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].content == doc.content


def test_chunk_long_doc_multiple_chunks_with_overlap() -> None:
    doc = DocRecord(
        doc_id="x",
        content="第一句话。" * 300,
        doc_source="s",
        doc_type="news",
    )
    chunks = chunk_document(doc, max_chars=100, overlap_chars=20)
    assert len(chunks) > 1
    # 相邻 chunk 应有重叠（第二个 chunk开头包含第一个 chunk 末尾片段）
    assert chunks[1].content[:10] in chunks[0].content


def test_chunk_preserves_metadata() -> None:
    doc = DocRecord(
        doc_id="x",
        content="内容。",
        doc_source="src",
        doc_type="tactics",
        league="E0",
        team="Liverpool",
        doc_date="2025-01-15",
    )
    chunks = chunk_document(doc)
    for c in chunks:
        assert c.doc_id == doc.doc_id
        assert c.doc_source == doc.doc_source
        assert c.doc_type == doc.doc_type
        assert c.league == doc.league
        assert c.team == doc.team


def test_recall_at_k_math() -> None:
    # 2 条 query；第 1 条 top-5 命中 d2，第 2 条未命中
    retrieved = [["d1", "d2", "d3"], ["d4", "d5", "d6"]]
    relevant = [["d2"], ["d9"]]
    assert recall_at_k(retrieved, relevant, k=5) == 0.5


def test_recall_at_k_empty() -> None:
    assert recall_at_k([], [], k=5) == 0.0


def test_mrr_math() -> None:
    retrieved = [["d1", "d2", "d3"], ["d4", "d5", "d6"]]
    relevant = [["d2"], ["d9"]]
    # 第 1 条命中 rank=2 -> 1/2；第 2 条未命中 -> 0；MRR = 0.25
    assert abs(mrr(retrieved, relevant, k=5) - 0.25) < 1e-9


def test_mrr_first_position() -> None:
    retrieved = [["d1", "d2"], ["d3", "d4"]]
    relevant = [["d1"], ["d3"]]
    # 两条都在 rank=1 命中 -> MRR = 1.0
    assert abs(mrr(retrieved, relevant, k=5) - 1.0) < 1e-9


def test_build_eval_pairs_from_corpus() -> None:
    docs = [
        DocRecord(
            doc_id=f"doc-{i}",
            content=f"内容{i}。",
            doc_source=f"src-{i}",
            doc_type="injury" if i % 2 == 0 else "tactics",
            league="E0",
            team="Liverpool",
        )
        for i in range(100)
    ]
    pairs = build_eval_pairs(docs, seed=42, n=80)
    assert len(pairs) == 80
    for p in pairs:
        assert "query" in p
        assert len(p["relevant_doc_ids"]) == 1
        assert p["relevant_doc_ids"][0] in [d.doc_id for d in docs]
        assert p["doc_type"] in ("injury", "tactics")
