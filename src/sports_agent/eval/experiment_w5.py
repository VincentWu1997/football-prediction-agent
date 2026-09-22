"""W5 实验：RAG 检索评测（recall@5/MRR）+ 无 RAG vs 有 RAG 答案对照。

运行：python -m sports_agent.eval.experiment_w5
产物（benchmarks/results/w5/）：
- retrieval_metrics.csv   recall@5 / MRR
- qa_comparison.csv       无 RAG vs 有 RAG 逐条对照（Ollama 离线时跳过）
- qa_summary.csv          汇总

注意：retrieval 评测用真实 bge-m3（需 ollama pull bge-m3）；
QA 对照依赖 qwen3:4b；任一缺失时跳过对应子实验并警告，不阻塞。
"""

import sys
import time

import pandas as pd

from sports_agent.rag.embedding import (
    MockEmbeddingClient,
    OllamaEmbeddingClient,
    get_embedding_client,
)
from sports_agent.rag.eval import build_eval_pairs, mrr, recall_at_k
from sports_agent.rag.ingest import count_chunks, ingest_documents, load_corpus_jsonl
from sports_agent.rag.retrieve import search_knowledge
from sports_agent.settings import REPO_ROOT

CORPUS_PATH = REPO_ROOT / "data" / "rag_corpus" / "docs.jsonl"
OUT_DIR = REPO_ROOT / "benchmarks" / "results" / "w5"


def _is_ollama_ready() -> bool:
    """检查 bge-m3 是否在 Ollama 可用。"""
    client = OllamaEmbeddingClient()
    st = client.status()
    return bool(st.get("reachable") and st.get("model_available"))


def _retrieve_doc_ids(query: str, client, k: int = 5) -> list[str]:
    """对 query 做 top-k 检索，返回 doc_source 列表（评测对用 doc_id）。"""
    from sports_agent.rag.retrieve import search_knowledge

    chunks = search_knowledge(query, k=k, embedding_client=client)
    # doc_source 是 URL，评测对用 doc_id（=doc_source 的末段 .html 前的 slug）
    # 这里用 doc_source 做 ID（唯一且与语料中 doc_id 一一对应）
    return [c.doc_source for c in chunks]


def _build_retrieval_eval(client, pairs: list[dict]) -> pd.DataFrame:
    """对每条评测对跑检索，算 recall@5 / MRR。"""
    retrieved_all: list[list[str]] = []
    for p in pairs:
        retrieved = _retrieve_doc_ids(p["query"], client, k=5)
        retrieved_all.append(retrieved)

    relevant_all = [p["relevant_doc_ids"] for p in pairs]
    # 语料中 doc_id 对应 doc_source，需转换
    # 但在评测中，relevant_doc_ids 存的是 doc_id；检索返回 doc_source
    # 为统一，先加载语料建 doc_id -> doc_source 映射
    docs = load_corpus_jsonl(CORPUS_PATH)
    id_to_source = {d.doc_id: d.doc_source for d in docs}
    relevant_sources = [
        [id_to_source.get(rid, rid) for rid in rids] for rids in relevant_all
    ]

    r5 = recall_at_k(retrieved_all, relevant_sources, k=5)
    mrr_v = mrr(retrieved_all, relevant_sources, k=5)
    return pd.DataFrame(
        {
            "metric": ["recall@5", "MRR"],
            "value": [r5, mrr_v],
            "n_queries": [len(pairs)] * 2,
            "k": [5] * 2,
        }
    )


def _qa_comparison(pairs: list[dict], n: int = 20) -> pd.DataFrame | None:
    """无 RAG vs 有 RAG 答案对照（依赖 qwen3:4b）。"""
    try:
        from sports_agent.inference.client import InferenceClient

        llm = InferenceClient()
        system = (
            "你是足球赛事分析助手。基于提供的信息作答；"
            "如果信息不足，回答'信息不足'。"
        )
    except Exception as e:
        print(f"  [WARN] LLM 不可用，跳过 QA 对照: {e}")
        return None

    sample = pairs[:n]
    rows: list[dict] = []
    for p in sample:
        query = p["query"]
        team = p.get("team", "")
        relevant = p["relevant_doc_ids"]
        # 无 RAG
        try:
            r_plain = llm.chat("fast", [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ])
            plain_answer = r_plain.content.strip()
        except Exception:
            plain_answer = "[LLM 调用失败]"

        # 有 RAG
        try:
            chunks = search_knowledge(query, k=5)
            context = "\n".join(
                f"[{i + 1}] {c.content}\n来源: {c.doc_source}"
                for i, c in enumerate(chunks)
            )
            r_rag = llm.chat("fast", [
                {"role": "system", "content": f"{system}\n\n参考资料:\n{context}"},
                {"role": "user", "content": query},
            ])
            rag_answer = r_rag.content.strip()
        except Exception:
            rag_answer = "[检索/LLM 调用失败]"

        # 硬匹配：答案中是否包含相关 doc 的 team 或 doc_id 关键词
        hit_key = team if team else (relevant[0] if relevant else "")
        plain_hit = 1 if hit_key and hit_key in plain_answer else 0
        rag_hit = 1 if hit_key and hit_key in rag_answer else 0

        rows.append({
            "query": query,
            "mode": "plain",
            "answer": plain_answer[:200],
            "hit_key": hit_key,
            "score": plain_hit,
        })
        rows.append({
            "query": query,
            "mode": "rag",
            "answer": rag_answer[:200],
            "hit_key": hit_key,
            "score": rag_hit,
        })

    return pd.DataFrame(rows)


def run() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 入库（幂等：已有数据时跳过）
    n_chunks = count_chunks()
    if n_chunks == 0:
        print("v_chunks 为空，开始摄入语料...")
        docs = load_corpus_jsonl(CORPUS_PATH)
        print(f"  语料: {len(docs)} 篇")
        client = get_embedding_client()
        result = ingest_documents(docs, client, batch_size=32)
        print(f"  入库: {result}")
        n_chunks = count_chunks()
    print(f"v_chunks 当前 {n_chunks} 条")

    # 2) 生成评测对
    docs = load_corpus_jsonl(CORPUS_PATH)
    pairs = build_eval_pairs(docs, seed=42, n=80)
    print(f"评测集: {len(pairs)} 条")

    # 3) Retrieval 评测
    use_mock = not _is_ollama_ready()
    if use_mock:
        print("[WARN] bge-m3 未就绪，用 Mock embedding 跑 retrieval 评测（无语义意义）")
        client = MockEmbeddingClient()
    else:
        print("bge-m3 就绪，用真实 embedding 跑 retrieval 评测")
        client = OllamaEmbeddingClient()

    metrics = _build_retrieval_eval(client, pairs)
    metrics_path = OUT_DIR / "retrieval_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print("\n=== 检索评测 ===")
    print(metrics.to_string(index=False))

    # 4) QA 对照
    qa_df = _qa_comparison(pairs, n=20)
    if qa_df is not None:
        qa_path = OUT_DIR / "qa_comparison.csv"
        qa_df.to_csv(qa_path, index=False)
        summary = qa_df.groupby("mode")["score"].agg(["mean", "count"]).reset_index()
        summary.columns = ["mode", "accuracy", "n"]
        summary.to_csv(OUT_DIR / "qa_summary.csv", index=False)
        print("\n=== QA 对照 ===")
        print(summary.to_string(index=False))
    else:
        print("\n[SKIP] QA 对照已跳过（Ollama 不可用）")

    print(f"\n耗时 {time.perf_counter() - t0:.1f}s")
    print(f"产物: {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(run())
