"""RAG 检索评测：recall@k / MRR / 评测对生成。

- recall_at_k：top-k 内命中≥1 个 relevant 记 1，否则 0，返回均值；
- mrr：第一个命中 rank r 得 1/r，未命中 0，返回均值；
- build_eval_pairs：从语料自动生成 query→relevant_doc_ids 对。
"""

import random

from sports_agent.rag.ingest import DocRecord


def recall_at_k(
    retrieved_ids: list[list[str]], relevant_ids: list[list[str]], k: int = 5
) -> float:
    """recall@k：每条 query 的 top-k 中至少命中一个 relevant 记 1。"""
    if not retrieved_ids:
        return 0.0
    hits = 0
    for retrieved, relevant in zip(retrieved_ids, relevant_ids, strict=True):
        rel_set = set(relevant)
        top_k = retrieved[:k]
        if any(rid in rel_set for rid in top_k):
            hits += 1
    return hits / len(retrieved_ids)


def mrr(
    retrieved_ids: list[list[str]], relevant_ids: list[list[str]], k: int = 5
) -> float:
    """MRR：第一个命中 rank r 得 1/r；未命中 0。"""
    if not retrieved_ids:
        return 0.0
    total = 0.0
    for retrieved, relevant in zip(retrieved_ids, relevant_ids, strict=True):
        rel_set = set(relevant)
        for rank, rid in enumerate(retrieved[:k], start=1):
            if rid in rel_set:
                total += 1.0 / rank
                break
    return total / len(retrieved_ids)


# 按 doc_type 生成 query 的模板
_QUERY_TEMPLATES = {
    "injury": "{team} 哪位球员受伤？预计缺阵多久？",
    "tactics": "{team} 最近比赛采用什么阵型？",
    "interview": "{team} 教练赛前说了什么？",
    "news": "{team} 最近一场比赛看点是什么？",
}


def build_eval_pairs(
    docs: list[DocRecord], *, seed: int = 42, n: int = 80
) -> list[dict]:
    """从语料自动生成评测对：每条 doc 生成一个 query，relevant = [doc_id]。

    按 doc_type 用模板生成 query；relevant_doc_ids 只含该 doc 的 id。
    """
    rng = random.Random(seed)
    sampled = rng.sample(docs, k=min(n, len(docs)))
    pairs: list[dict] = []
    for doc in sampled:
        template = _QUERY_TEMPLATES.get(doc.doc_type, "{team} 近期情况如何？")
        team = doc.team or "该队"
        query = template.format(team=team)
        pairs.append({
            "query": query,
            "relevant_doc_ids": [doc.doc_id],
            "doc_type": doc.doc_type,
            "league": doc.league,
            "team": doc.team,
        })
    return pairs
