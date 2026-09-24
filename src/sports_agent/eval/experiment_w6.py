"""W6 实验：路由分类准确率 + L1-L4 端到端 ReAct 跑通。

运行：python -m sports_agent.eval.experiment_w6
产物（benchmarks/results/w6/）：
- routing_metrics.csv      规则 vs LLM 分类准确率 / 加权 F1 / 平均延迟
- confusion_matrix.csv     规则版 4×4 混淆矩阵
- e2e_demo.json             L1-L4 各取 1 条 query，跑完 ReAct 循环，
                            记录 plan / final_answer / trace 摘要

依赖：路由评测集 benchmarks/eval_sets/routing.jsonl（100 条）；
LLM 分类需要 Ollama qwen3:4b；端到端 ReAct 需要 Ollama qwen3:4b（precise qwen3:8b 缺失时降级）。
任一不可用时跳过对应子实验并警告，不阻塞。
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict

import pandas as pd

from sports_agent.settings import REPO_ROOT

EVAL_SET = REPO_ROOT / "benchmarks" / "eval_sets" / "routing.jsonl"
OUT_DIR = REPO_ROOT / "benchmarks" / "results" / "w6"

LABELS = ["L1", "L2", "L3", "L4"]


def _load_eval() -> list[dict]:
    with EVAL_SET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _confusion(gold: list[str], pred: list[str]) -> pd.DataFrame:
    """gold → pred 的 4×4 混淆矩阵。"""
    cm = defaultdict(lambda: Counter())
    for g, p in zip(gold, pred, strict=True):
        cm[g][p] += 1
    rows = []
    for g in LABELS:
        rows.append({"gold": g, **{p: cm[g].get(p, 0) for p in LABELS}})
    return pd.DataFrame(rows)


def _f1_per_class(gold: list[str], pred: list[str]) -> dict[str, float]:
    """每类 F1（micro 简单实现）。"""
    f1s: dict[str, float] = {}
    for c in LABELS:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s[c] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1s


def _is_ollama_ready() -> bool:
    try:
        from sports_agent.inference.client import InferenceClient

        return "fast" in InferenceClient().available_routes()
    except Exception:
        return False


def _run_rule(eval_set: list[dict]) -> tuple[list[str], list[int]]:
    from sports_agent.agent.planner import Planner

    p = Planner()
    preds: list[str] = []
    lats: list[int] = []
    for item in eval_set:
        plan = p.plan(item["query"])
        preds.append(plan.level)
        lats.append(getattr(p, "_last_latency_ms", 0))
    return preds, lats


def _run_llm(
    eval_set: list[dict], inference_client, sample: int = 30
) -> tuple[list[str], list[int], list[dict]]:
    """LLM 版分类。sample 限制抽样数（默认 30 条，省时间），返回 (preds, lats, sampled_items)。"""
    import random

    from sports_agent.agent.planner import Planner

    p = Planner()
    # 分层抽样：每层取 min(sample//4, n) 条，保证 L1-L4 都有覆盖
    by_level: dict[str, list[dict]] = {}
    for item in eval_set:
        by_level.setdefault(item["level"], []).append(item)
    sampled: list[dict] = []
    per_level = max(1, sample // 4)
    rng = random.Random(42)
    for lv in ("L1", "L2", "L3", "L4"):
        pool = by_level.get(lv, [])
        sampled.extend(rng.sample(pool, min(per_level, len(pool))))
    sampled = sampled[:sample]
    rng.shuffle(sampled)

    preds: list[str] = []
    lats: list[int] = []
    for i, item in enumerate(sampled):
        plan = p.plan_with_llm(item["query"], inference_client)
        preds.append(plan.level)
        lats.append(getattr(p, "_last_llm_latency_ms", 0))
        if (i + 1) % 5 == 0:
            print(f"  LLM 分类进度 {i + 1}/{len(sampled)}")
    return preds, lats, sampled


def _e2e_demo(inference_client) -> list[dict]:
    """L1-L4 各取 1 条样本 query，跑完 ReAct 循环。"""
    from sports_agent.agent.graph import build_graph

    g = build_graph(inference=inference_client)
    samples = [
        ("L1", "英超最新积分榜"),
        ("L2", "Man United对Liverpool谁会赢"),
        ("L3", "详细分析Arsenal对Chelsea，考虑双方伤停"),
        ("L4", "模拟本赛季英超前四的归属概率"),
    ]
    out: list[dict] = []
    for level, q in samples:
        print(f"  [E2E] {level}: {q[:30]}")
        t0 = time.perf_counter()
        try:
            result = g.invoke({"query": q})
            elapsed = time.perf_counter() - t0
            out.append(
                {
                    "level": level,
                    "query": q,
                    "elapsed_s": round(elapsed, 2),
                    "plan": result.get("trace", [{}])[0].get("detail", {})
                    if result.get("trace")
                    else None,
                    "iters_used": result.get("iters_used"),
                    "final_answer": (result.get("final_answer") or "")[:500],
                    "n_tools_called": sum(
                        1 for t in result.get("trace", []) if t.get("kind") == "tool"
                    ),
                    "tool_names": [
                        t.get("name") for t in result.get("trace", []) if t.get("kind") == "tool"
                    ],
                    "inference_summary": result.get("inference"),
                    "sources": result.get("sources", []),
                    "trace_steps": [
                        {
                            "kind": t.get("kind"),
                            "name": t.get("name"),
                            "latency_ms": t.get("latency_ms"),
                        }
                        for t in result.get("trace", [])
                    ],
                }
            )
        except Exception as e:
            out.append(
                {
                    "level": level,
                    "query": q,
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                }
            )
    return out


def run() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_set = _load_eval()
    print(f"评测集: {len(eval_set)} 条（{EVAL_SET.relative_to(REPO_ROOT)}）")
    gold = [x["level"] for x in eval_set]

    # 1) 规则版
    print("\n=== 规则版 Planner ===")
    rule_preds, rule_lats = _run_rule(eval_set)
    rule_acc = sum(1 for g, p in zip(gold, rule_preds, strict=True) if g == p) / len(gold)
    rule_f1 = _f1_per_class(gold, rule_preds)
    rule_macro_f1 = sum(rule_f1.values()) / len(rule_f1)
    avg_lat = sum(rule_lats) / len(rule_lats)
    print(
        f"  Accuracy: {rule_acc:.3f}  Macro-F1: {rule_macro_f1:.3f}"
        f"  Avg latency: {avg_lat:.2f}ms"
    )

    # 2) LLM 版（可选）
    rows = [
        {
            "classifier": "rule",
            "accuracy": round(rule_acc, 4),
            "macro_f1": round(rule_macro_f1, 4),
            "avg_latency_ms": round(sum(rule_lats) / len(rule_lats), 2),
            "n_samples": len(eval_set),
        }
    ]
    llm_preds: list[str] | None = None
    llm_gold: list[str] = []
    if _is_ollama_ready():
        print("\n=== LLM 版 Planner (qwen3:4b few-shot) ===")
        from sports_agent.inference.client import InferenceClient

        client = InferenceClient()
        try:
            llm_preds, llm_lats, sampled = _run_llm(eval_set, client, sample=30)
            llm_gold = [item["level"] for item in sampled]
            llm_acc = sum(1 for g, p in zip(llm_gold, llm_preds, strict=True) if g == p) / len(
                llm_gold
            )
            llm_f1 = _f1_per_class(llm_gold, llm_preds)
            llm_macro_f1 = sum(llm_f1.values()) / len(llm_f1)
            llm_avg_lat = sum(llm_lats) / len(llm_lats)
            print(
                f"  Accuracy: {llm_acc:.3f}  Macro-F1: {llm_macro_f1:.3f}"
                f"  Avg latency: {llm_avg_lat:.2f}ms"
            )
            rows.append(
                {
                    "classifier": "llm",
                    "accuracy": round(llm_acc, 4),
                    "macro_f1": round(llm_macro_f1, 4),
                    "avg_latency_ms": round(sum(llm_lats) / len(llm_lats), 2),
                    "n_samples": len(sampled),
                }
            )
        except Exception as e:
            print(f"  [WARN] LLM 分类失败: {e}")
    else:
        print("\n[SKIP] LLM 版 Planner：Ollama qwen3:4b 不可用")

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "routing_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n路由指标 → {metrics_path.relative_to(REPO_ROOT)}")
    print(metrics_df.to_string(index=False))

    # 混淆矩阵（规则版；LLM 版若可用也输出）
    cm_df = _confusion(gold, rule_preds)
    cm_df.to_csv(OUT_DIR / "confusion_matrix_rule.csv", index=False)
    if llm_preds is not None:
        _confusion(llm_gold, llm_preds).to_csv(OUT_DIR / "confusion_matrix_llm.csv", index=False)
        print(f"混淆矩阵 → {OUT_DIR.relative_to(REPO_ROOT)}/confusion_matrix_{{rule,llm}}.csv")
    else:
        print(f"混淆矩阵 → {OUT_DIR.relative_to(REPO_ROOT)}/confusion_matrix_rule.csv")
    print(cm_df.to_string(index=False))

    # 3) L1-L4 端到端 ReAct demo
    print("\n=== L1-L4 端到端 ReAct demo ===")
    if _is_ollama_ready():
        from sports_agent.inference.client import InferenceClient

        client = InferenceClient()
        demo = _e2e_demo(client)
    else:
        demo = [{"error": "Ollama 不可用，跳过端到端 demo"}]
    demo_path = OUT_DIR / "e2e_demo.json"
    with demo_path.open("w", encoding="utf-8") as f:
        json.dump(demo, f, ensure_ascii=False, indent=2)
    print(f"端到端 demo → {demo_path.relative_to(REPO_ROOT)}")
    for d in demo:
        if "error" in d:
            print(f"  [{d.get('level', '?')}] ERROR: {d['error']}")
        else:
            ans = (d.get("final_answer") or "")[:80].replace("\n", " ")
            lv = d.get("level")
            nt = d.get("n_tools_called")
            it = d.get("iters_used")
            print(f"  [{lv}] tools={nt} iters={it} ans='{ans}...'")

    print(f"\n总耗时 {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(run())
