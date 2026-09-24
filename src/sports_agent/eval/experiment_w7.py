"""W7 本地汇总与质量评测。

两个子命令：

1) E5 质量评测（方案 §5.5）：经 SSH 隧道连云端 vLLM，跑结构化输出任务。
   先建隧道：ssh -CNL 8000:localhost:8000 <autodl-ssh>
   然后：
       python -m sports_agent.eval.experiment_w7 quality --base-url http://localhost:8000
   指标：JSON 合法率、schema 字段准确率、数字与给定输入一致率（防 LLM 篡改概率）。

2) 汇总：解析从云机拉回的 E2/E3/E4 原始日志，产出对照表与图表。
       scp -r <autodl>:/root/.../benchmarks/results/w7 benchmarks/results/
       python -m sports_agent.eval.experiment_w7 summarize

所有汇总数字来自 benchmarks/results/w7 原始日志，缺失写空，不编造。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

import pandas as pd

from sports_agent.settings import REPO_ROOT

W7_DIR = REPO_ROOT / "benchmarks" / "results" / "w7"

LEVELS = ("bf16", "awq", "gptq", "fp8")


# ============ E5：结构化输出质量评测 ============


# 每条任务：给定已知概率（模拟工具返回），要求模型转成 schema JSON，
# 然后校验数字是否被模型篡改 + JSON 是否合法。
def _build_quality_tasks(n: int = 24) -> list[dict]:
    tasks: list[dict] = []
    rng_pairs = [
        ("Man United", "Liverpool"),
        ("Arsenal", "Chelsea"),
        ("Real Madrid", "Barcelona"),
        ("Bayern Munich", "Dortmund"),
    ]
    for i in range(n):
        h, a = rng_pairs[i % len(rng_pairs)]
        # 确定性生成已知概率（保证和为 1）
        ph = round(0.30 + (i % 5) * 0.02, 2)
        pa = round(0.40 + (i % 4) * 0.01, 2)
        pd_ = round(1 - ph - pa, 2)
        tasks.append(
            {
                "home": h,
                "away": a,
                "given": {"home": ph, "draw": pd_, "away": pa},
            }
        )
    return tasks


_QUALITY_SYS = (
    "你是足球预测结果格式化器。用户给你两队名和三个概率，"
    "你只能把这些数字原样填入 JSON，不得四舍五入或改动任何数字。"
    '只输出一行 JSON：{"home": <主队>, "away": <客队>, '
    '"prob_home": <主胜概率>, "prob_draw": <平局概率>, "prob_away": <客胜概率>}'
)


def run_quality(base_url: str, model: str, n: int = 24) -> None:
    """跑 E5 结构化输出评测，逐行记录。"""
    from openai import OpenAI

    client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key="EMPTY", timeout=120)
    tasks = _build_quality_tasks(n)
    rows: list[dict] = []
    for i, t in enumerate(tasks):
        user = (
            f"主队 {t['home']}，客队 {t['away']}。"
            f"主胜概率 {t['given']['home']}，平局概率 {t['given']['draw']}，"
            f"客胜概率 {t['given']['away']}。请输出 JSON。"
        )
        t0 = time.perf_counter()
        status = "ok"
        parsed = None
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _QUALITY_SYS},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            raw = r.choices[0].message.content or ""
        except Exception as e:
            raw = ""
            status = f"llm_error:{type(e).__name__}"
        lat = int((time.perf_counter() - t0) * 1000)

        json_valid = False
        schema_ok = False
        number_ok = False
        if status == "ok":
            try:
                from json_repair import repair_json

                parsed = json.loads(repair_json(raw))
                json_valid = True
            except Exception:
                status = "json_parse_fail"
        if json_valid and parsed is not None:
            required = {"home", "away", "prob_home", "prob_draw", "prob_away"}
            schema_ok = required.issubset(parsed.keys())
            if schema_ok:
                try:
                    number_ok = (
                        abs(float(parsed["prob_home"]) - t["given"]["home"]) < 1e-6
                        and abs(float(parsed["prob_draw"]) - t["given"]["draw"]) < 1e-6
                        and abs(float(parsed["prob_away"]) - t["given"]["away"]) < 1e-6
                    )
                except (TypeError, ValueError):
                    number_ok = False

        rows.append(
            {
                "i": i,
                "home": t["home"],
                "away": t["away"],
                "status": status,
                "json_valid": int(json_valid),
                "schema_ok": int(schema_ok),
                "number_consistent": int(number_ok),
                "latency_ms": lat,
                "raw": raw[:150],
            }
        )

    W7_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = W7_DIR / "e5_quality.csv"
    df.to_csv(out_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "n": len(df),
                "json_legal_rate": round(df["json_valid"].mean(), 4),
                "schema_field_accuracy": round(df["schema_ok"].mean(), 4),
                "number_consistency": round(df["number_consistent"].mean(), 4),
                "avg_latency_ms": round(df["latency_ms"].mean(), 1),
            }
        ]
    )
    summary.to_csv(W7_DIR / "e5_quality_summary.csv", index=False)
    print("=== E5 结构化输出质量 ===")
    print(summary.to_string(index=False))
    print(f"明细 -> {out_csv.relative_to(REPO_ROOT)}")


# ============ 汇总：E2/E3/E4 日志解析 ============

# benchmark_latency.py 常见输出键（不同 vLLM 版本措辞略有差异，逐个尝试）
_LAT_KEYS = {
    "decode_tput_tok_s": [
        r"decode\s*throughput[^\d]*([\d.]+)",
        r"decoder\s*throughput[^\d]*([\d.]+)",
    ],
    "tpot_ms": [r"time per output token[^\d]*([\d.]+)\s*ms", r"TPOT[^\d]*([\d.]+)"],
    "ttft_ms": [r"time to first token[^\d]*([\d.]+)\s*ms", r"TTFT[^\d]*([\d.]+)"],
}


def _extract_first(patterns: list[str], text: str) -> float | None:
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _ttft_p50(d: dict):
    """从 serving 结果 JSON 安全提取 TTFT p50（不同字段结构兼容）。"""
    ttft = d.get("ttft")
    if isinstance(ttft, dict):
        return ttft.get("p50")
    return d.get("ttft_p50_ms")


def summarize_e2() -> pd.DataFrame:
    rows: list[dict] = []
    for q in LEVELS:
        log = W7_DIR / f"e2_latency_{q}.log"
        mem_log = W7_DIR / f"e2_mem_{q}.log"
        if not log.exists():
            continue
        text = log.read_text(encoding="utf-8", errors="ignore")
        row: dict = {"quant": q, "ran": True}
        for k, pats in _LAT_KEYS.items():
            row[k] = _extract_first(pats, text)
        # 显存峰值（取采样日志最大值）
        peak = None
        if mem_log.exists():
            mems = [float(x) for x in re.findall(r"(\d+)\s*MiB", mem_log.read_text())]
            peak = max(mems) if mems else None
        row["peak_mem_mib"] = peak
        rows.append(row)
    return pd.DataFrame(rows)


def summarize() -> None:
    if not W7_DIR.exists():
        print(f"[ERROR] 目录不存在：{W7_DIR}（先从云机 scp 拉回结果）")
        sys.exit(1)

    # E2
    e2 = summarize_e2()
    if not e2.empty:
        e2.to_csv(W7_DIR / "e2_summary.csv", index=False)
        print("=== E2 量化对比（解析自原始日志）===")
        print(e2.to_string(index=False))
        _plot_e2(e2)
    else:
        print("[SKIP] 未找到 E2 日志")

    # E3：汇总各网格目录下 serving_c*.json
    e3_rows: list[dict] = []
    for grid_dir in sorted(W7_DIR.glob("e3_grid_*")):
        tag = grid_dir.name.replace("e3_grid_", "")
        for jf in sorted(grid_dir.glob("serving_c*.json")):
            try:
                d = json.loads(jf.read_text())
            except json.JSONDecodeError:
                continue
            e3_rows.append(
                {
                    "grid": tag,
                    "concurrency": d.get("max_concurrency"),
                    "request_throughput": d.get("request_throughput"),
                    "output_tput_tok_s": d.get("output_throughput"),
                    "ttft_p50_ms": _ttft_p50(d),
                    "tpot_ms": d.get("mean_tpot_ms"),
                }
            )
    if e3_rows:
        e3 = pd.DataFrame(e3_rows)
        e3.to_csv(W7_DIR / "e3_summary.csv", index=False)
        print("\n=== E3 并发/网格（解析自 serving JSON）===")
        print(e3.to_string(index=False))
        _plot_e3(e3)

    # E4
    e4_rows: list[dict] = []
    for backend in ("sglang", "vllm"):
        jf = W7_DIR / f"e4_{backend}" / "serving.json"
        if not jf.exists():
            continue
        try:
            d = json.loads(jf.read_text())
        except json.JSONDecodeError:
            continue
        e4_rows.append(
            {
                "backend": backend,
                "request_throughput": d.get("request_throughput"),
                "ttft_p50_ms": _ttft_p50(d),
                "output_tput_tok_s": d.get("output_throughput"),
            }
        )
    if e4_rows:
        e4 = pd.DataFrame(e4_rows)
        e4.to_csv(W7_DIR / "e4_summary.csv", index=False)
        print("\n=== E4 RadixAttention vs prefix caching ===")
        print(e4.to_string(index=False))

    print(f"\n汇总产物 -> {W7_DIR.relative_to(REPO_ROOT)}")


def _plot_e2(df: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib 未安装，跳过出图")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    valid = df.dropna(subset=["decode_tput_tok_s"])
    axes[0].bar(valid["quant"], valid["decode_tput_tok_s"])
    axes[0].set_title("Decode throughput (tok/s)")
    validm = df.dropna(subset=["peak_mem_mib"])
    axes[1].bar(validm["quant"], validm["peak_mem_mib"])
    axes[1].set_title("Peak memory (MiB)")
    fig.tight_layout()
    out = W7_DIR / "e2_plot.png"
    fig.savefig(out, dpi=120)
    print(f"E2 图 -> {out.relative_to(REPO_ROOT)}")


def _plot_e3(df: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for grid, g in df.groupby("grid"):
        g = g.sort_values("concurrency")
        ax.plot(g["concurrency"], g["output_tput_tok_s"], marker="o", label=grid)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("output throughput (tok/s)")
    ax.set_title("E3 vLLM concurrency scaling")
    ax.legend()
    fig.tight_layout()
    out = W7_DIR / "e3_plot.png"
    fig.savefig(out, dpi=120)
    print(f"E3 图 -> {out.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality", help="E5 结构化输出质量评测")
    q.add_argument("--base-url", default="http://localhost:8000")
    q.add_argument("--model", default="qwen3-8b-awq")
    q.add_argument("--n", type=int, default=24)
    sub.add_parser("summarize", help="汇总云机拉回的 E2/E3/E4 日志")
    args = parser.parse_args()
    if args.cmd == "quality":
        run_quality(args.base_url, args.model, n=args.n)
    elif args.cmd == "summarize":
        summarize()


if __name__ == "__main__":
    main()
