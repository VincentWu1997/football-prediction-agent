#!/usr/bin/env bash
# E4：SGLang RadixAttention vs vLLM prefix caching 的前缀复用对照（方案 §5.4）。
#
# 关键：SGLang 与 vLLM 共环境易依赖冲突，故 SGLang 装在独立 venv。
#
# 云机内运行（先完成 vLLM 部分；需已克隆 vllm-bench）：
#   bash benchmarks/sglang/bench_e4.sh
#
# 产物：benchmarks/results/w7/e4_<sglang|vllm>/{serve.log, serving.json}

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
OUT_DIR="benchmarks/results/w7"
WORKLOAD="$OUT_DIR/prefix_workload.jsonl"
VLLM_BENCH_DIR="${VLLM_BENCH_DIR:-/tmp/vllm-bench}"
SGLANG_VENV="${SGLANG_VENV:-$HOME/sglang-venv}"
PORT=8010

# 1) 生成负载
python benchmarks/sglang/make_prefix_workload.py --n 50 --out "$WORKLOAD"

# 2) 准备 SGLang venv
if [ ! -x "$SGLANG_VENV/bin/python" ]; then
  echo "创建 SGLang 独立 venv（含 sglang，约 5-10 分钟）"
  python -m venv "$SGLANG_VENV"
  "$SGLANG_VENV/bin/pip" install -U pip
  "$SGLANG_VENV/bin/pip" install "sglang[all]" 2>&1 | tail -3
fi
SP="$SGLANG_VENV/bin/python"

run_sglang() {
  local dir="$OUT_DIR/e4_sglang"; mkdir -p "$dir"
  echo ""
  echo "=== E4 [SGLang RadixAttention] ==="
  # RadixAttention 默认启用
  $SP -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B-AWQ \
    --port $PORT --host 0.0.0.0 \
    --mem-fraction-static 0.90 \
    > "$dir/serve.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 60); do
    curl -s "http://localhost:$PORT/health" >/dev/null 2>&1 && break
    sleep 3
  done
  set +e
  $SP -m sglang.bench_serving \
    --backend sglang \
    --base-url "http://localhost:$PORT" \
    --dataset-name sharegpt \
    --dataset-path "$WORKLOAD" \
    --num-prompts 50 \
    --max-concurrency 8 \
    --output-file "$dir/serving.json" \
    > "$dir/bench.log" 2>&1
  set -e
  kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; sleep 3
  # RadixAttention 命中率（serve 日志中 cache hit 统计）
  grep -iE "radix|cache hit|prefix" "$dir/serve.log" | tail -5 > "$dir/radix_hits.txt" || true
}

run_vllm() {
  local dir="$OUT_DIR/e4_vllm"; mkdir -p "$dir"
  echo ""
  echo "=== E4 [vLLM prefix caching] ==="
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B-AWQ \
    --served-model-name qwen3-8b-awq \
    --port $PORT --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --disable-log-requests \
    > "$dir/serve.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 60); do
    curl -s http://localhost:$PORT/health >/dev/null 2>&1 && break
    sleep 3
  done
  set +e
  python "$VLLM_BENCH_DIR/benchmarks/benchmark_serving.py" \
    --backend openai-chat \
    --base-url "http://localhost:$PORT" \
    --model qwen3-8b-awq \
    --dataset-name sharegpt \
    --dataset-path "$WORKLOAD" \
    --num-prompts 50 \
    --max-concurrency 8 \
    --save-result --result-dir "$dir" --result-filename serving.json \
    > "$dir/bench.log" 2>&1
  set -e
  kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; sleep 3
  grep -iE "prefix" "$dir/serve.log" | tail -5 > "$dir/prefix_hits.txt" || true
}

run_sglang
run_vllm

echo ""
echo "E4 完成，产物：$OUT_DIR/e4_{sglang,vllm}/"
echo "对照 TTFT / 吞吐 / 前缀命中率，由本地 experiment_w7.py 汇总。"
