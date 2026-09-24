#!/usr/bin/env bash
# E3：vLLM 服务化调优——并发吞吐/TTFT 曲线 + 参数网格。
#
# 方案 §5.3 要点：
#   - 并发 1→64 看吞吐 / TTFT；
#   - gpu_memory_utilization × max_num_seqs × max_model_len 网格；
#   - 启动日志提取 GPU KV cache blocks 数，解释 PagedAttention 分块机制。
#
# 压测严格用 vLLM 自带 benchmarks/benchmark_serving.py（不自造）。
#
# 云机内运行：
#   bash benchmarks/cloud_vllm/bench_e3.sh
#
# 产物：benchmarks/results/w7/e3_grid_<gpuutil>_<seqs>_<len>/
#         serve.log（含 KV blocks）、serving_c<并发>.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
OUT_DIR="benchmarks/results/w7"
VLLM_BENCH_DIR="${VLLM_BENCH_DIR:-/tmp/vllm-bench}"
SERVE_PY="$VLLM_BENCH_DIR/benchmarks/benchmark_serving.py"

MODEL="Qwen/Qwen3-8B-AWQ"   # 以 AWQ 为主（方案 §5.3）
SERVED_NAME="qwen3-8b-awq"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 8 16 32 64}"
NUM_PROMPTS="${E3_NUM_PROMPTS:-200}"
INPUT_LEN=512
OUTPUT_LEN=128

if [ ! -f "$SERVE_PY" ]; then
  echo "[ERROR] 未找到 $SERVE_PY，先运行 bash benchmarks/cloud_vllm/bench_e2.sh"
  exit 1
fi

# 启动服务（参数由环境变量覆盖）
start_server() {
  local gpuutil="$1" seqs="$2" maxlen="$3" logfile="$4"
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization "$gpuutil" \
    --max-num-seqs "$seqs" \
    --max-model-len "$maxlen" \
    --enable-prefix-caching \
    --disable-log-requests \
    > "$logfile" 2>&1 &
  SERVER_PID=$!
  # 等健康（最多 180s）
  for _ in $(seq 1 60); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[ERROR] 服务进程退出，见 $logfile"; tail -20 "$logfile"; return 1
    fi
    sleep 3
  done
  echo "[ERROR] 服务 180s 未就绪"; return 1
}

stop_server() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  sleep 3
}

# 参数网格（控制时长：默认只跑 1 个组合；全网格设 E3_FULL_GRID=1）
if [ "${E3_FULL_GRID:-0}" = "1" ]; then
  GRID=(
    "0.85 32 2048"
    "0.90 64 4096"
    "0.95 64 8192"
  )
else
  GRID=("0.90 64 4096")
fi

for combo in "${GRID[@]}"; do
  read -r GPUUTIL SEQS MAXLEN <<< "$combo"
  TAG="e3_grid_${GPUUTIL}_${SEQS}_${MAXLEN}"
  RUN_DIR="$OUT_DIR/$TAG"
  mkdir -p "$RUN_DIR"
  echo ""
  echo "=== E3 网格组合 gpu_mem=$GPUUTIL max_seqs=$SEQS max_len=$MAXLEN ==="

  start_server "$GPUUTIL" "$SEQS" "$MAXLEN" "$RUN_DIR/serve.log"

  # 提取 KV blocks 数（启动日志中 "# GPU blocks: N"）
  grep -E "GPU KV cache size|# GPU blocks" "$RUN_DIR/serve.log" | head -2 \
    > "$RUN_DIR/kv_blocks.txt" || true
  cat "$RUN_DIR/kv_blocks.txt"

  for C in $CONCURRENCY_LEVELS; do
    echo "  -> 并发 $C（prompts=$NUM_PROMPTS）"
    set +e
    python "$SERVE_PY" \
      --backend openai-chat \
      --base-url http://localhost:8000 \
      --model "$SERVED_NAME" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$C" \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --save-result \
      --result-dir "$RUN_DIR" \
      --result-filename "serving_c${C}.json" \
      > "$RUN_DIR/serving_c${C}.log" 2>&1
    RC=$?
    set -e
    [ "$RC" -ne 0 ] && echo "[WARN] 并发 $C 压测非零，见 serving_c${C}.log"
  done

  stop_server
done

echo ""
echo "E3 完成，产物：$OUT_DIR/e3_grid_*"
