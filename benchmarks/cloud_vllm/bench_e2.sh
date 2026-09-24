#!/usr/bin/env bash
# E2：量化档离线解码性能对比（BF16 / AWQ / GPTQ / FP8）。
#
# 严格使用 vLLM 自带 benchmarks/benchmark_latency.py（方案 §5.3：不自造压测器）。
#
# 云机内运行（先 00_setup.sh，模型已下载/量化）：
#   bash benchmarks/cloud_vllm/bench_e2.sh
#
# 产物：benchmarks/results/w7/e2_latency_<quant>.log（vLLM 原始输出）
#       汇总由本地 experiment_w7.py 解析这些 log。
#
# 指标：decode tok/s、TPOT、首 token 延迟；峰值显存另用 nvidia-smi 采样。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
OUT_DIR="benchmarks/results/w7"
mkdir -p "$OUT_DIR"

# 1) 获取 vLLM 自带 benchmark 脚本（按已装版本浅克隆，避免手造压测器）
VLLM_BENCH_DIR="${VLLM_BENCH_DIR:-/tmp/vllm-bench}"
if [ ! -f "$VLLM_BENCH_DIR/benchmarks/benchmark_latency.py" ]; then
  VLLM_VER="$(python -c 'import vllm; print(vllm.__version__)')"
  echo "克隆 vLLM v$VLLM_VER 的 benchmark 脚本到 $VLLM_BENCH_DIR"
  git clone --depth 1 --branch "v$VLLM_VER" https://github.com/vllm-project/vllm.git \
    "$VLLM_BENCH_DIR" || {
    echo "[WARN] GitHub 克隆失败，尝试 hf-mirror 不可；请手动放置 vllm benchmarks"
    exit 1
  }
fi
LAT_PY="$VLLM_BENCH_DIR/benchmarks/benchmark_latency.py"

# 2) 量化档 → 模型路径
declare -A MODELS=(
  [bf16]="Qwen/Qwen3-8B"
  [awq]="Qwen/Qwen3-8B-AWQ"
  [gptq]="$(pwd)/Qwen3-8B-GPTQ"
  [fp8]="Qwen/Qwen3-8B-FP8"
)

# 3) 逐档压测；后台采显存峰值
INPUT_LEN=1024
OUTPUT_LEN=512
N_PROMPTS=1

for QUANT in bf16 awq gptq fp8; do
  MODEL="${MODELS[$QUANT]}"
  if [ "$QUANT" = "gptq" ] && [ ! -d "$MODEL" ]; then
    echo "[SKIP] GPTQ 模型不存在（先跑 deploy/cloud/quant_gptq.py）"
    continue
  fi
  echo ""
  echo "=== E2 [$QUANT] $MODEL ==="
  LOG="$OUT_DIR/e2_latency_${QUANT}.log"

  # 显存采样（每 0.5s 取 used MiB），压测结束后停
  nvidia-smi --query-gpu=memory.used --format=csv -l 1 > "$OUT_DIR/e2_mem_${QUANT}.log" &
  MEM_PID=$!

  set +e
  python "$LAT_PY" \
    --model "$MODEL" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --num-prompts "$N_PROMPTS" \
    2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  set -e

  kill "$MEM_PID" 2>/dev/null || true
  if [ "$RC" -ne 0 ]; then
    echo "[NOTE] $QUANT 压测非零退出（FP8 在 Ada 上可能不支持，属预期，记录原始日志）"
  fi
done

echo ""
echo "E2 完成，原始日志：$OUT_DIR/e2_latency_*.log / e2_mem_*.log"
