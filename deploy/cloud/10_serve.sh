#!/usr/bin/env bash
# W7 vLLM 服务启动器：按量化档启动 OpenAI 兼容服务。
#
# 用法（云机内）：
#   bash deploy/cloud/10_serve.sh <quant> [port]
#     quant = bf16 | awq | gptq | fp8
#     port  默认 8000
#
# 示例：
#   bash deploy/cloud/10_serve.sh bf16        # BF16 基线
#   bash deploy/cloud/10_serve.sh awq         # 官方 AWQ checkpoint
#   bash deploy/cloud/10_serve.sh gptq        # 自量化 GPTQ（先跑 quant_gptq.py）
#   bash deploy/cloud/10_serve.sh fp8         # 官方 FP8（Ada 仅验证可跑）
#
# 关键参数（E3 网格调优时覆盖）：
#   VLLM_GPU_MEM_UTIL (默认 0.90)
#   VLLM_MAX_NUM_SEQS (默认 64)
#   VLLM_MAX_MODEL_LEN (默认 4096)
#   VLLM_PREFIX_CACHING (默认 1)
#
# 设计：每次启动写日志到 logs/serve_<quant>.log，便于提取 GPU KV blocks 数。

set -euo pipefail

QUANT="${1:?用法: 10_serve.sh <bf16|awq|gptq|fp8> [port]}"
PORT="${2:-8000}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_PREFIX_CACHING="${VLLM_PREFIX_CACHING:-1}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/serve_${QUANT}.log"

case "$QUANT" in
  bf16) MODEL="Qwen/Qwen3-8B" ;;
  awq)  MODEL="Qwen/Qwen3-8B-AWQ" ;;
  gptq)
    # 自量化产物路径（quant_gptq.py 输出）
    GPTQ_DIR="${GPTQ_DIR:-./Qwen3-8B-GPTQ}"
    if [ ! -d "$GPTQ_DIR" ]; then
      echo "[ERROR] GPTQ 目录不存在: $GPTQ_DIR（先运行 python deploy/cloud/quant_gptq.py）"
      exit 1
    fi
    MODEL="$GPTQ_DIR"
    ;;
  fp8)  MODEL="Qwen/Qwen3-8B-FP8" ;;
  *)
    echo "[ERROR] 未知量化档: $QUANT（支持 bf16/awq/gptq/fp8）"
    exit 1
    ;;
esac

PREFIX_FLAG=""
if [ "$VLLM_PREFIX_CACHING" = "1" ]; then
  PREFIX_FLAG="--enable-prefix-caching"
fi

echo "=== vLLM serve ==="
echo "  quant      : $QUANT"
echo "  model      : $MODEL"
echo "  port       : $PORT"
echo "  gpu_mem    : $VLLM_GPU_MEM_UTIL"
echo "  max_seqs   : $VLLM_MAX_NUM_SEQS"
echo "  max_len    : $VLLM_MAX_MODEL_LEN"
echo "  prefix     : $VLLM_PREFIX_CACHING"
echo "  log        : $LOG_FILE"
echo ""

# --disable-log-requests 关闭请求日志（压测时减少 IO）；KV blocks 数在启动日志中
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "qwen3-8b-${QUANT}" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  $PREFIX_FLAG \
  --disable-log-requests \
  2>&1 | tee "$LOG_FILE"
