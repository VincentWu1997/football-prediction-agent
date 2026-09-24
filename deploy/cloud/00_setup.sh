#!/usr/bin/env bash
# W7 云端环境初始化（AutoDL RTX 4090 / CUDA 12.4 / PyTorch 镜像）。
#
# 作用：
#   1) 配置 HF 国内镜像（AutoDL 学术加速 / hf-mirror）；
#   2) 安装 vLLM>=0.8.5（E2/E3）、GPTQModel（E2 自量化）；
#   3) 校验 GPU 与 vLLM 版本。
#
# 用法（云机内）：
#   bash deploy/cloud/00_setup.sh
#
# 设计：分步打印、失败即停（set -euo pipefail）；可重复执行（幂等）。

set -euo pipefail

echo "=== 1) GPU / CUDA 信息 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
nvcc --version 2>/dev/null | tail -2 || true

echo ""
echo "=== 2) 配置 HuggingFace 国内镜像 ==="
# AutoDL 自带学术加速；huggingface.co 被阻断时用 hf-mirror.com
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 持久化到 ~/.bashrc（幂等：grep 去重）
if ! grep -q "HF_ENDPOINT" "$HOME/.bashrc" 2>/dev/null; then
  echo "export HF_ENDPOINT=https://hf-mirror.com" >> "$HOME/.bashrc"
fi
echo "HF_ENDPOINT=$HF_ENDPOINT"

echo ""
echo "=== 3) 安装 vLLM (>=0.8.5) ==="
# AutoDL 镜像已带 torch；用 --no-build-isolation 避免重装 torch
pip install -U "vllm>=0.8.5" 2>&1 | tail -5

echo ""
print_version() {
  python - "$1" <<'PY'
import importlib, sys
name = sys.argv[1]
try:
    m = importlib.import_module(name)
    print(f"{name}: {getattr(m, '__version__', 'unknown')}")
except Exception as e:
    print(f"{name}: NOT INSTALLED ({e})")
PY
}
print_version vllm
print_version torch

echo ""
echo "=== 4) 安装 GPTQModel（E2 自量化 GPTQ）==="
pip install -U gptqmodel 2>&1 | tail -3
print_version gptqmodel

echo ""
echo "=== Setup 完成 ==="
echo "下一步："
echo "  # 自量化 GPTQ（可选，~15 分钟）"
echo "  python deploy/cloud/quant_gptq.py"
echo "  # 启动 vLLM 服务（不同量化档）"
echo "  bash deploy/cloud/10_serve.sh awq   # bf16 / awq / gptq / fp8"
