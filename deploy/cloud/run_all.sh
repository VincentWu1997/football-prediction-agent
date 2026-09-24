#!/usr/bin/env bash
# W7 云端一键流水线：环境 → GPTQ 量化 → E2 → E3 → E4。
#
# 云机内运行（把仓库放到 ~/sports-agent 或直接在仓库根目录）：
#   bash deploy/cloud/run_all.sh
#
# 可跳过部分：
#   SKIP_SGLANG=1   跳过 E4 的 SGLang（省装独立 venv 的时间）
#   SKIP_GPTQ=1     跳过自量化 GPTQ（E2 会自动跳过 gptq 档）
#
# 产物全部落在 benchmarks/results/w7/，跑完 scp 回本地 summarize。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$HOME/.bashrc" 2>/dev/null || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "########## 0) 环境初始化 ##########"
bash deploy/cloud/00_setup.sh

if [ "${SKIP_GPTQ:-0}" != "1" ]; then
  echo "########## 1) GPTQ 自量化 ##########"
  python deploy/cloud/quant_gptq.py
fi

echo "########## 2) E2 量化对比 ##########"
bash benchmarks/cloud_vllm/bench_e2.sh

echo "########## 3) E3 服务化调优 ##########"
bash benchmarks/cloud_vllm/bench_e3.sh

if [ "${SKIP_SGLANG:-0}" != "1" ]; then
  echo "########## 4) E4 SGLang 对照 ##########"
  bash benchmarks/sglang/bench_e4.sh
fi

echo ""
echo "########## 全部完成 ##########"
echo "结果目录：$REPO_ROOT/benchmarks/results/w7"
echo "本地拉回：scp -r <ssh>:'$REPO_ROOT/benchmarks/results/w7' benchmarks/results/"
echo "本地汇总：python -m sports_agent.eval.experiment_w7 summarize"
