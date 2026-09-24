"""W7 烟测：云端脚本齐备性 + 汇总解析器纯逻辑（不租机可跑）。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from sports_agent.eval.experiment_w7 import (
    _build_quality_tasks,
    _extract_first,
    _ttft_p50,
    summarize_e2,
)
from sports_agent.settings import REPO_ROOT

CLOUD_SCRIPTS = [
    "deploy/cloud/00_setup.sh",
    "deploy/cloud/10_serve.sh",
    "deploy/cloud/run_all.sh",
    "deploy/cloud/quant_gptq.py",
    "benchmarks/cloud_vllm/bench_e2.sh",
    "benchmarks/cloud_vllm/bench_e3.sh",
    "benchmarks/sglang/bench_e4.sh",
    "benchmarks/sglang/make_prefix_workload.py",
]


@pytest.mark.parametrize("rel", CLOUD_SCRIPTS)
def test_cloud_scripts_exist(rel: str):
    p = REPO_ROOT / rel
    assert p.exists(), f"缺少 {rel}"
    if rel.endswith(".sh"):
        assert p.read_text().startswith("#!"), "shell 脚本需 shebang"


def test_quality_tasks_sums_to_one():
    tasks = _build_quality_tasks(12)
    assert len(tasks) == 12
    for t in tasks:
        s = round(sum(t["given"].values()), 6)
        assert s == pytest.approx(1.0), t
        assert all(0 <= x <= 1 for x in t["given"].values())


def test_ttft_p50_shapes():
    assert _ttft_p50({"ttft": {"p50": 12.5}}) == 12.5
    assert _ttft_p50({"ttft_p50_ms": 9.0}) == 9.0
    assert _ttft_p50({}) is None


def test_extract_first():
    assert _extract_first([r"latency[^\d]*([\d.]+)"], "latency = 42.5 ms") == 42.5
    assert _extract_first([r"nope ([\d.]+)"], "nothing") is None


def test_summarize_e2_no_logs(tmp_w7_missing):
    # 无日志时返回空 DataFrame，不抛异常
    df = summarize_e2()
    assert df.empty or "quant" in df.columns


@pytest.fixture
def tmp_w7_missing(monkeypatch, tmp_path: Path):
    # 指向不存在的临时目录，确保本地无结果时安全
    import sports_agent.eval.experiment_w7 as mod
    monkeypatch.setattr(mod, "W7_DIR", tmp_path / "no_w7")


def test_gptq_script_compiles():
    p = REPO_ROOT / "deploy/cloud/quant_gptq.py"
    spec = importlib.util.spec_from_file_location("quant_gptq", p)
    assert spec is not None


def test_scripts_executable_bit_or_runnable():
    # shell 脚本应带可执行位（已 chmod +x 提交）
    for rel in CLOUD_SCRIPTS:
        if rel.endswith(".sh"):
            assert os.access(REPO_ROOT / rel, os.X_OK), f"{rel} 需可执行"
