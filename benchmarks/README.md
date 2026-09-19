# 实验与评测（benchmarks/）

本目录是简历中"推理优化"关键词的全部证据来源。**纪律：README 里的每个性能/质量数字都必须能从 `results/` 的原始数据复现。**

## 实验索引（对应方案 v2 §5）

| 编号 | 名称 | 脚本位置 | 原始结果 | 周次 |
|---|---|---|---|---|
| E1 | 本地 GGUF 多精度（Q4_K_M/Q5_K_M/Q6_K/Q8_0，延迟/tok/s/内存） | `local_gguf/` | `results/e1/` | W4-W7 |
| E2 | vLLM 量化对比（BF16/AWQ/GPTQ/FP8，官方 checkpoint） | `cloud_vllm/` | `results/e2/` | W7 |
| E3 | vLLM 服务化调优（并发曲线、KV blocks、gpu_memory_utilization 网格） | `cloud_vllm/` | `results/e3/` | W7 |
| E4 | SGLang RadixAttention 公共前缀复用 vs vLLM | `sglang/` | `results/e4/` | W7 |
| E5 | 体育领域质量评测（JSON 合法率/字段一致率/正确率/路由准确率） | `eval_sets/` | `results/e5/` | W5-W7 |

## 每个实验必须提交的三件套

1. 可复现脚本（含环境说明：GPU 型号、vLLM/SGLang 版本、启动命令）
2. 原始结果 CSV/JSON（每次运行一份，文件名带日期与参数）
3. 出图脚本（matplotlib 从原始数据生成，图表与数据一一对应）

引用论文数字（如 PagedAttention 吞吐提升）必须在图表脚注标注论文，不与自测结果混列。

## 评测集格式

`eval_sets/` 下 jsonl，每行一条：

- `routing.jsonl`：`{"query": "...", "level": "L3"}`，用于 Planner 分类准确率/混淆矩阵（样例见 `routing.sample.jsonl`，正式集 W6 扩到 100 条）
- `quality.jsonl`：`{"query": "...", "expect_schema": {...}, "answer_key": "..."}`
- `retrieval.jsonl`：`{"query": "...", "relevant_doc_ids": ["..."]}`，用于 recall@5 / MRR
