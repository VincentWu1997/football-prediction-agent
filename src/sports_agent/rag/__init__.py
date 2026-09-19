"""RAG 层（W5）：只承载非结构化文本（伤停新闻、战术报告、赛前采访）。

- ingest.py    文档切片（512-1024 token，重叠 10%）+ bge-m3 向量化入库
- retrieve.py  HNSW cosine top-k + 元数据过滤，返回带来源的文本块
- embedding 默认走 Ollama /v1/embeddings（bge-m3），避免 Python 侧引入 torch
"""
