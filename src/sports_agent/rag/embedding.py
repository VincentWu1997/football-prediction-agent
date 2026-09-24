"""嵌入客户端：bge-m3 via Ollama 原生 /api/embeddings（默认），或 Mock（测试用）。

设计要点：
- 不引入 torch；走 Ollama 本地服务（不用有 bug 的 /v1/embeddings 端点）；
- 所有实现返回 1024 维向量；
- get_embedding_client() 读 config/models.yaml rag 段，RAG_EMBEDDING=mock 可切 Mock。
"""

import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from sports_agent.inference.registry import ModelRegistry
from sports_agent.settings import settings


class EmbeddingClient(Protocol):
    """嵌入客户端协议：所有实现必须返回 1024 维、L2 归一化的向量。"""

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def status(self) -> dict: ...


class OllamaEmbeddingClient:
    """通过 Ollama 原生 /api/embeddings 调 bge-m3。

    注意：不用 OpenAI 兼容的 /v1/embeddings——实测 Ollama 0.34.2 该端点
    对 bge-m3 会忽略输入、返回常量向量（已确认为 bug）。
    原生端点一次只接收一条 prompt，用线程并发降低往返开销。
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int = 1024,
        max_workers: int = 8,
    ) -> None:
        reg = ModelRegistry()
        rag = reg.rag_config
        url = base_url or settings.ollama_base_url
        # 去掉末尾 /v1，原生端点挂在服务根路径
        self._root = url.rstrip("/").removesuffix("/v1")
        self._model = model or rag.get("embedding_model", "bge-m3")
        self._dimensions = dimensions
        self._max_workers = max_workers

    def _embed_one(self, text: str) -> list[float]:
        payload = json.dumps({"model": self._model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self._root}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        if "embedding" not in data:
            raise RuntimeError(f"Ollama 未返回 embedding: {data}")
        vec = data["embedding"]
        if len(vec) != self._dimensions:
            raise AssertionError(
                f"嵌入维度 {len(vec)} != 预期 {self._dimensions}；"
                f"模型 {self._model} 可能不是 bge-m3"
            )
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(self._embed_one, texts))

    def status(self) -> dict:
        url = f"{self._root}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                payload = json.loads(resp.read())
                ids = [m.get("name", "").split(":")[0] for m in payload.get("models", [])]
            return {
                "provider": "ollama",
                "model": self._model,
                "reachable": True,
                "model_available": self._model in ids,
            }
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            return {
                "provider": "ollama",
                "model": self._model,
                "reachable": False,
                "error": str(e.reason) if isinstance(e, urllib.error.URLError) else str(e),
            }


class MockEmbeddingClient:
    """确定性哈希嵌入：sha256(text) 派生 1024 维向量后 L2 归一。

    不携带语义信息，仅供 ingest/retrieve 单元测试与 CI 使用。
    同一文本两次 embed 结果一致；不同文本向量不同。
    """

    def __init__(self, dim: int = 1024, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(f"{self.seed}:{t}".encode()).digest()
            rng = random.Random(h)
            v = [rng.gauss(0, 1) for _ in range(self.dim)]
            norm = sum(x * x for x in v) ** 0.5
            out.append([x / norm for x in v])
        return out

    def status(self) -> dict:
        return {"provider": "mock", "model": "hash-1024", "reachable": True}


def get_embedding_client() -> EmbeddingClient:
    """工厂：读 config/models.yaml rag 段 + 环境变量。

    - RAG_EMBEDDING=mock  -> MockEmbeddingClient（测试/CI）
    - 默认               -> OllamaEmbeddingClient（bge-m3 via Ollama）
    """
    if os.environ.get("RAG_EMBEDDING", "").lower() == "mock":
        return MockEmbeddingClient()
    return OllamaEmbeddingClient()
