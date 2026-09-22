"""嵌入客户端：bge-m3 via Ollama /v1/embeddings（默认），或 Mock（测试用）。

设计要点：
- 复用 openai SDK 调 Ollama，与 inference/client.py 同构，不引入 torch；
- 所有实现返回 1024 维、L2 归一化的 float 向量；
- get_embedding_client() 读 config/models.yaml rag 段，RAG_EMBEDDING=mock 可切 Mock。
"""

import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from typing import Protocol

from openai import OpenAI

from sports_agent.inference.registry import ModelRegistry
from sports_agent.settings import settings


class EmbeddingClient(Protocol):
    """嵌入客户端协议：所有实现必须返回 1024 维、L2 归一化的向量。"""

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def status(self) -> dict: ...


class OllamaEmbeddingClient:
    """通过 Ollama /v1/embeddings 调 bge-m3；复用 openai SDK。"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int = 1024,
    ) -> None:
        reg = ModelRegistry()
        rag = reg.rag_config
        self._base_url = base_url or f"{settings.ollama_base_url.rstrip('/v1')}/v1"
        # ollama_base_url 已含 /v1 后缀，直接用
        self._base_url = base_url or settings.ollama_base_url
        self._model = model or rag.get("embedding_model", "bge-m3")
        self._dimensions = dimensions
        self._client = OpenAI(base_url=self._base_url, api_key="EMPTY", timeout=60)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vectors = [d.embedding for d in resp.data]
        for v in vectors:
            assert len(v) == self._dimensions, (
                f"嵌入维度 {len(v)} != 预期 {self._dimensions}；"
                f"模型 {self._model} 可能不是 bge-m3"
            )
        return vectors

    def status(self) -> dict:
        url = f"{self._base_url.rstrip('/')}/models"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                payload = json.loads(resp.read())
                ids = [m.get("id") for m in payload.get("data", [])]
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
