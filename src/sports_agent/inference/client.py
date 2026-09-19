"""OpenAI 兼容推理客户端：Ollama / vLLM / SGLang 通用。

设计要点：
- 所有后端只暴露 /v1/chat/completions，故直接使用 openai SDK；
- Qwen3 默认先输出思考内容，fast 路由延迟会失真，通过 chat_template_kwargs 关闭；
- 每次调用返回延迟与 token 用量，供推理 trace 与 W7 实验复用；
- status() 对双档（fast/precise）做 /models 探测，供服务健康页展示。
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from openai import OpenAI

from sports_agent.inference.registry import BackendConfig, ModelRegistry


@dataclass
class CompletionResult:
    """一次推理调用的结果与可观测指标。"""

    content: str
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int


class InferenceClient:
    """按后端名懒加载 OpenAI 客户端。"""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self._clients: dict[str, OpenAI] = {}
        # 实测（Ollama 0.34.2 + qwen3:4b）：chat_template_kwargs.enable_thinking=false
        # 反而会拉长推理过程且可能导致 content 为空，故不发送该参数；
        # 默认行为下 Qwen3 用 ~120 tokens 完成思考+作答，可接受。
        self._enable_thinking: bool = self.registry.defaults.get("enable_thinking", False)

    def _get_sdk_client(self, cfg: BackendConfig) -> OpenAI:
        if cfg.name not in self._clients:
            self._clients[cfg.name] = OpenAI(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                timeout=cfg.timeout_s,
            )
        return self._clients[cfg.name]

    def chat(
        self,
        route: str,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
    ) -> CompletionResult:
        """同步调用指定档位（fast/precise/...）后端。"""
        cfg = self.registry.backend(route)
        client = self._get_sdk_client(cfg)

        kwargs: dict = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        # 仅在显式开启思考时才透传参数（关闭时发送反而恶化，见 __init__ 注释）
        if self._enable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True}
            }
        if tools:
            kwargs["tools"] = tools

        start = time.perf_counter()
        completion = client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)

        message = completion.choices[0].message
        usage = completion.usage
        return CompletionResult(
            content=message.content or "",
            model=cfg.model,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def status(self, timeout_s: float = 2.0) -> dict:
        """探测各档位后端可达性（GET /models），不触发推理。"""
        result: dict = {}
        for route in ("fast", "precise"):
            cfg = self.registry.backend(route)
            url = f"{cfg.base_url.rstrip('/')}/models"
            try:
                with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                    payload = json.loads(resp.read())
                    ids = [m.get("id") for m in payload.get("data", [])]
                result[route] = {
                    "base_url": cfg.base_url,
                    "model": cfg.model,
                    "reachable": True,
                    "model_available": cfg.model in ids,
                }
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                result[route] = {
                    "base_url": cfg.base_url,
                    "model": cfg.model,
                    "reachable": False,
                    "error": str(e.reason) if isinstance(e, urllib.error.URLError) else str(e),
                }
        return result
