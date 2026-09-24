"""OpenAI 兼容推理客户端：Ollama / vLLM / SGLang 通用。

设计要点：
- 所有后端只暴露 /v1/chat/completions，故直接使用 openai SDK；
- Qwen3 默认先输出思考内容，fast 路由延迟会失真，通过 chat_template_kwargs 关闭；
- 每次调用返回延迟与 token 用量，供推理 trace 与 W7 实验复用；
- status() 对双档（fast/precise）做 /models 探测，供服务健康页展示。
"""

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from openai import OpenAI

from sports_agent.inference.registry import BackendConfig, ModelRegistry

_THINK_RE = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)


def _strip_think(content: str) -> str:
    """去掉 Qwen3 输出的 <think>...</think> 思考块，返回真正答案。

    若没有 </think> 结束标签，说明模型在思考中被截断，返回原内容（调用方可见思考过程）。
    """
    m = _THINK_RE.match(content)
    if m:
        return content[m.end() :].strip()
    return content


@dataclass
class CompletionResult:
    """一次推理调用的结果与可观测指标。"""

    content: str
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    tool_calls: list[dict] | None = None  # OpenAI tool_calls（W6 ReAct 循环用）
    finish_reason: str | None = None  # stop / tool_calls / length


class InferenceClient:
    """按后端名懒加载 OpenAI 客户端。"""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self._clients: dict[str, OpenAI] = {}
        # 实测（Ollama 0.34.2 + qwen3:4b）：chat_template_kwargs.enable_thinking=false
        # 反而会拉长推理过程且可能导致 content 为空，故不发送该参数；
        # 默认行为下 Qwen3 用 ~120 tokens 完成思考+作答，可接受。
        self._enable_thinking: bool = self.registry.defaults.get("enable_thinking", False)
        self._routes_cache: set[str] | None = None  # 可用路由缓存（§6.5 降级用）

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
        max_retries: int = 2,
    ) -> CompletionResult:
        """同步调用指定档位（fast/precise/...）后端。

        容错（§6.5）：
        - 指数退避重试 max_retries 次（1s → 2s → 4s）；
        - 仅对连接/超时类异常重试，4xx 业务错误（鉴权/参数错）直接抛出。
        """
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
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        if tools:
            kwargs["tools"] = tools

        start = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                completion = client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                last_exc = e
                # 4xx 业务错误不重试（openai.APIStatusError.status_code 4xx）
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise
                if attempt == max_retries:
                    raise
                backoff = 2**attempt  # 1, 2, 4 ...
                time.sleep(backoff)
        else:
            if last_exc:
                raise last_exc
        latency_ms = int((time.perf_counter() - start) * 1000)

        message = completion.choices[0].message
        usage = completion.usage
        # 提取 tool_calls（W6 ReAct 循环）
        tool_calls: list[dict] | None = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]
        # Qwen3 默认先输出 <think>...</think> 再回答；Ollama /v1/chat/completions
        # 会把 thinking 合并进 content，导致 answer 被思考内容淹没。
        # 这里把 </think> 之后的内容作为真正答案。
        content = message.content or ""
        content = _strip_think(content)
        return CompletionResult(
            content=content,
            model=cfg.model,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            tool_calls=tool_calls,
            finish_reason=completion.choices[0].finish_reason,
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

    def available_routes(self, refresh: bool = False) -> set[str]:
        """返回可用路由集合（reachable=True 且 model_available=True）。

        缓存到首次调用后；refresh=True 强制重测。供 §6.5 路由降级使用。
        """
        if self._routes_cache is not None and not refresh:
            return self._routes_cache
        st = self.status()
        self._routes_cache = {
            r for r, v in st.items() if v.get("reachable") and v.get("model_available")
        }
        return self._routes_cache
