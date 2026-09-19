"""加载 config/models.yaml：后端注册表 + 任务层级预算。"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from sports_agent.settings import settings


@dataclass(frozen=True)
class BackendConfig:
    """单个推理后端的连接与采样配置。"""

    name: str
    base_url: str
    model: str
    timeout_s: int = 60
    max_tokens: int = 1024
    temperature: float = 0.6
    api_key: str = "EMPTY"
    provider: str = "ollama"
    enabled: bool = True


@dataclass(frozen=True)
class LevelBudget:
    """任务层级预算：Planner 决定层级，执行循环在此预算内自主调工具。"""

    route: str
    max_tool_iters: int
    allow_simulation: bool


class ModelRegistry:
    """models.yaml 的内存视图。"""

    def __init__(self, config_path: Path | None = None) -> None:
        path = config_path or settings.config_path
        with open(path, encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

    @property
    def defaults(self) -> dict:
        return self._raw.get("defaults", {})

    def backend(self, name: str) -> BackendConfig:
        if name not in self._raw["backends"]:
            raise KeyError(f"未注册的推理后端: {name}")
        data = dict(self._raw["backends"][name])
        if not data.get("enabled", True):
            raise KeyError(f"推理后端已禁用: {name}")
        return BackendConfig(name=name, **data)

    def level(self, name: str) -> LevelBudget:
        if name not in self._raw["levels"]:
            raise KeyError(f"未定义的任务层级: {name}")
        data = self._raw["levels"][name]
        return LevelBudget(**data)

    @property
    def rag_config(self) -> dict:
        return self._raw.get("rag", {})
