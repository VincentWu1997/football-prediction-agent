"""全局配置：从环境变量 / .env 读取，路径以仓库根目录为基准。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/sports_agent/settings.py -> 上溯三级即仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """运行时设置。字段可被同名环境变量覆盖。"""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://sports:sports@localhost:5432/sports"
    ollama_base_url: str = "http://localhost:11434/v1"

    config_path: Path = REPO_ROOT / "config" / "models.yaml"
    data_raw_dir: Path = REPO_ROOT / "data" / "raw"


settings = Settings()
