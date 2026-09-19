"""数据库连接：SQLAlchemy engine 单例。"""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from sports_agent.settings import settings

_engine: Engine | None = None


def get_engine() -> Engine:
    """惰性创建全局 engine（连接失败不会阻断不依赖 DB 的模块导入）。"""
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, future=True, pool_pre_ping=True)
    return _engine
