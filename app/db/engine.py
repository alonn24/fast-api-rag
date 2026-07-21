from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
