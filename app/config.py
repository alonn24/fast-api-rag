from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    voyage_api_key: str
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag"
    test_database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag_test"
    embedding_model: str = "voyage-3"
    embedding_dim: int = 1024
    claude_model: str = "claude-opus-4-8"
    chunk_size: int = 500
    chunk_overlap: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
