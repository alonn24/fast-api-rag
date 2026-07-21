import pytest
from app.config import Settings


def test_settings_loads_required_fields_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key == "sk-ant-test"
    assert settings.voyage_api_key == "pa-test"
    assert settings.embedding_model == "voyage-3"
    assert settings.claude_model == "claude-opus-4-8"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50


def test_settings_missing_required_field_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(Exception):
        Settings(_env_file=None)
