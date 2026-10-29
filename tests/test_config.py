"""Tests for the configuration loader."""

from rag_assistant.config import Settings, load_settings


def test_defaults_load_from_yaml():
    settings = load_settings()
    assert settings.llm_provider in {"groq", "gemini", "openai", "ollama"}
    assert settings.chunk_size > settings.chunk_overlap
    assert settings.top_k > 0


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("RAG_USE_REWRITE_LOOP", "false")
    settings = load_settings()
    assert settings.top_k == 9
    assert settings.use_rewrite_loop is False


def test_with_overrides_returns_copy():
    base = Settings()
    changed = base.with_overrides(chunk_size=2000)
    assert changed.chunk_size == 2000
    assert base.chunk_size == 500
