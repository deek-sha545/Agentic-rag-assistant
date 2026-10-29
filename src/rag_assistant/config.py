"""Load settings from config.yaml (plus optional environment overrides).

Usage:
    from rag_assistant.config import load_settings
    settings = load_settings()
    print(settings.llm_model)

Any field can be overridden with an environment variable named
RAG_<FIELD_NAME_IN_UPPER_CASE>, e.g. `RAG_LLM_PROVIDER=gemini`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

# The project root is two levels above this file: src/rag_assistant/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Settings:
    """All tunable settings. Defaults here mirror config.yaml."""

    # LLM for the agent
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    # LLM used as a judge in evaluation
    judge_provider: str = "groq"
    judge_model: str = "openai/gpt-oss-120b"

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "chroma_db"
    chunk_size: int = 500
    chunk_overlap: int = 75

    # Agent behaviour
    top_k: int = 8
    max_retries: int = 2
    use_rewrite_loop: bool = True
    check_grounding: bool = True
    sanitize_injections: bool = True
    memory_turns: int = 3

    # Paths
    raw_data_dir: str = "data/raw"
    poisoned_data_dir: str = "data/poisoned"
    log_dir: str = "logs"
    cache_dir: str = ".cache/llm"
    use_llm_cache: bool = False

    # model name -> [input price, output price] in US$ per 1M tokens
    prices: dict = field(default_factory=dict)

    def path(self, value: str) -> Path:
        """Turn a (possibly relative) path from the config into an absolute one."""
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def with_overrides(self, **changes) -> "Settings":
        """Return a copy with some fields changed (used by experiments)."""
        return replace(self, **changes)


def _convert(value: str, current):
    """Convert an environment-variable string to the type of the current value."""
    if isinstance(current, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def load_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Settings:
    """Read config.yaml, apply RAG_* environment overrides, return Settings."""
    # Load API keys from .env into environment variables (never printed).
    load_dotenv(PROJECT_ROOT / ".env")

    values: dict = {}
    config_path = Path(config_path)
    if config_path.exists():
        values = yaml.safe_load(config_path.read_text()) or {}

    known = {f.name for f in fields(Settings)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown setting(s) in {config_path.name}: {sorted(unknown)}")

    settings = Settings(**values)

    # Environment overrides, e.g. RAG_LLM_MODEL=llama-3.1-8b-instant
    overrides = {}
    for name in known - {"prices"}:
        env_value = os.getenv(f"RAG_{name.upper()}")
        if env_value is not None:
            overrides[name] = _convert(env_value, getattr(settings, name))
    return settings.with_overrides(**overrides)
