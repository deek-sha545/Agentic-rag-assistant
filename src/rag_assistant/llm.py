"""Model-agnostic LLM layer.

The rest of the code only ever calls `LLMClient.complete(system, user)`. Which
provider answers is decided by config.yaml:

    llm_provider: groq     -> langchain-groq        (needs GROQ_API_KEY)
    llm_provider: gemini   -> langchain-google-genai (needs GOOGLE_API_KEY)
    llm_provider: openai   -> langchain-openai      (needs OPENAI_API_KEY)
    llm_provider: ollama   -> langchain-ollama      (local, no key)

The client also:
  * retries when the provider says "rate limit" (HTTP 429) or has a hiccup,
  * reports token usage and cost to the Tracer,
  * can cache responses on disk (used by the evaluation so that re-running an
    experiment does not cost money again).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from rag_assistant.config import Settings
from rag_assistant.tracing import Tracer

logger = logging.getLogger(__name__)


def make_chat_model(provider: str, model: str, temperature: float = 0.0):
    """Create a LangChain chat model for the given provider.

    Imports happen inside each branch so you only need the package for the
    provider you actually use.
    """
    provider = provider.lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        extra = {}
        if model.startswith("openai/gpt-oss"):
            # Hidden reasoning tokens count against the free tier's daily token
            # limit; "low" cuts them sharply and is enough for these short tasks.
            extra["reasoning_effort"] = "low"
        return ChatGroq(model=model, temperature=temperature, max_retries=0, **extra)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Retries are handled by LLMClient (the library's own retries can hang
        # for minutes on a rate limit without telling us).
        return ChatGoogleGenerativeAI(model=model, temperature=temperature,
                                      max_retries=0, timeout=120)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, max_retries=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)
    raise ValueError(f"Unknown llm_provider '{provider}'. Use groq, gemini, openai or ollama.")


def _is_retryable(error: Exception) -> bool:
    """Rate limits and temporary server errors are worth retrying; bad keys are not."""
    text = f"{type(error).__name__} {error}".lower()
    # Whole-word matching: "rate" must not match inside e.g. "generateContent".
    permanent = r"\b(invalid api key|authentication|401|permission|404|not_found|400|invalid_argument)\b"
    if re.search(permanent, text):
        return False
    # A *daily* quota will not come back in a few seconds: fail fast.
    if "perday" in text or "per day" in text:
        return False
    temporary = (r"\b(rate limit|rate_limit|429|resource_exhausted|quota|500|502|503|504|"
                 r"unavailable|timeout|timed out|overloaded|connection|temporarily)\b")
    return bool(re.search(temporary, text))


class LLMClient:
    """Send a (system, user) prompt to the configured model and return the text."""

    def __init__(self, settings: Settings, provider: str | None = None, model: str | None = None,
                 chat_model=None, max_attempts: int = 6):
        self.settings = settings
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        # `chat_model` lets tests plug in a fake model (anything with .invoke()).
        self._chat_model = chat_model
        self.max_attempts = max_attempts
        self.cache_dir = settings.path(settings.cache_dir)

    @property
    def chat_model(self):
        if self._chat_model is None:   # created on first use
            self._chat_model = make_chat_model(self.provider, self.model, self.settings.llm_temperature)
        return self._chat_model

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """US$ cost of one call, using the per-1M-token prices in config.yaml."""
        price_in, price_out = self.settings.prices.get(self.model, (0.0, 0.0))
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    def complete(self, system: str, user: str, tracer: Tracer | None = None) -> str:
        """Run one chat completion and return the reply text."""
        cache_file = self._cache_file(system, user)
        if cache_file and cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if tracer:   # count cached tokens/cost as if paid, so experiments stay comparable
                tracer.record_llm_call(cached["input_tokens"], cached["output_tokens"],
                                       self.cost(cached["input_tokens"], cached["output_tokens"]))
            return cached["text"]

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        response = self._invoke_with_retries(messages)
        # Some providers (e.g. Gemini) return a list of content blocks instead of a
        # string; `.text` joins the text blocks (and skips "thinking" blocks).
        text = response.content if isinstance(response.content, str) else response.text

        # Token usage: LangChain puts it in `usage_metadata`. If a provider does
        # not report it, estimate ~4 characters per token.
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens") or (len(system) + len(user)) // 4
        output_tokens = usage.get("output_tokens") or len(text) // 4
        if tracer:
            tracer.record_llm_call(input_tokens, output_tokens, self.cost(input_tokens, output_tokens))

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"text": text, "input_tokens": input_tokens,
                                              "output_tokens": output_tokens}))
        return text

    def _invoke_with_retries(self, messages):
        delay = 2.0
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.chat_model.invoke(messages)
            except Exception as error:
                if attempt == self.max_attempts or not _is_retryable(error):
                    raise
                wait = delay + random.uniform(0, 1)
                logger.warning("LLM call failed (%s). Retry %d/%d in %.0fs",
                               str(error)[:120], attempt, self.max_attempts - 1, wait)
                time.sleep(wait)
                delay = min(delay * 2, 60)

    def _cache_file(self, system: str, user: str) -> Path | None:
        if not self.settings.use_llm_cache:
            return None
        key = hashlib.sha256(json.dumps([self.provider, self.model, system, user]).encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"
