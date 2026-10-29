"""Shared test helpers: a fake LLM and a fake vector store.

They let us test the agent's *logic* (routing, retries, citations, refusals)
without network access, API keys or cost. Real-LLM tests live in
test_injection_live.py and are skipped when no working API key is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
from langchain_core.messages import AIMessage

from rag_assistant import prompts
from rag_assistant.agent import ResearchAssistant
from rag_assistant.config import Settings
from rag_assistant.llm import LLMClient
from rag_assistant.vector_store import Passage


class FakeChatModel:
    """Pretends to be a LangChain chat model. `responder(system, user)` decides the reply.

    Every call is recorded in `self.calls` so tests can inspect the prompts.
    """

    def __init__(self, responder: Callable[[str, str], str]):
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    def invoke(self, messages):
        system, user = messages[0].content, messages[1].content
        self.calls.append((system, user))
        reply = self.responder(system, user)
        return AIMessage(content=reply, usage_metadata={
            "input_tokens": 100, "output_tokens": 10, "total_tokens": 110})


def task_of(system: str) -> str:
    """Which agent step is calling? (based on the system prompt)"""
    if system == prompts.CONTEXTUALIZE_SYSTEM:
        return "contextualize"
    if system == prompts.GRADE_SYSTEM:
        return "grade"
    if system == prompts.REWRITE_SYSTEM:
        return "rewrite"
    if system == prompts.GENERATE_SYSTEM:
        return "generate"
    if system == prompts.GROUNDING_SYSTEM:
        return "grounding"
    raise AssertionError("unknown prompt")


def make_passage(title: str, section: str, text: str, n: int = 0, url: str = "") -> Passage:
    return Passage(chunk_id=f"{title}-{n}", text=text, title=title, section=section,
                   url=url or f"https://en.wikipedia.org/wiki/{title}", doc_id=title.lower(),
                   score=0.5)


@dataclass
class FakeStore:
    """Returns a fixed list of passages, whatever the query (records the queries)."""

    passages: list[Passage]

    def __post_init__(self):
        self.queries: list[str] = []

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        self.queries.append(query)
        # Return copies: the agent edits passage text when sanitizing.
        return [Passage(**vars(p)) for p in self.passages[: k or 5]]


MARS_PASSAGES = [
    make_passage("Mars", "Moons", "Mars has two small moons, Phobos and Deimos.", 1),
    make_passage("Jupiter", "Overview", "Jupiter is the largest planet.", 2),
]


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Default settings, but logs go to a temporary folder."""
    return Settings(log_dir=str(tmp_path / "logs"), max_retries=2)


def build_agent(settings: Settings, responder, passages=None) -> tuple[ResearchAssistant, FakeChatModel, FakeStore]:
    fake_model = FakeChatModel(responder)
    store = FakeStore(passages if passages is not None else MARS_PASSAGES)
    llm = LLMClient(settings, provider="fake", model="fake-model", chat_model=fake_model, max_attempts=1)
    return ResearchAssistant(settings, llm=llm, store=store), fake_model, store
