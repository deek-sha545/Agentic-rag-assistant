"""Tests for the agent's control flow, using the fake LLM from conftest.py."""

import json

from conftest import build_agent, task_of

from rag_assistant import prompts
from rag_assistant.agent import (extract_citation_numbers, is_refusal, normalize_citations,
                                 parse_json_reply)


def happy_responder(system, user):
    task = task_of(system)
    if task == "grade":
        return '{"relevant": [1]}'
    if task == "generate":
        return "Mars has two moons, Phobos and Deimos [1]."
    if task == "grounding":
        return '{"grounded": true}'
    if task == "contextualize":
        return "What are the moons of Mars?"
    return "unused"


# ---------------------------------------------------------------- happy path

def test_answer_has_citation_with_document_and_section(settings):
    agent, _, _ = build_agent(settings, happy_responder)
    result = agent.ask("How many moons does Mars have?")
    assert not result.refused
    assert "Phobos" in result.answer
    assert result.citations == [{
        "marker": 1, "title": "Mars", "section": "Moons",
        "url": "https://en.wikipedia.org/wiki/Mars", "doc_id": "mars", "chunk_id": "Mars-1"}]
    assert result.grounded is True
    assert result.retries == 0


def test_steps_tokens_and_cost_are_logged(settings, tmp_path):
    agent, _, _ = build_agent(settings, happy_responder)
    result = agent.ask("How many moons does Mars have?")
    step_names = [s["name"] for s in result.steps]
    assert step_names == ["contextualize", "retrieve", "grade", "generate", "check_grounding", "finalize"]
    # 3 LLM calls (grade, generate, grounding) x 100 input tokens each
    assert result.usage["llm_calls"] == 3
    assert result.usage["input_tokens"] == 300
    assert result.usage["latency_s"] >= 0
    # The trace was written as one JSON line.
    lines = (tmp_path / "logs" / "traces.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["trace_id"] == result.trace_id


# ---------------------------------------------------------------- retry loop

def test_rewrite_loop_searches_again_when_nothing_relevant(settings):
    grades = iter(['{"relevant": []}', '{"relevant": [1]}'])

    def responder(system, user):
        task = task_of(system)
        if task == "grade":
            return next(grades)
        if task == "rewrite":
            return "Phobos Deimos Martian satellites"
        return happy_responder(system, user)

    agent, _, store = build_agent(settings, responder)
    result = agent.ask("What orbits the red planet?")
    assert result.retries == 1
    assert store.queries == ["What orbits the red planet?", "Phobos Deimos Martian satellites"]
    assert not result.refused


def test_gives_up_after_max_retries_and_refuses(settings):
    def responder(system, user):
        task = task_of(system)
        if task == "grade":
            return '{"relevant": []}'
        if task == "rewrite":
            return f"another query {len(user)}"
        raise AssertionError(f"{task} should not be called")

    agent, _, store = build_agent(settings, responder)
    result = agent.ask("Who won the 2022 football world cup?")
    assert result.refused
    assert result.answer == prompts.REFUSAL
    assert result.citations == []
    assert result.retries == settings.max_retries
    assert len(store.queries) == settings.max_retries + 1   # first search + one per rewrite


def test_without_rewrite_loop_refuses_immediately(settings):
    settings = settings.with_overrides(use_rewrite_loop=False)

    def responder(system, user):
        assert task_of(system) == "grade"
        return '{"relevant": []}'

    agent, _, store = build_agent(settings, responder)
    result = agent.ask("Something not in the documents?")
    assert result.refused and result.retries == 0 and len(store.queries) == 1


# ---------------------------------------------------------------- grounding

def test_ungrounded_answer_is_regenerated_then_refused(settings):
    def responder(system, user):
        task = task_of(system)
        if task == "grade":
            return '{"relevant": [1]}'
        if task == "generate":
            return "Mars has 14 moons [1]."
        if task == "grounding":
            return '{"grounded": false, "reason": "14 moons is not supported"}'
        raise AssertionError(task)

    agent, model, _ = build_agent(settings, responder)
    result = agent.ask("How many moons does Mars have?")
    assert result.refused
    assert result.answer == prompts.REFUSAL
    generate_calls = [u for s, u in model.calls if task_of(s) == "generate"]
    assert len(generate_calls) == 2
    assert "14 moons is not supported" in generate_calls[1]   # feedback reaches the retry


def test_answer_without_citations_is_rejected(settings):
    answers = iter(["Mars has two moons.", "Mars has two moons [1]."])

    def responder(system, user):
        if task_of(system) == "generate":
            return next(answers)
        return happy_responder(system, user)

    agent, _, _ = build_agent(settings, responder)
    result = agent.ask("How many moons does Mars have?")
    assert not result.refused and result.citations[0]["title"] == "Mars"


def test_model_refusal_is_passed_through(settings):
    def responder(system, user):
        if task_of(system) == "generate":
            return prompts.REFUSAL
        return happy_responder(system, user)

    agent, _, _ = build_agent(settings, responder)
    result = agent.ask("What is the population of Mars?")
    assert result.refused and result.citations == []


# ---------------------------------------------------------------- memory

def test_follow_up_question_uses_session_memory(settings):
    agent, model, store = build_agent(settings, happy_responder)
    agent.ask("Tell me about Mars.", session_id="s1")
    result = agent.ask("What are its moons?", session_id="s1")
    assert result.standalone_question == "What are the moons of Mars?"
    assert store.queries[-1] == "What are the moons of Mars?"
    contextualize_prompt = [u for s, u in model.calls if task_of(s) == "contextualize"][0]
    assert "Tell me about Mars." in contextualize_prompt


def test_sessions_do_not_share_memory(settings):
    agent, model, _ = build_agent(settings, happy_responder)
    agent.ask("Tell me about Mars.", session_id="a")
    agent.ask("What are its moons?", session_id="b")
    assert not any(task_of(s) == "contextualize" for s, _ in model.calls)


# ---------------------------------------------------------------- helpers

def test_parse_json_reply_handles_code_fences_and_prose():
    assert parse_json_reply('Sure!\n```json\n{"relevant": [2, 3]}\n```') == {"relevant": [2, 3]}
    assert parse_json_reply("no json here") is None


def test_extract_citation_numbers():
    assert extract_citation_numbers("A [1][3]. B [2, 4]. C [1].") == [1, 3, 2, 4]


def test_full_width_citations_are_normalized():
    """gpt-oss on Groq writes 【2】 instead of [2]."""
    answer = normalize_citations("Io has nine volcanoes【2】 and plumes【1, 3】.")
    assert answer == "Io has nine volcanoes[2] and plumes[1, 3]."
    assert extract_citation_numbers(answer) == [2, 1, 3]
    # Its browsing-tool style: 【n†L1-L2】, sometimes with square brackets.
    answer = normalize_citations("Found by Herschel【1†L1-L2】【2†L3-L4】 in 1789 [3†source].")
    assert answer == "Found by Herschel[1][2] in 1789 [3]."


def test_is_refusal():
    assert is_refusal("I could not find the answer in the documents.")
    assert not is_refusal("Mars has two moons [1].")


def test_content_blocks_are_turned_into_text(settings):
    """Gemini returns [{'type': 'text', 'text': ...}] instead of a plain string."""
    from langchain_core.messages import AIMessage
    from rag_assistant.llm import LLMClient

    class BlockModel:
        def invoke(self, messages):
            return AIMessage(content=[{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}])

    llm = LLMClient(settings, provider="fake", model="fake", chat_model=BlockModel())
    assert llm.complete("system", "user") == "Hello there"
