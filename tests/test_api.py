"""Tests for the FastAPI endpoints, with the agent replaced by the fake one."""

import pytest
from conftest import build_agent
from fastapi.testclient import TestClient
from test_agent import happy_responder

from rag_assistant.api import app, get_assistant


@pytest.fixture
def client(settings):
    agent, _, store = build_agent(settings, happy_responder)
    store.count = lambda: 2                       # FakeStore has no count()
    store.collection_name = "fake"
    app.dependency_overrides[get_assistant] = lambda: agent
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks_indexed"] == 2


def test_ask_returns_answer_citations_and_usage(client):
    body = client.post("/ask", json={"question": "How many moons does Mars have?"}).json()
    assert "Phobos" in body["answer"]
    assert body["citations"][0]["title"] == "Mars"
    assert body["usage"]["llm_calls"] == 3


def test_empty_question_is_rejected(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_reset_session(client):
    client.post("/ask", json={"question": "Tell me about Mars.", "session_id": "x"})
    assert client.delete("/sessions/x").json() == {"cleared": "x"}
