"""Streamlit chat UI.

Two ways to run it:

* Standalone (default, used on Streamlit Community Cloud): the agent runs inside
  the Streamlit process. Nothing else to start:
      streamlit run app/streamlit_app.py
  On first start, if the vector store is empty, it is built from data/raw
  (about a minute).

* Against the FastAPI backend: set API_URL, e.g.
      API_URL=http://localhost:8000 streamlit run app/streamlit_app.py

The API key is read from .env locally, or from the app's Secrets on Streamlit
Cloud (add GROQ_API_KEY = "..." under Settings > Secrets).
"""

import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # works without `pip install -e .`

API_URL = os.getenv("API_URL", "").rstrip("/")

st.set_page_config(page_title="Solar System Research Assistant", page_icon="🪐")


def load_secrets_into_env() -> None:
    """Copy Streamlit secrets (e.g. GROQ_API_KEY) into environment variables."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:   # no secrets.toml: fine locally, .env is used instead
        pass


@st.cache_resource(show_spinner=False)
def get_assistant():
    """Create the agent once per server process; build the index if it is missing."""
    load_secrets_into_env()
    from rag_assistant.agent import ResearchAssistant
    from rag_assistant.config import load_settings
    from rag_assistant.ingest import ingest
    from rag_assistant.vector_store import VectorStore

    settings = load_settings()
    if VectorStore(settings).count() == 0:
        ingest(settings)
    return ResearchAssistant(settings)


def friendly_error(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    if "per day" in lowered or "tpd" in lowered:
        return ("The free daily token limit of the language model has been used up. "
                "Please try again later.")
    if "rate limit" in lowered or "429" in lowered:
        return "The language model is busy (rate limit). Please wait a minute and try again."
    if "api key" in lowered or "401" in lowered or "authentication" in lowered:
        return "The language model rejected the API key. Check GROQ_API_KEY in the app's secrets."
    return f"The language model call failed: {text[:300]}"


def ask(question: str) -> dict:
    """Return the agent's result as a dict (same shape in both modes)."""
    if API_URL:
        response = requests.post(f"{API_URL}/ask", timeout=300, json={
            "question": question, "session_id": st.session_state.session_id})
        if response.status_code >= 400:
            raise RuntimeError(response.json().get("detail", "Request failed"))
        return response.json()
    return get_assistant().ask(question, st.session_state.session_id).to_dict()


def reset_session() -> None:
    if API_URL:
        try:
            requests.delete(f"{API_URL}/sessions/{st.session_state.session_id}", timeout=10)
        except requests.RequestException:
            pass
    else:
        get_assistant().memory.clear(st.session_state.session_id)


def show_details(result: dict) -> None:
    """Citations, search queries and the step-by-step trace under an answer."""
    if result.get("citations"):
        st.markdown("**Sources**")
        for c in result["citations"]:
            st.markdown(f"[{c['marker']}] [{c['title']} › {c['section']}]({c['url']})")
    elif result.get("refused"):
        st.info(result.get("refusal_reason") or "Not found in the documents.")

    usage = result.get("usage", {})
    with st.expander(f"Agent trace · {usage.get('latency_s', 0):.1f}s · "
                     f"{usage.get('input_tokens', 0) + usage.get('output_tokens', 0)} tokens · "
                     f"${usage.get('cost_usd', 0):.4f}"):
        st.write("Search queries:", result.get("queries", []))
        steps = pd.DataFrame(result.get("steps", []))
        if not steps.empty:
            st.dataframe(steps[["name", "latency_ms", "llm_calls", "input_tokens",
                                "output_tokens", "cost_usd"]], hide_index=True)
        for p in result.get("retrieved", []):
            st.markdown(f"*{p['title']} › {p['section']}* (similarity {p['score']:.2f})")
            st.caption(p["text"][:400] + ("…" if len(p["text"]) > 400 else ""))


st.title("🪐 Solar System Research Assistant")
st.caption("Answers come only from ~40 Wikipedia articles, with citations. "
           "If the documents don't say, the assistant says so.")

# One session id per browser tab = one conversation memory.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
    st.session_state.messages = []   # list of {"role", "content", "result"}

with st.sidebar:
    st.write(f"Session: `{st.session_state.session_id}`")
    if API_URL:
        try:
            health = requests.get(f"{API_URL}/health", timeout=30).json()
            st.success(f"Backend OK · {health['chunks_indexed']} chunks · {health['llm']}")
        except Exception:
            st.error(f"Backend not reachable at {API_URL}. Start it with "
                     "`uvicorn rag_assistant.api:app`.")
    else:
        try:
            with st.spinner("Loading the documents and the embedding model "
                            "(first start can take a minute)…"):
                assistant = get_assistant()
            st.success(f"Ready · {assistant.store.count()} chunks · "
                       f"{assistant.llm.provider}/{assistant.llm.model}")
        except Exception as error:
            st.error(f"Could not start the assistant: {str(error)[:300]}")
            st.stop()
    if st.button("New conversation"):
        reset_session()
        st.session_state.messages = []
        st.rerun()
    st.markdown("**Try:**\n- How many moons does Mars have?\n- What are its names?\n"
                "- Who discovered the Great Red Spot?\n- What is the capital of France?")
    st.caption("[Source code](https://github.com/deek-sha545/Agentic-rag-assistant)")

# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("result"):
            show_details(message["result"])

if question := st.chat_input("Ask about planets, moons, comets or missions…", max_chars=2000):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents…"):
            try:
                result = ask(question)
            except Exception as error:
                st.session_state.messages.pop()   # let the user ask again
                st.error(friendly_error(error))
                st.stop()
        st.markdown(result["answer"])
        show_details(result)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})
