"""Streamlit chat UI. It talks to the FastAPI backend over HTTP.

Start the backend first, then:
    streamlit run app/streamlit_app.py

Set API_URL if the backend is not on http://localhost:8000.
"""

import os
import uuid

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Solar System Research Assistant", page_icon="🪐")
st.title("🪐 Solar System Research Assistant")
st.caption("Answers come only from ~40 Wikipedia articles, with citations. "
           "If the documents don't say, the assistant says so.")

# One session id per browser tab = one conversation memory on the server.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
    st.session_state.messages = []   # list of {"role", "content", "result"}


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


with st.sidebar:
    st.write(f"Session: `{st.session_state.session_id}`")
    if st.button("New conversation"):
        requests.delete(f"{API_URL}/sessions/{st.session_state.session_id}", timeout=10)
        st.session_state.messages = []
        st.rerun()
    try:
        health = requests.get(f"{API_URL}/health", timeout=30).json()
        st.success(f"Backend OK · {health['chunks_indexed']} chunks · {health['llm']}")
    except Exception:
        st.error(f"Backend not reachable at {API_URL}. Start it with "
                 "`uvicorn rag_assistant.api:app`.")
    st.markdown("**Try:**\n- How many moons does Mars have?\n- What are its names?\n"
                "- Who discovered the Great Red Spot?\n- What is the capital of France?")

# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("result"):
            show_details(message["result"])

if question := st.chat_input("Ask about planets, moons, comets or missions…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents…"):
            try:
                response = requests.post(f"{API_URL}/ask", timeout=300, json={
                    "question": question, "session_id": st.session_state.session_id})
                response.raise_for_status()
                result = response.json()
            except requests.HTTPError:
                st.error(response.json().get("detail", "Request failed"))
                st.stop()
            except requests.RequestException as error:
                st.error(f"Could not reach the backend: {error}")
                st.stop()
        st.markdown(result["answer"])
        show_details(result)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})
