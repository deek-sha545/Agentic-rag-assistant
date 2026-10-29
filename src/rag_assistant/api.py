"""FastAPI backend.

Start it with:
    uvicorn rag_assistant.api:app --reload

Endpoints:
    GET    /health                -> is the service ready? how many chunks?
    POST   /ask                   -> {"question": "...", "session_id": "..."}
    DELETE /sessions/{session_id} -> forget a conversation

Interactive docs are generated automatically at http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_assistant.agent import ResearchAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic RAG Research Assistant",
              description="Ask questions about the Solar System document collection.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          examples=["How many moons does Mars have?"])
    session_id: str = Field("default", max_length=100)


@lru_cache(maxsize=1)
def get_assistant() -> ResearchAssistant:
    """Create the agent once (loads the embedding model) and reuse it."""
    return ResearchAssistant()


@app.get("/health")
def health(assistant: ResearchAssistant = Depends(get_assistant)) -> dict:
    return {
        "status": "ok",
        "chunks_indexed": assistant.store.count(),
        "collection": assistant.store.collection_name,
        "llm": f"{assistant.llm.provider}/{assistant.llm.model}",
    }


@app.post("/ask")
def ask(request: AskRequest, assistant: ResearchAssistant = Depends(get_assistant)) -> dict:
    if assistant.store.count() == 0:
        raise HTTPException(503, "The vector store is empty. Run `python -m rag_assistant.ingest`.")
    try:
        return assistant.ask(request.question, request.session_id).to_dict()
    except Exception as error:   # e.g. invalid API key, provider down
        logger.exception("Agent failed")
        raise HTTPException(502, f"The language model call failed: {str(error)[:300]}") from error


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str, assistant: ResearchAssistant = Depends(get_assistant)) -> dict:
    assistant.memory.clear(session_id)
    return {"cleared": session_id}
