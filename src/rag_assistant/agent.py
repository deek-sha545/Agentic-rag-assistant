"""The agent: an explicit LangGraph state machine.

    contextualize -> retrieve -> grade --(relevant passages?)--> generate -> check_grounding -> finalize
                        ^                 |                         ^              |
                        |            none relevant                  +-- not grounded (1 retry)
                        +--- rewrite <----+ (up to max_retries)                    |
                                          |                                        v
                                     retries used up -----------------------> refuse

Each box is a "node" (a plain Python function that reads the state and returns
the fields it changes). The arrows with conditions are "conditional edges".

`ResearchAssistant.ask()` is the entry point used by the API, UI and evaluation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_assistant import prompts
from rag_assistant.config import Settings, load_settings
from rag_assistant.guardrails import remove_urls, sanitize, unexpected_urls
from rag_assistant.llm import LLMClient
from rag_assistant.memory import SessionMemory
from rag_assistant.tracing import Tracer
from rag_assistant.vector_store import Passage, VectorStore

logger = logging.getLogger(__name__)

MAX_GENERATION_ATTEMPTS = 2   # first answer + one retry after a failed grounding check


class AgentState(TypedDict, total=False):
    """Everything the nodes share. Each node returns a dict of fields to update."""

    question: str                     # what the user typed
    history: list[tuple[str, str]]    # previous (question, answer) pairs
    standalone_question: str          # question rewritten without pronouns
    query: str                        # current search query
    queries_tried: list[str]
    retries: int                      # how many times the query was rewritten
    passages: list[Passage]           # latest search results
    relevant: list[Passage]           # passages the grader kept
    answer: str
    generation_attempts: int
    feedback: str                     # why the previous answer was rejected
    grounded: bool | None
    refused: bool
    refusal_reason: str
    citations: list[dict]
    tracer: Tracer


@dataclass
class AgentResult:
    """What `ask()` returns."""

    question: str
    answer: str
    citations: list[dict]
    refused: bool
    refusal_reason: str = ""
    grounded: bool | None = None
    standalone_question: str = ""
    queries: list[str] = field(default_factory=list)
    retries: int = 0
    retrieved: list[dict] = field(default_factory=list)   # passages the answer was based on
    trace_id: str = ""
    usage: dict = field(default_factory=dict)              # tokens, cost, latency
    steps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- helpers

def parse_json_reply(text: str) -> dict | None:
    """LLMs sometimes wrap JSON in prose or ```code fences```. Pull out the {...} part."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def is_refusal(answer: str) -> bool:
    return prompts.REFUSAL.lower().rstrip(".") in answer.lower()


def extract_citation_numbers(answer: str) -> list[int]:
    """'... moons [1][3] and rings [2, 4].' -> [1, 3, 2, 4] (unique, in order)."""
    numbers: list[int] = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for n in group.split(","):
            if int(n) not in numbers:
                numbers.append(int(n))
    return numbers


def passage_dict(p: Passage) -> dict:
    return {"title": p.title, "section": p.section, "url": p.url, "doc_id": p.doc_id,
            "chunk_id": p.chunk_id, "score": p.score, "text": p.text}


# --------------------------------------------------------------------------- the agent

class ResearchAssistant:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None,
                 store: VectorStore | None = None, memory: SessionMemory | None = None):
        self.settings = settings or load_settings()
        self.llm = llm or LLMClient(self.settings)
        self.store = store or VectorStore(self.settings)
        self.memory = memory or SessionMemory(self.settings.memory_turns)
        self.graph = self._build_graph()

    # ----- graph wiring ---------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("contextualize", self.contextualize)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("grade", self.grade)
        graph.add_node("rewrite", self.rewrite)
        graph.add_node("generate", self.generate)
        graph.add_node("check_grounding", self.check_grounding)
        graph.add_node("refuse", self.refuse)
        graph.add_node("finalize", self.finalize)

        graph.add_edge(START, "contextualize")
        graph.add_edge("contextualize", "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", self.route_after_grade,
                                    {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
        graph.add_edge("rewrite", "retrieve")
        graph.add_conditional_edges("generate", self.route_after_generate,
                                    {"check_grounding": "check_grounding", "finalize": "finalize"})
        graph.add_conditional_edges("check_grounding", self.route_after_grounding,
                                    {"finalize": "finalize", "generate": "generate", "refuse": "refuse"})
        graph.add_edge("refuse", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    # ----- nodes ----------------------------------------------------------------
    def contextualize(self, state: AgentState) -> dict:
        """Use the conversation to turn 'And its moons?' into 'What are Mars's moons?'."""
        question, history = state["question"], state.get("history", [])
        with state["tracer"].step("contextualize") as step:
            if not history:   # nothing to resolve -> skip the LLM call
                standalone = question
            else:
                standalone = self.llm.complete(
                    prompts.CONTEXTUALIZE_SYSTEM,
                    prompts.CONTEXTUALIZE_USER.format(history=prompts.format_history(history),
                                                      question=question),
                    state["tracer"]).strip().strip('"') or question
            step.details["standalone_question"] = standalone
        return {"standalone_question": standalone, "query": standalone, "queries_tried": [],
                "retries": 0, "generation_attempts": 0, "feedback": ""}

    def retrieve(self, state: AgentState) -> dict:
        """Search the vector store; redact injection-like text (guardrail layer 1)."""
        with state["tracer"].step("retrieve", query=state["query"]) as step:
            passages = self.store.search(state["query"], self.settings.top_k)
            flagged = 0
            if self.settings.sanitize_injections:
                for p in passages:
                    clean, changed = sanitize(p.text)
                    if changed or p.suspicious:
                        flagged += 1
                    p.text = clean
            step.details.update(found=len(passages), redacted=flagged,
                                top=[p.header() for p in passages])
        return {"passages": passages, "queries_tried": state["queries_tried"] + [state["query"]]}

    def grade(self, state: AgentState) -> dict:
        """Ask the LLM which passages actually help answer the question."""
        passages = state["passages"]
        with state["tracer"].step("grade") as step:
            reply = self.llm.complete(
                prompts.GRADE_SYSTEM,
                prompts.GRADE_USER.format(question=state["standalone_question"],
                                          passages=prompts.format_passages(passages)),
                state["tracer"])
            parsed = parse_json_reply(reply)
            if parsed is None or not isinstance(parsed.get("relevant"), list):
                # Could not understand the grader: keep everything and let the
                # generator (which may refuse) decide. Safer than dropping all.
                logger.warning("Grader reply not understood: %r", reply[:200])
                relevant = list(passages)
            else:
                keep = {int(n) for n in parsed["relevant"] if str(n).isdigit()}
                relevant = [p for i, p in enumerate(passages, start=1) if i in keep]
            step.details.update(relevant=len(relevant), of=len(passages))
        return {"relevant": relevant}

    def rewrite(self, state: AgentState) -> dict:
        """Nothing relevant was found: ask the LLM for a different search query."""
        with state["tracer"].step("rewrite") as step:
            query = self.llm.complete(
                prompts.REWRITE_SYSTEM,
                prompts.REWRITE_USER.format(question=state["standalone_question"],
                                            tried="; ".join(state["queries_tried"])),
                state["tracer"]).strip().strip('"').splitlines()[0]
            step.details["new_query"] = query
        return {"query": query or state["standalone_question"], "retries": state["retries"] + 1}

    def generate(self, state: AgentState) -> dict:
        """Write the answer from the relevant passages, citing them as [n]."""
        attempt = state["generation_attempts"] + 1
        with state["tracer"].step("generate", attempt=attempt) as step:
            feedback = (prompts.REGENERATE_FEEDBACK.format(reason=state["feedback"])
                        if state.get("feedback") else "")
            answer = self.llm.complete(
                prompts.GENERATE_SYSTEM,
                prompts.GENERATE_USER.format(history=prompts.format_history(state.get("history", [])),
                                             passages=prompts.format_passages(state["relevant"]),
                                             question=state["standalone_question"],
                                             feedback=feedback),
                state["tracer"]).strip()
            refused = is_refusal(answer)
            step.details.update(refused=refused, chars=len(answer))
        return {"answer": answer, "refused": refused, "generation_attempts": attempt,
                "refusal_reason": "The passages found do not contain the answer." if refused else ""}

    def check_grounding(self, state: AgentState) -> dict:
        """Checks that the answer is supported: citations present, no injected
        links (guardrail layer 3), and an LLM fact-check against the passages."""
        answer, relevant = state["answer"], state["relevant"]
        with state["tracer"].step("check_grounding") as step:
            numbers = extract_citation_numbers(answer)
            valid = [n for n in numbers if 1 <= n <= len(relevant)]
            bad_urls = unexpected_urls(answer, {p.url for p in relevant})
            if not valid:
                grounded, reason = False, "the answer has no valid [n] citations"
            elif bad_urls:
                grounded, reason = False, "the answer contains links that are not sources"
            elif not self.settings.check_grounding:
                grounded, reason = True, ""
            else:
                reply = self.llm.complete(
                    prompts.GROUNDING_SYSTEM,
                    prompts.GROUNDING_USER.format(passages=prompts.format_passages(relevant),
                                                  question=state["standalone_question"],
                                                  answer=answer),
                    state["tracer"])
                parsed = parse_json_reply(reply) or {}
                # If the checker's reply is unreadable we do not block the answer.
                grounded = parsed.get("grounded", True) is not False
                reason = str(parsed.get("reason", "")) if not grounded else ""
            step.details.update(grounded=grounded, reason=reason)
        return {"grounded": grounded, "feedback": reason}

    def refuse(self, state: AgentState) -> dict:
        """Say honestly that the documents do not contain the answer."""
        with state["tracer"].step("refuse"):
            if not state.get("relevant"):
                reason = "No relevant passages were found, even after rewriting the query."
            else:
                reason = f"Could not write an answer supported by the documents ({state.get('feedback', '')})."
        return {"answer": prompts.REFUSAL, "refused": True, "refusal_reason": reason}

    def finalize(self, state: AgentState) -> dict:
        """Turn [n] markers into citations (document + section + link)."""
        with state["tracer"].step("finalize"):
            answer, relevant = state["answer"], state.get("relevant", [])
            citations = []
            if not state.get("refused"):
                # Last line of defence: never show a link the sources do not contain.
                answer = remove_urls(answer, unexpected_urls(answer, {p.url for p in relevant}))
                for n in extract_citation_numbers(answer):
                    if 1 <= n <= len(relevant):
                        p = relevant[n - 1]
                        citations.append({"marker": n, "title": p.title, "section": p.section,
                                          "url": p.url, "doc_id": p.doc_id, "chunk_id": p.chunk_id})
        return {"answer": answer, "citations": citations}

    # ----- routing (conditional edges) -----------------------------------------
    def route_after_grade(self, state: AgentState) -> str:
        if state["relevant"]:
            return "generate"
        if self.settings.use_rewrite_loop and state["retries"] < self.settings.max_retries:
            return "rewrite"
        return "refuse"

    def route_after_generate(self, state: AgentState) -> str:
        return "finalize" if state["refused"] else "check_grounding"

    def route_after_grounding(self, state: AgentState) -> str:
        if state["grounded"]:
            return "finalize"
        if state["generation_attempts"] < MAX_GENERATION_ATTEMPTS:
            return "generate"
        return "refuse"

    # ----- public entry point ---------------------------------------------------
    def ask(self, question: str, session_id: str = "default") -> AgentResult:
        tracer = Tracer(question, session_id)
        history = self.memory.history(session_id)
        final: dict[str, Any] = self.graph.invoke(
            {"question": question, "history": history, "tracer": tracer},
            config={"recursion_limit": 40},
        )
        self.memory.add(session_id, question, final["answer"])

        result = AgentResult(
            question=question,
            answer=final["answer"],
            citations=final.get("citations", []),
            refused=bool(final.get("refused")),
            refusal_reason=final.get("refusal_reason", ""),
            grounded=final.get("grounded"),
            standalone_question=final.get("standalone_question", question),
            queries=final.get("queries_tried", []),
            retries=final.get("retries", 0),
            retrieved=[passage_dict(p) for p in final.get("relevant", [])],
            trace_id=tracer.trace_id,
            usage=tracer.summary(),
            steps=[asdict(s) for s in tracer.steps],
        )
        tracer.write(self.settings.path(self.settings.log_dir), extra={
            "model": f"{self.llm.provider}/{self.llm.model}", "answer": result.answer,
            "refused": result.refused, "citations": result.citations})
        return result
