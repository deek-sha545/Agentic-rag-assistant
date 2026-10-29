"""Record what the agent does for each question: steps, tokens, latency, cost.

Every node of the agent runs inside `tracer.step("name")`. LLM calls made
inside that block add their token counts and cost to the step. At the end the
whole trace is written as one JSON line to logs/traces.jsonl, so you can
inspect any past question.

Example trace summary:
    {"steps": 6, "llm_calls": 4, "input_tokens": 3120, "output_tokens": 210,
     "cost_usd": 0.0020, "latency_s": 2.84}
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("rag_assistant.trace")


@dataclass
class Step:
    name: str
    latency_ms: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    details: dict = field(default_factory=dict)   # e.g. {"query": ..., "relevant": 3}


class Tracer:
    """Collects the steps of one question."""

    def __init__(self, question: str, session_id: str = ""):
        self.trace_id = uuid.uuid4().hex[:12]
        self.question = question
        self.session_id = session_id
        self.started = time.perf_counter()
        self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.steps: list[Step] = []
        self._current: Step | None = None

    @contextmanager
    def step(self, name: str, **details):
        """Time a block of code and collect the LLM usage that happens inside it."""
        step = Step(name=name, details=dict(details))
        previous, self._current = self._current, step
        start = time.perf_counter()
        try:
            yield step
        finally:
            step.latency_ms = round((time.perf_counter() - start) * 1000, 1)
            self._current = previous
            self.steps.append(step)
            logger.info("[%s] step=%s latency=%.0fms tokens=%d/%d cost=$%.5f %s",
                        self.trace_id, name, step.latency_ms, step.input_tokens,
                        step.output_tokens, step.cost_usd, step.details)

    def record_llm_call(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        """Called by the LLM client after every call."""
        step = self._current or Step(name="(outside step)")
        step.llm_calls += 1
        step.input_tokens += input_tokens
        step.output_tokens += output_tokens
        step.cost_usd += cost_usd
        if self._current is None:
            self.steps.append(step)

    def summary(self) -> dict:
        return {
            "steps": len(self.steps),
            "llm_calls": sum(s.llm_calls for s in self.steps),
            "input_tokens": sum(s.input_tokens for s in self.steps),
            "output_tokens": sum(s.output_tokens for s in self.steps),
            "cost_usd": round(sum(s.cost_usd for s in self.steps), 6),
            "latency_s": round(time.perf_counter() - self.started, 3),
        }

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "question": self.question,
            "summary": self.summary(),
            "steps": [asdict(s) for s in self.steps],
        }

    def write(self, log_dir: Path, extra: dict | None = None) -> None:
        """Append this trace as one JSON line to <log_dir>/traces.jsonl."""
        log_dir.mkdir(parents=True, exist_ok=True)
        record = self.to_dict() | (extra or {})
        with open(log_dir / "traces.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
