"""Simple conversation memory: the last few (question, answer) pairs per session.

Kept in RAM only, so it disappears when the server restarts. That is enough for
"what about its moons?"-style follow-up questions within one chat session.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque


class SessionMemory:
    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self._turns: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_turns))
        self._lock = threading.Lock()   # the API may serve several requests at once

    def history(self, session_id: str) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._turns.get(session_id, []))

    def add(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            self._turns[session_id].append((question, answer))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._turns.pop(session_id, None)
