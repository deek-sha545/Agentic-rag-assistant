"""Ask one question from the command line (no server needed).

    python scripts/ask.py "How many moons does Mars have?"
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_assistant.agent import ResearchAssistant  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python scripts/ask.py "your question"')
    logging.basicConfig(level=logging.INFO, format="%(message)s")   # shows each agent step
    result = ResearchAssistant().ask(" ".join(sys.argv[1:]))
    print("\n" + result.answer + "\n")
    for c in result.citations:
        print(f"  [{c['marker']}] {c['title']} › {c['section']}  {c['url']}")
    print("\nusage:", json.dumps(result.usage))
