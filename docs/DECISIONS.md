# Design decisions

Each entry lists what was chosen, the alternatives, and why. Numbers come from
`results/` where an experiment exists. Entries marked **(provisional)** depend on
the end-to-end LLM experiments, which could not be run at build time because
the Groq API key was rejected (`401 Invalid API Key`). Re-run
`python scripts/run_experiments.py` and revisit those entries.

---

## Data

### D1. Corpus: 41 English Wikipedia articles about the Solar System
* **Alternatives:** an open-source project's documentation, arXiv papers, a mixed-topic Wikipedia sample.
* **Why:** a single theme produces realistic *distractors*: many articles mention Jupiter, moons
  or missions, so retrieval has to be precise. The text is freely licensed (CC BY-SA 4.0), well
  structured into sections (good for "document › section" citations) and easy to write
  checkable questions for. 41 articles = 972 sections, ~1.8M characters: big enough to be
  non-trivial and small enough to embed in about a minute on a CPU.
* Downloaded through the MediaWiki API as plain text with `== Heading ==` markers
  (`explaintext` + `exsectionformat=wiki`), so HTML parsing isn't needed. URL and revision id
  are kept for attribution.

### D2. Poisoned documents ingested alongside real ones
* **Alternatives:** test injections only in unit tests.
* **Why:** an attack only matters if retrieval actually surfaces it. Each poisoned file is
  written to rank highly for a normal question (for "Why was Pluto reclassified…", the poisoned
  document is the #1 hit), so the live tests exercise the real path.

## Ingestion and retrieval

### D3. Section-aware, sentence-packing chunker (written by hand)
* **Alternatives:** LangChain `RecursiveCharacterTextSplitter`, fixed-size windows, semantic chunking.
* **Why:** chunks never cross a section boundary, so a citation's section name is always right.
  Packing whole sentences avoids cutting facts in half. It is ~40 readable lines with no extra
  dependency. Overlap = 15% of the chunk size, carried as whole trailing sentences.

### D4. Chunk size 500 characters, top-k = 8
* **Alternatives tested:** 500 / 1000 / 2000 characters × k = 3, 5, 8, 10 (`scripts/eval_retrieval.py`).
* **Evidence:** at roughly the same context budget (3–4k characters sent to the LLM):

  | Setting | Answer passage retrieved | Context |
  |---|---|---|
  | **500 × 8** | **75%** | 3.1k chars |
  | 1000 × 5 | 58% | 3.9k chars |
  | 2000 × 3 | 39% | 4.0k chars |

  The right *article* is found ~97% of the time in every setting. The difference is whether the
  *specific passage* with the answer is included. Small chunks let the top-k cover more distinct
  places in an article. Large chunks only win with much more context (2000 × 8 = 83% at
  10.6k chars, 3.4× the tokens of the grading and answering steps).
* **Decision:** 500 × 8. It is the best recall per token measured. The first default was 1000 × 5.
* **Risk / to confirm:** small chunks give the LLM less surrounding context. Every passage
  carries its "Title › Section" header to compensate. The `chunk_1000` and `chunk_2000`
  end-to-end experiments (same context budget) will show whether answer accuracy agrees with
  retrieval recall.

### D5. Embed `"Title - Section\ntext"`, store only the text
* **Alternatives:** embed the raw chunk.
* **Why:** a chunk from *Io › Volcanism* often never says "Io". Prefixing the header lets a
  question about Io find it. The stored text stays clean for the prompt, and the header is added
  there separately.

### D6. Embedding model `all-MiniLM-L6-v2`
* **Alternatives:** `bge-small-en-v1.5`, `e5-base`, OpenAI embeddings.
* **Why:** open source, 384 dimensions, runs fast on a laptop CPU (~1 minute for 4,500 chunks), and a
  well-known baseline. The model is one config line (`embedding_model`), so a stronger model can be
  swapped in after re-ingesting. *Not benchmarked here; a natural next experiment given the
  retrieval bottleneck (see the README's Limitations section).*

### D7. ChromaDB, one collection per chunk size
* **Alternatives:** FAISS, Qdrant, pgvector.
* **Why:** embedded (no server), persistent on disk, metadata stored with vectors, simple
  Python API. One collection per chunk size (`wiki_c500`, `wiki_c1000`, `wiki_c2000`) lets
  experiments switch chunk size without re-ingesting. Cosine space with normalised vectors.

## The agent

### D8. An explicit LangGraph state graph, not a free-form ReAct agent
* **Alternatives:** a ReAct/tool-calling agent that decides freely, or a plain retrieve-then-answer chain.
* **Why:** the required behaviour (grade → rewrite → answer → verify) is a known procedure. A
  fixed graph makes every path testable (each routing branch has a unit test), keeps the cost
  predictable (bounded LLM calls), and works with smaller models that are unreliable at tool calling.
  The LLM still makes the *decisions* (relevance, the new query, groundedness); the graph decides
  what happens next.

### D9. Grade all passages in one LLM call
* **Alternatives:** one call per passage (the classic "CRAG" pattern).
* **Why:** 8 passages would mean 8 calls and 8× the latency and rate-limit pressure. A single
  call returning `{"relevant": [ids]}` is cheaper. If the reply can't be parsed, all passages are
  kept and the generator (which may refuse) decides. That fails open on recall but closed
  on answers.

### D10. Rewrite only when *no* passage is relevant; `max_retries = 2` **(provisional)**
* **Alternatives:** rewrite when fewer than N are relevant; unlimited retries; 1 or 3 retries.
* **Why:** if anything relevant was found, answering is usually better than searching again.
  Two rewrites bound the worst case at 3 searches and ~7 LLM calls for an unanswerable question.
  The `no_loop` experiment measures what the loop buys (accuracy, refusals) against what it costs
  (latency, tokens). If it buys nothing on this corpus, turning it off is a valid outcome.

### D11. Citations via `[n]` markers, mapped to sources in code
* **Alternatives:** ask the LLM to write titles/URLs, or use structured (JSON) output.
* **Why:** the model only has to write small numbers, which even small models do reliably. Titles,
  sections and URLs are filled in by code from the passage metadata, so they can't be
  hallucinated. Out-of-range markers are ignored.

### D12. Refusal as one exact sentence, plus code-level refusals
* The generator must reply with exactly *"I could not find the answer in the documents."*
  when the passages don't contain the answer. The graph also refuses on its own when grading
  finds nothing after all retries, or when an answer fails grounding twice. A fixed sentence
  makes refusals easy to detect, both in the UI and in the evaluation.

### D13. Grounding check: citations + link check + LLM fact check, one regeneration
* **Alternatives:** no check; NLI model; claim-by-claim verification.
* **Why:** cheap checks first (no `[n]`, or an unknown URL, means fail without an LLM call),
  then one LLM call. On failure the generator gets one retry *with the reason as feedback*. A
  second failure becomes a refusal, which favours "no answer" over "wrong answer".
  An unparseable checker reply passes, to avoid needless refusals.

### D14. Conversation memory: last 3 turns in RAM + a "contextualize" step
* **Alternatives:** LangGraph checkpointer, Redis, sending full history to retrieval.
* **Why:** follow-ups like "what about its moons?" fail retrieval unless the pronoun is resolved
  first. The contextualize node rewrites the question into a standalone one, and it is skipped
  (no LLM call) when there is no history. RAM storage is enough for a single-process demo; the
  README lists it as a limitation.

## Guardrails

### D15. Three layers: redact → harden prompt → check output
* **Alternatives:** prompt hardening only; an LLM-based injection classifier; a dedicated model
  such as Llama Guard / Prompt Guard.
* **Why:** no single layer is reliable, so each one covers a different failure:
  1. *Regex redaction* is free and explainable. It catches all 4 test attacks with
     **0 false positives across all 972 real sections** (`test_no_false_positives_on_downloaded_articles`),
     but it can be rephrased around.
  2. *Prompt hardening* (`<passage>` tags, escaping, explicit rules) is the general defence.
  3. *Output checks* catch what gets through. Citations are added by code, so the LLM never
     needs to output a URL, which makes any URL in an answer suspicious. The tests show a model
     that obeys every injected instruction still can't get the phishing link into the answer,
     even with redaction disabled.
* Redaction works per sentence: one bad sentence doesn't hide the useful facts around it.
  A bug found during development (single line breaks split an attack sentence, so only half
  was redacted) led to treating only `. ! ?` and blank lines as sentence ends.

## LLM layer and models

### D16. LangChain chat-model classes behind a tiny `LLMClient`
* **Alternatives:** LiteLLM, each provider's SDK directly.
* **Why:** LangGraph already depends on `langchain-core`, and every provider has a maintained
  `langchain-*` package with the same `.invoke()` interface. Provider imports are lazy, so
  you only need the package you use. `LLMClient` adds what the project needs: retries on
  429/5xx (but not on a bad key), token and cost accounting, and a response cache.

### D17. Default model `llama-3.3-70b-versatile` on Groq; second model `llama-3.1-8b-instant` **(provisional)**
* **Why:** Groq was the requested default and is fast. The 70B model is the stronger
  general-purpose option there; the 8B model is ~10× cheaper per token, so the `model_b`
  experiment answers a practical question: is the expensive model needed for grounded,
  cited answers? Choose the 8B model if its accuracy and hallucination rate are within a
  few points of the 70B model.

### D18. Judge `openai/gpt-oss-120b`, fixed across experiments
* **Why:** a *different model family* from the agent's models reduces self-preference bias, and a
  fixed judge keeps experiments comparable. The judge sees the reference answer and returns
  correct / partially correct / incorrect. Refusals are scored by rule, without the judge.
  For unanswerable questions, correctly rejecting a false premise counts as correct.

## Evaluation design

### D19. 44 questions with evidence quotes; one factor changed at a time
* 36 answerable + 8 unanswerable (off-topic, unknowable, and false-premise traps). Each
  answerable question stores an exact quote from its article, and a test checks that the quote
  exists, which catches typos in the eval set.
* Experiments change one factor against the baseline, so differences can be attributed.
  The chunk-size comparison holds the context budget roughly constant (D4).
* Runs are resumable and LLM responses are cached on disk, so a rate-limit interruption costs
  nothing and re-running the report is free. Token counts and cost are recorded even for cached
  calls, so experiments remain comparable.
* **Retrieval-only metrics** (no LLM) were measured first. They are cheap, deterministic, and
  isolate the retrieval component.

## Engineering

### D20. Python 3.12 virtual environment created with `uv`
* The machine's Python is 3.14, which some ML packages (ChromaDB's dependencies, PyTorch) may not
  have wheels for yet. `uv venv --python 3.12` provides 3.12 inside `.venv` without touching the
  system. PyTorch comes from the CPU-only index (~200 MB instead of ~2.5 GB with CUDA).

### D21. Offline tests with a scripted fake LLM
* **Why:** the agent's value is in its control flow. A fake model that returns scripted replies
  (and records prompts) tests every branch deterministically, for free, with no key. Live-LLM tests
  are marked `llm` and skip themselves when the key or vector store is missing.

### D22. FastAPI backend + Streamlit frontend talking over HTTP
* **Alternatives:** Streamlit calling the agent in-process.
* **Why:** the agent (embedding model, vector store) loads once in the API and serves any client;
  the UI stays a thin presentation layer. The agent is injected with `Depends`, so API tests use the fake.
