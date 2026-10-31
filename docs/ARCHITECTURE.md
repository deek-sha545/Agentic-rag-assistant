# Architecture

This page explains every important file and how data moves through the system.
If you are new to the code, read the ⭐ **core files** first, in the order given.

## Read these first (core files)

| # | File | Why it matters |
|---|---|---|
| 1 ⭐ | [`config.yaml`](../config.yaml) | Every setting in one place: model, chunk size, retries, prices. |
| 2 ⭐ | [`src/rag_assistant/agent.py`](../src/rag_assistant/agent.py) | **The heart of the project.** The LangGraph graph: nodes, routing, citations, refusal. |
| 3 ⭐ | [`src/rag_assistant/prompts.py`](../src/rag_assistant/prompts.py) | All prompts, including the security rules. Read this alongside agent.py. |
| 4 ⭐ | [`src/rag_assistant/vector_store.py`](../src/rag_assistant/vector_store.py) | Embeddings + ChromaDB: how a question becomes a list of passages. |
| 5 ⭐ | [`src/rag_assistant/ingest.py`](../src/rag_assistant/ingest.py) | The ingestion pipeline in ~40 lines: load → chunk → scan → embed → store. |
| 6 ⭐ | [`src/rag_assistant/guardrails.py`](../src/rag_assistant/guardrails.py) | Prompt-injection detection, redaction and output link checks. |
| 7 ⭐ | [`src/rag_assistant/llm.py`](../src/rag_assistant/llm.py) | Provider-agnostic LLM calls with retries, token/cost accounting and caching. |
| 8 ⭐ | [`src/rag_assistant/evaluation.py`](../src/rag_assistant/evaluation.py) | How answers are scored: LLM judge, citations, refusals, hallucinations. |

## All files

### Package `src/rag_assistant/`

| File | What it does |
|---|---|
| `config.py` | Loads `config.yaml` into a frozen `Settings` dataclass. Applies `RAG_*` environment overrides and loads API keys from `.env`. |
| `documents.py` | Reads Wikipedia JSON (`== Heading ==`) and Markdown (`## Heading`) into `Section` objects (doc id, title, section path, text, url). Drops "References"-type sections. |
| `chunking.py` | Packs sentences into ~`chunk_size`-character chunks with overlap, never crossing a section boundary. Chunk ids are deterministic hashes. |
| `vector_store.py` | Loads the sentence-transformers model once. Embeds `"Title - Section\ntext"` and stores the vectors in a ChromaDB collection per chunk size (`wiki_c500`, …). `search()` returns `Passage` objects with a cosine score. |
| `ingest.py` | CLI and function that builds a collection. It also flags chunks that look like injections (`metadata.suspicious`). |
| `guardrails.py` | `find_injections`, `sanitize` (sentence-level redaction, HTML comments removed) and `unexpected_urls`/`remove_urls` (output check). |
| `prompts.py` | Prompt templates for the five LLM tasks (contextualize, grade, rewrite, generate, grounding), the `REFUSAL` sentence, `SECURITY_RULES`, and `format_passages` (numbered, tag-escaped `<passage>` blocks). |
| `llm.py` | `make_chat_model()` picks the LangChain class for groq / gemini / openai / ollama. `LLMClient.complete()` handles retries with exponential backoff on 429/5xx, usage → `Tracer`, and an optional disk cache. |
| `tracing.py` | `Tracer` with a `step()` context manager that times each node and collects tokens and cost from the LLM calls inside it. It writes one JSON line per question to `logs/traces.jsonl`. |
| `memory.py` | `SessionMemory`: the last N (question, answer) pairs per session id, in RAM, thread-safe. |
| `agent.py` | `ResearchAssistant` builds and runs the graph. `ask()` returns an `AgentResult` (answer, citations, refusal, queries, passages, usage, steps). |
| `api.py` | FastAPI app: `GET /health`, `POST /ask`, `DELETE /sessions/{id}`. The agent is created once and injected with `Depends` (tests swap in a fake). |
| `evaluation.py` | Judge prompt, `score_question`, `summarize` (all metrics) and `run_experiment` (resumable, one JSONL line per question). |

### Scripts, app, data

| File | What it does |
|---|---|
| `scripts/download_data.py` | Downloads 41 Wikipedia articles via the MediaWiki API (plain text with headings), with retries for rate limits. |
| `scripts/ask.py` | Ask one question from the terminal and print the answer, citations and usage. |
| `scripts/eval_retrieval.py` | Retrieval-only experiment (no LLM). Chunk size × k → doc hit, evidence hit, MRR, poisoned@k, context size. |
| `scripts/run_experiments.py` | End-to-end experiments (baseline, chunk sizes, loop off, second model, injection suite) → `results/`. |
| `scripts/make_report.py` | Charts from the results CSVs → `results/charts/`. |
| `scripts/docker_start.sh` | Container entry point: ingests on first run, then starts uvicorn. |
| `app/streamlit_app.py` | Chat UI with citations plus an expandable trace (queries, steps, tokens, cost, passages). Runs the agent in-process by default (this is how it is hosted on Streamlit Community Cloud, building the index on first start); with `API_URL` set it calls the FastAPI backend instead. |
| `app/requirements.txt` | Slim dependency list for Streamlit Community Cloud (CPU-only PyTorch via direct wheel links). |
| `eval/questions.jsonl` | 44 evaluation questions with reference answers, gold documents and evidence quotes. |
| `data/poisoned/*.md` | Four prompt-injection test documents (see the README in that folder). |
| `data/raw/` *(git-ignored)* | The downloaded articles. |
| `chroma_db/` *(git-ignored)* | The persistent vector store. |
| `logs/traces.jsonl` *(git-ignored)* | One trace per question asked. |

### Tests `tests/`

| File | What it covers |
|---|---|
| `conftest.py` | `FakeChatModel` (scripted LLM that records prompts) and `FakeStore`, so the agent's logic can be tested offline. |
| `test_agent.py` | Happy path, the rewrite loop, max retries, loop disabled, the grounding retry, missing citations, memory, the trace. |
| `test_guardrails.py` | Detection (no false positives on the real articles), redaction, prompt hardening, the end-to-end "gullible model" attacks. |
| `test_injection_live.py` | The same attacks against the real LLM and vector store (marked `llm`; skipped without a working key). |
| `test_ingestion.py`, `test_config.py`, `test_api.py`, `test_evaluation.py`, `test_eval_set.py` | The remaining units, plus a check that every eval reference is backed by its article. |

## Data flow

### 1. Ingestion (run once)

```
Wikipedia API ──download_data.py──▶ data/raw/<article>.json  {title, url, revision_id, text}
data/poisoned/*.md ─┐
                    ▼
documents.load_directory()  ─▶ 980 Sections (972 real + 8 poisoned): (title, "Moons > Inner moons", text, url)
chunking.chunk_sections()   ─▶ ~4,500 Chunks of ≤500 chars, 75-char overlap
guardrails.is_suspicious()  ─▶ metadata.suspicious = True for 4 poisoned chunks
vector_store.add_chunks()   ─▶ embed "Title - Section\ntext" (384-dim, normalised)
                               ─▶ ChromaDB collection "wiki_c500" (cosine)
```

### 2. Answering a question

`ResearchAssistant.ask(question, session_id)` runs the graph below. Each node
receives the shared `AgentState` (a dict) and returns only the fields it changes.

```
AgentState: question, history, standalone_question, query, queries_tried, retries,
            passages, relevant, answer, generation_attempts, feedback, grounded,
            refused, refusal_reason, citations, tracer
```

| Node | Reads | Does | Writes |
|---|---|---|---|
| `contextualize` | question, history | If there is history, the LLM rewrites a follow-up ("its moons?") into a standalone question. Otherwise no LLM call. | standalone_question, query |
| `retrieve` | query | Vector search, top 8. Redacts injection-like sentences. | passages, queries_tried |
| `grade` | passages | One LLM call grades all passages at once → `{"relevant": [ids]}`. | relevant |
| *route* | relevant, retries | some relevant → `generate`; none and retries left → `rewrite`; else → `refuse` | |
| `rewrite` | queries_tried | The LLM writes a different query (synonyms, specific names). | query, retries+1 |
| `generate` | relevant, history, feedback | Answers only from the passages, `[n]` after every sentence, or the exact refusal sentence. | answer, refused |
| `check_grounding` | answer, relevant | (a) has valid `[n]`? (b) no non-source URLs? (c) LLM fact check → `{"grounded": bool}`. | grounded, feedback |
| *route* | grounded, attempts | grounded → `finalize`; first failure → `generate` again with feedback; second failure → `refuse` | |
| `refuse` | | Sets the standard refusal and a reason. | answer, refused |
| `finalize` | answer, relevant | Removes stray URLs and maps `[n]` → {title, section, url, chunk_id}. | citations |

After the graph finishes, `ask()` stores the Q/A in session memory, builds the
`AgentResult` and appends the trace to `logs/traces.jsonl`. LLM calls per
question are typically **3** (grade, generate, grounding), plus 1 for a
follow-up question and 2 per query rewrite.

### 3. Logging

Every node runs inside `tracer.step(name)`, which records latency and the
tokens/cost of the LLM calls inside it, and logs a line like:

```
[3f9a1c2b7d10] step=grade latency=412ms tokens=1650/9 cost=$0.00098 {'relevant': 3, 'of': 8}
```

The JSONL trace holds the same data per step plus totals, the model name, the answer and citations.

### 4. API and UI

```
Streamlit (8501) ──HTTP (if API_URL)──▶ FastAPI (8000) ──▶ ResearchAssistant (one shared instance)
Streamlit (standalone / Community Cloud) ─────in-process──▶ ResearchAssistant (st.cache_resource)
                                                ├─ VectorStore → chroma_db/
                                                ├─ LLMClient   → Groq / Gemini / OpenAI / Ollama
                                                └─ SessionMemory (RAM)
```
