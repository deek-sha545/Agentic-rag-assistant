# Design decisions

Each entry lists what was chosen, the alternatives, and why. Numbers come from
`results/` (end-to-end: `results/summary.md`; retrieval: `results/retrieval_chunk_sizes.md`).
The end-to-end runs used Gemini on the free tier. Its daily quota stopped `chunk_1000` after
20/44 questions, and `chunk_2000` and the injection-suite table were not run. Entries that
depend on those runs say so.

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
* **End-to-end check (partial):** on the 20 questions `chunk_1000` (1000 × 5) completed, it
  scored 100% vs 90% for the baseline on the *same* 20 questions. That is 2 questions: one is
  an ambiguous reference answer (Great Red Spot winds, see D19), and the other (Ceres) was
  rescued by the rewrite loop, which fires more often with top-5 retrieval. 20 questions
  cannot separate the settings, so the decision stays with the larger 36-question retrieval
  experiment: **500 × 8**. *Revisit when `chunk_1000`/`chunk_2000` finish.*

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

### D10. Rewrite only when *no* passage is relevant; `max_retries = 1`
* **Alternatives:** no loop; rewrite when fewer than N are relevant; 2 or more retries.
* **Evidence** (`baseline` vs `no_loop`, 44 questions each):

  | | Accuracy | Correct refusals | LLM calls / q | Tokens / q | Cost / 1k q |
  |---|---|---|---|---|---|
  | loop on (2 retries) | 94.4% | 100% | 3.34 | 2,482 | $0.85 |
  | loop off | 94.4% | 100% | 2.61 | 2,039 | $0.71 |

  With top-8 retrieval the loop **never fired for an answerable question**. All 16 rewrites went
  to the 8 unanswerable questions (2 each), which were refused anyway: +28% LLM calls for nothing.
  In `chunk_1000` (top 5) the loop **did rescue** one question. For "Who discovered Ceres?" the
  first search had nothing relevant; the rewrite "Ceres discovery date discoverer Giuseppe
  Piazzi" found the passage and the answer was correct. That rescue needed **one** rewrite.
* **Decision:** keep the loop (it is cheap insurance when retrieval misses, and the core
  agentic behaviour) but lower `max_retries` from 2 to **1**. The second rewrite never helped in
  any run and only cost 2 extra calls per unanswerable question. *Note:* the measured baseline
  used 2; for answerable questions the results are identical, since none used more than 1.
* **Caveat:** the eval questions are worded close to the articles. Harder, paraphrased
  questions would likely make the loop fire, and help, more often.

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

### D17. Default model `gemini-3.5-flash-lite`; compared with `gemini-3.1-flash-lite`
* **History:** the first plan was Groq (`llama-3.3-70b-versatile`), but the Groq key was rejected.
  On Gemini, `gemini-2.5-flash` is closed to new API keys, `gemini-3.8-flash` timed out, and
  `gemini-3.5-flash` worked but allows only **20 requests/day** on the free tier (a full experiment
  needs ~200). The lite models allow 500/day, so they were used.
* **Evidence** (44 questions each):

  | | Accuracy | False refusals | Hallucination | Cost / 1k q | Mean latency |
  |---|---|---|---|---|---|
  | `gemini-3.5-flash-lite` | 94.4% | 2.8% | 2.3% | $0.85 | 10.5 s |
  | `gemini-3.1-flash-lite` | 95.8% | 0.0% | 2.3% | $0.68 | 16.8 s |

* **Decision:** keep `gemini-3.5-flash-lite`. Quality is a tie: the difference is a single
  question, the Ceres refusal. It was faster in this run, though latency includes rate-limit
  waits. If cost matters more than speed, `gemini-3.1-flash-lite` is a valid switch (~20% cheaper,
  same quality on this eval set). Prices are from third-party trackers citing Google's price list;
  verify them in `config.yaml`.

### D18. Judge `gemini-3.5-flash-lite`, fixed across experiments
* **Ideal:** a stronger judge from a *different* model family, to avoid self-preference bias.
  The planned `openai/gpt-oss-120b` (Groq) was unavailable, and Gemini Pro models returned
  "quota exceeded" on the free tier.
* **Decision:** use the same model as the default agent, kept fixed for every experiment so they
  stay comparable. The judge sees the reference answer and returns correct / partially correct /
  incorrect. Refusals are scored by rule, without the judge. For unanswerable questions,
  correctly rejecting a false premise counts as correct.
* **Known weakness:** spot checks found it slightly lenient. It accepted "Ceres was discovered by
  Giuseppe Piazzi" without the date as correct (`model_b`). Treat accuracy as an upper bound.

## Evaluation design

### D19. 44 questions with evidence quotes; one factor changed at a time
* 36 answerable + 8 unanswerable (off-topic, unknowable, and false-premise traps). Each
  answerable question stores an exact quote from its article, and a test checks that the quote
  exists, which catches typos in the eval set.
* Experiments change one factor against the baseline, so differences can be attributed.
  The chunk-size comparison holds the context budget roughly constant (D4).
* Runs are resumable: finished questions are skipped, so an interruption (such as the daily
  quota) costs nothing. Experiments run **without** the response cache, because cached answers
  would make latency meaningless. `make_report.py` rebuilds every table from `results/raw/*.jsonl`
  and compares a partial run with the baseline on the same questions.
* **Eval-set flaw found by the experiments:** the reference for *"How fast can the Great Red
  Spot's winds get?"* only says 496 km/h, but the article also gives 432 km/h (at the spot's edge).
  All three complete runs answered 432 and were marked incorrect. It is kept unchanged for
  comparability with the finished runs; fix it before a fresh evaluation.
* **Latency caveat:** the free tier throttled the runs (128 retried calls), so latency, and
  especially p95, includes back-off waits.
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
