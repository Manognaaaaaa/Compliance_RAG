# Compliance RAG

A Retrieval-Augmented Generation (RAG) assistant for querying DIFC/DFSA compliance and regulatory documents in plain English. Ask a question — via the CLI, the API, or the web UI — and it retrieves the most relevant passages from the source PDFs and has an LLM answer using only that context, with citations back to the exact document and page.

## Contents

- [How it works](#how-it-works)
- [Documents included](#documents-included)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [API reference](#api-reference)
- [Evaluation](#evaluation)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## How it works

1. Loads a set of regulatory PDFs and splits them into overlapping text chunks.
2. Embeds each chunk and stores it in a local [ChromaDB](https://www.trychroma.com/) vector store.
3. On a query, first **rewrites the question into the regulator's likely vocabulary** with one LLM call (e.g. "filing" → "submission"/"notification") — the DFSA General Module never uses the word "filing" at all, so this step matters for real questions, not just an edge case.
4. Retrieves relevant chunks using **hybrid search** — BM25 keyword search combined with semantic (embedding) search, merged via `EnsembleRetriever`, run on both the original and rewritten query — then reranks the merged candidates with a cross-encoder.
5. Builds a prompt from the top chunks and sends it to a Groq-hosted LLM, which returns **one coherent answer** (not a bullet dump of every retrieved passage) with inline citations to document name and page.
6. Answer quality is measurable via a RAGAS evaluation harness that scores faithfulness, answer relevancy, context precision, and context recall against a sampled Q&A set.

## Documents included

| File | Description |
|---|---|
| `DIFC_Court_Rules.pdf` | DIFC Court Rules |
| `DFSA_Mkt_Rules_24-25.pdf` | DFSA Markets Rules |
| `DFSA_Gen_module.pdf` | DFSA General Module |
| `DFSA_Gen_Appendix_1.pdf` – `DFSA_Gen_Appendix_4.pdf` | DFSA GEN Appendices 1–4 |
| `UAE_Federal_Aml LAW.pdf` | UAE Federal AML Law |
| `DIFC_Data_Protection_Law.pdf` | DIFC Data Protection Law |

To point this at a different set of documents, edit `DOCUMENTS_MAP` in [`config.py`](config.py) and re-run `python ingest.py`.

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | [LangChain](https://www.langchain.com/) (`langchain`, `langchain-classic`, `langchain-community`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` |
| Vector store | [ChromaDB](https://www.trychroma.com/) via `langchain-chroma` |
| Keyword retrieval | BM25 (`rank_bm25`), combined with semantic search via `EnsembleRetriever` |
| Reranking | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| LLM | [Groq](https://groq.com/) via `langchain-groq` — query rewriting and answer generation |
| PDF loading | PyPDF |
| API | FastAPI + Uvicorn (`api.py`), rate-limited with `slowapi` |
| Frontend | Vanilla HTML/CSS/JS (`Frontend/`) — no build step, no framework |
| Evaluation | [RAGAS](https://docs.ragas.io/) |
| Packaging | Docker — see [Run the backend in Docker](#4-run-the-backend-in-docker-optional) |

The LLM model is set in [`config.py`](config.py) (`GROQ_MODEL`) — Groq periodically retires models, so if you see a `model_not_found` error, that's the first thing to check against [Groq's current model list](https://console.groq.com/docs/models).

## Prerequisites

- **Python 3.11+** (the Docker image pins exactly `3.11-slim` for reliable prebuilt-wheel availability of `torch`/`sentence-transformers`; local development works fine on newer versions too).
- **A [Groq API key](https://console.groq.com/keys)** (free tier available) — required for query rewriting and answer generation.
- **Docker** (optional) — only needed if you want to run the backend containerized instead of directly with `uvicorn`.

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### 1. Build the index (one-time, or whenever the source PDFs change)
```bash
python ingest.py
```
This loads all PDFs, chunks them, persists the chunk corpus to `difc_chunks.pkl` (used for BM25), and embeds + persists the vector store to `difc_chroma_db/`. Both are gitignored — nothing large ever needs to go through git.

### 2. Ask questions via the CLI
```bash
python main.py "What are the filing requirements under the DFSA General Module?"
```
Or run `python main.py` with no arguments for an interactive prompt loop.

### 3. Run the API + web UI

Backend:
```bash
uvicorn api:app --reload --port 8000
```
Loads the vector store and hybrid retriever once at startup (not per-request). Exposes `GET /health` and `POST /query` — see [API reference](#api-reference) below.

Frontend (separate terminal, must be served — don't open `index.html` via `file://`, `fetch()`/CORS behave differently):
```bash
cd Frontend
python -m http.server 5500
```
Open `http://localhost:5500`. `Frontend/config.js` sets `API_BASE_URL`; that's the only environment-specific value in the frontend — point it at a different backend by changing that one line.

### 4. Run the backend in Docker (optional)

```bash
docker build -t compliance-rag-api .
docker run -d --name compliance-rag-api -p 8000:8000 --env-file .env compliance-rag-api
```
The image builds the vector store itself during `docker build` (a `RUN python ingest.py` step embeds all PDFs into the image) rather than copying a pre-built store from disk — so the image is fully self-contained and there's nothing to prepare beforehand. Expect the build to take several minutes the first time (installing `torch`/`transformers`, then embedding every document).

Same `/health` and `/query` contract as running `uvicorn` directly — the frontend doesn't need to know which one it's talking to. `GROQ_API_KEY` is injected at `docker run` time via `--env-file`, never baked into the image. `docker logs compliance-rag-api` should show `Loading vector store and retrievers...` then `Ready.` on startup.

> **Note on hosting:** this app's dependencies (`torch` + `sentence-transformers` + a Chroma vector store) need meaningfully more than 512MB RAM to run — free tiers on hosts like Render will likely fail with an out-of-memory error. Budget for a host/tier with at least 1-2GB RAM if deploying publicly.

### 5. Evaluate answer quality (optional)
```bash
python generate_eval_set.py   # samples ~25 chunks across all documents, generates Q&A pairs -> eval_set.json
python evaluate_rag.py        # runs each question through the real pipeline, scores with RAGAS -> eval_results.csv
```
`faithfulness` and `context_recall` make several sequential LLM calls per question and can be rate-limited on a free-tier Groq key — see the comments in `evaluate_rag.py` if you see `NaN` scores.

### 6. Explore interactively (optional)

Open `compliance_rag.ipynb` for step-by-step exploration of ingestion, retrieval, and generation — it imports the same functions as `main.py` rather than duplicating logic. It's a scratchpad for demos, not the entry point.

## API reference

| Endpoint | Method | Request body | Response |
|---|---|---|---|
| `/health` | `GET` | — | `{"status": "ok"}` |
| `/query` | `POST` | `{"question": string}` | `{"answer": string, "sources": [{"doc": string, "page": int}]}` |

Notes:
- `question` is capped at 2000 characters — longer requests get a `400`.
- `/query` is rate-limited to **10 requests/minute per IP** (via `slowapi`) — each call triggers a paid Groq API call, so this guards against runaway cost from an unprotected public endpoint. Exceeding it returns a `429`.
- CORS currently allows all origins (`allow_origins=["*"]`) — narrow this in `api.py` before hosting the frontend on a fixed domain.

## Evaluation

Baseline RAGAS scores (24 sampled questions, before the query-rewrite/reranking-pool improvements described above):

| Metric | Score |
|---|---|
| Faithfulness | 0.81 |
| Answer relevancy | 0.86 |
| Context precision | 0.84 |
| Context recall | 0.71 |

`faithfulness` and `context_recall` had a few `NaN` rows (Groq rate limits during the run), so those two are averaged over fewer than 24 samples. Rerun `python evaluate_rag.py` for current numbers — this table reflects a specific point-in-time run, not the current code.

## Project structure

```
.
├── config.py                # Shared constants (models, paths, chunking/retrieval params)
├── ingest.py                # Load PDFs -> chunk -> embed -> persist to Chroma
├── retrieve.py              # Query rewrite + hybrid (BM25 + semantic) retrieval + cross-encoder reranking
├── generate.py              # Prompt construction + Groq LLM call
├── main.py                  # CLI wiring retrieve.py + generate.py together
├── api.py                   # FastAPI backend (GET /health, POST /query)
├── Frontend/                # Static HTML/CSS/JS web UI (served separately, no build step)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── config.js            # API_BASE_URL - the only environment-specific value
├── generate_eval_set.py     # Samples chunks, generates a RAGAS eval set -> eval_set.json
├── evaluate_rag.py          # Runs the real pipeline against eval_set.json, scores with RAGAS
├── Dockerfile                # Containerized backend - builds the vector store during `docker build`
├── .dockerignore             # Excludes notebook, eval scripts, .env, Frontend/, etc. (PDFs are NOT excluded - the image needs them to build the vector store)
├── compliance_rag.ipynb      # Exploration / demo notebook (not the entry point)
├── requirements.txt          # Python dependencies
├── .env                      # GROQ_API_KEY (not committed)
├── difc_chroma_db/           # Persisted vector store (not committed - built by ingest.py)
├── difc_chunks.pkl           # Persisted chunk corpus for BM25 (not committed - built by ingest.py)
├── eval_set.json             # Generated Q&A evaluation set
├── eval_results.csv          # Per-question RAGAS scores from the last eval run
└── *.pdf                     # Source regulatory documents
```

## Troubleshooting

- **`groq.NotFoundError: model_not_found`** — Groq has retired the model set in `GROQ_MODEL` (`config.py`). Pick a currently available one from [Groq's model list](https://console.groq.com/docs/models) and update `config.py`.
- **`NaN` scores from `evaluate_rag.py`** — usually Groq rate-limiting during the eval run, not a bug in the pipeline. See the comments at the top of `evaluate_rag.py`.
- **Frontend requests fail with a CORS or network error** — make sure the frontend is served over HTTP (`python -m http.server`), not opened directly as a `file://` URL, and that `Frontend/config.js`'s `API_BASE_URL` points at a running backend.
- **`huggingface_hub` symlink warning on Windows** — harmless; it's a caching optimization Windows blocks without Developer Mode enabled. Everything still works, just with slightly more disk use for cached model files.
- **Docker build runs out of memory on a hosting platform** — see the [note on hosting](#4-run-the-backend-in-docker-optional) above; this app needs more RAM than most free hosting tiers provide.
