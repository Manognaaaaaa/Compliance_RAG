# Compliance RAG

A Retrieval-Augmented Generation (RAG) assistant for querying DIFC/DFSA compliance and regulatory documents in plain English. Ask a question — via the CLI, the API, or the web UI — and it retrieves the most relevant passages from the source PDFs and has an LLM answer using only that context, with citations back to the exact document and page.

## Contents

- [How it works](#how-it-works)
- [Documents included](#documents-included)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Deploy to Vercel](#deploy-to-vercel)
- [API reference](#api-reference)
- [Evaluation](#evaluation)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## How it works

1. Loads a set of regulatory PDFs and splits them into overlapping text chunks.
2. Embeds each chunk and saves the chunks (`index/chunks.json`) and a normalized embedding matrix (`index/embeddings.npy`). This prebuilt index is committed, so the deployed app only loads it.
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
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, run as ONNX via [fastembed](https://github.com/qdrant/fastembed) (no torch) |
| Vector search | Exact cosine similarity over a NumPy matrix (5.4k chunks, a few ms) |
| Keyword retrieval | BM25 (`rank_bm25`), fused with semantic search via weighted reciprocal rank fusion |
| Reranking | Cross-encoder `ms-marco-MiniLM-L-6-v2`, ONNX via fastembed |
| LLM | [Groq](https://groq.com/) via the `groq` SDK — query rewriting and answer generation |
| PDF loading / chunking | PyPDF + LangChain text splitters (index build only, `requirements-dev.txt`) |
| API | FastAPI + Uvicorn (`api.py`), rate-limited with `slowapi`; also serves the frontend |
| Frontend | Vanilla HTML/CSS/JS (`Frontend/`) — no build step, no framework |
| Evaluation | [RAGAS](https://docs.ragas.io/) |
| Hosting | [Vercel](https://vercel.com/) (single Python function) — Docker also supported |

**Why no torch/Chroma?** The serving path has to fit in a Vercel Function (500MB bundle). `torch` alone is over that, so both models run as ONNX through `fastembed` (same models, same retrieval results — verified against the old torch/Chroma pipeline) and Chroma is replaced by a plain NumPy matrix. The runtime dependencies total ~170MB.

The LLM model is set in [`config.py`](config.py) (`GROQ_MODEL`) — Groq periodically retires models, so if you see a `model_not_found` error, that's the first thing to check against [Groq's current model list](https://console.groq.com/docs/models).

## Prerequisites

- **Python 3.11+** (Vercel runs 3.12).
- **A [Groq API key](https://console.groq.com/keys)** (free tier available) — required for query rewriting and answer generation.
- **A [Vercel](https://vercel.com/) account** (free Hobby plan) — only for deploying.
- **Docker** (optional) — only if you want to run the app containerized.

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements-dev.txt
```
`requirements.txt` is the light runtime set (what Vercel installs); `requirements-dev.txt` adds what's needed to rebuild the index and run the evaluation.

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
This loads all PDFs, chunks them, and writes `index/chunks.json` (chunk text + citation metadata, also used for BM25) and `index/embeddings.npy` (~10MB total). Commit the `index/` folder — the deployed app reads it and never ingests. The repo already includes a built index, so you only need this after changing the PDFs or chunking settings.

### 2. Ask questions via the CLI
```bash
python main.py "What are the filing requirements under the DFSA General Module?"
```
Or run `python main.py` with no arguments for an interactive prompt loop.

### 3. Run the API + web UI
```bash
uvicorn api:app --reload --port 8000
```
Open `http://localhost:8000`. The same FastAPI app serves the frontend at `/` and the API under `/api`, so there's no separate frontend server and no CORS setup. The index and models load on the first question (a few seconds), then stay cached.

### 4. Run in Docker (optional)

```bash
docker build -t compliance-rag .
docker run -d --name compliance-rag -p 8000:8000 --env-file .env compliance-rag
```
The image uses the committed `index/` (no ingestion step) and pre-downloads the ONNX models during the build. `GROQ_API_KEY` is injected at `docker run` time via `--env-file`, never baked into the image.

### 5. Evaluate answer quality (optional)
```bash
python generate_eval_set.py   # samples ~25 chunks across all documents, generates Q&A pairs -> eval_set.json
python evaluate_rag.py        # runs each question through the real pipeline, scores with RAGAS -> eval_results.csv
```
`faithfulness` and `context_recall` make several sequential LLM calls per question and can be rate-limited on a free-tier Groq key — see the comments in `evaluate_rag.py` if you see `NaN` scores.

## Deploy to Vercel

The whole app (frontend + API) deploys as one Vercel project on the free Hobby plan.

1. Push the repo to GitHub, including the `index/` folder.
2. On [vercel.com](https://vercel.com/new), **Add New → Project** and import the repo. Leave **Root Directory** as the repo root and the **Framework Preset** as detected (FastAPI) — no build command needed.
3. Under **Environment Variables**, add `GROQ_API_KEY`.
4. Deploy, then open the project URL.

How it's wired: `pyproject.toml` points Vercel at `api:app` (`[tool.vercel] entrypoint`), Vercel installs only the light runtime deps, and `vercel.json` excludes the PDFs and dev scripts from the function bundle. The ONNX models (~180MB) download from Hugging Face into `/tmp` on each cold start, so the first question after a period of inactivity takes noticeably longer (~10–20s); warm requests are just the Groq round-trips.

## API reference

| Endpoint | Method | Request body | Response |
|---|---|---|---|
| `/api/health` | `GET` | — | `{"status": "ok"}` |
| `/api/query` | `POST` | `{"question": string}` | `{"answer": string, "sources": [{"doc": string, "page": int}]}` |

Notes:
- `question` is capped at 2000 characters — longer requests get a `400`.
- `/api/query` is rate-limited to **10 requests/minute per IP** (via `slowapi`) — each call triggers a paid Groq API call, so this guards against runaway cost from an unprotected public endpoint. Exceeding it returns a `429`. The counter is in-memory, so on Vercel each warm instance counts separately — fine for casual abuse, not a hard guarantee.
- The bundled frontend is same-origin, so CORS doesn't matter for it. If a frontend on another domain calls the API, set `ALLOWED_ORIGINS` (comma-separated) — it defaults to allowing any origin.

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
├── ingest.py                # Load PDFs -> chunk -> embed -> write index/
├── retrieve.py              # Query rewrite + hybrid (BM25 + semantic) retrieval + cross-encoder reranking
├── generate.py              # Prompt construction + Groq LLM call
├── main.py                  # CLI wiring retrieve.py + generate.py together
├── api.py                   # FastAPI app: /api/health, /api/query, and the frontend at /
├── index/                   # Prebuilt index (committed): chunks.json + embeddings.npy
├── Frontend/                # Static HTML/CSS/JS web UI (served by api.py, no build step)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── config.js            # API_BASE_URL (relative "/api")
├── generate_eval_set.py     # Samples chunks, generates a RAGAS eval set -> eval_set.json
├── evaluate_rag.py          # Runs the real pipeline against eval_set.json, scores with RAGAS
├── requirements.txt         # Runtime deps (what Vercel installs)
├── requirements-dev.txt     # + index building and evaluation deps
├── pyproject.toml           # Vercel entrypoint (api:app) + runtime deps
├── vercel.json              # Function config: timeout, files excluded from the bundle
├── Dockerfile               # Optional container image
├── .env                     # GROQ_API_KEY (not committed)
├── eval_set.json            # Generated Q&A evaluation set
├── eval_results.csv         # Per-question RAGAS scores from the last eval run
└── *.pdf                    # Source regulatory documents
```

## Troubleshooting

- **`groq.NotFoundError: model_not_found`** — Groq has retired the model set in `GROQ_MODEL` (`config.py`). Pick a currently available one from [Groq's model list](https://console.groq.com/docs/models) and update `config.py`.
- **`NaN` scores from `evaluate_rag.py`** — usually Groq rate-limiting during the eval run, not a bug in the pipeline. See the comments at the top of `evaluate_rag.py`.
- **Frontend shows "Something went wrong"** — open the app through `uvicorn` (`http://localhost:8000`), not by double-clicking `index.html`. On Vercel, check the function logs and that `GROQ_API_KEY` is set.
- **`huggingface_hub` symlink warning on Windows** — harmless; it's a caching optimization Windows blocks without Developer Mode enabled. Everything still works, just with slightly more disk use for cached model files.
- **Slow first answer on Vercel** — a cold start downloads the two ONNX models into `/tmp`. Expected; later requests on the same instance are fast.
