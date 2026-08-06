# Compliance RAG

A Retrieval-Augmented Generation (RAG) assistant for querying DIFC/DFSA compliance and regulatory documents in plain English. Ask a question, and it retrieves the most relevant passages from the source PDFs and has an LLM answer using only that context — with citations back to the exact document and page.

## What it does

1. Loads a set of regulatory PDFs and splits them into overlapping text chunks.
2. Embeds each chunk and stores it in a local [ChromaDB](https://www.trychroma.com/) vector store.
3. On a query, retrieves relevant chunks using **hybrid search** — BM25 keyword search combined with semantic (embedding) search, merged via `EnsembleRetriever`.
4. Builds a prompt containing only the retrieved context and sends it to a Groq-hosted LLM.
5. Returns an answer that cites the source document name and page number for every point made.

## Documents included

| File | Description |
|---|---|
| `DIFC_Court_Rules.pdf` | DIFC Court Rules |
| `DFSA_Mkt_Rules_24-25.pdf` | DFSA Markets Rules |
| `DFSA_Gen_module.pdf` | DFSA General Module |
| `DFSA_Gen_Appendix_1.pdf` – `DFSA_Gen_Appendix_4.pdf` | DFSA GEN Appendices 1–4 |
| `UAE_Federal_Aml LAW.pdf` | UAE Federal AML Law |
| `DIFC_Data_Protection_Law.pdf` | DIFC Data Protection Law |

## Tech stack

- **LangChain** (`langchain`, `langchain-classic`, `langchain-community`) — orchestration
- **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) via `langchain-huggingface`
- **ChromaDB** via `langchain-chroma` — vector storage
- **BM25** (`rank_bm25`) — keyword retrieval, combined with semantic search via `EnsembleRetriever`
- **Groq** (`llama-3.1-8b-instant`) via `langchain-groq` — the LLM that generates answers
- **PyPDF** — PDF loading

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
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

**1. Build the index (one-time, or whenever the source PDFs change)**
```bash
python ingest.py
```
This loads all PDFs, chunks them, persists the chunk corpus to `difc_chunks.pkl` (used for BM25), and embeds + persists the vector store to `difc_chroma_db/`.

**2. Ask questions**
```bash
python main.py "What are the KYC requirements for a licensed firm under DIFC regulations?"
```
Or run `python main.py` with no arguments for an interactive prompt loop.

**3. Explore interactively (optional)**

Open `compliance_rag.ipynb` for step-by-step exploration of ingestion, retrieval, and generation — it imports the same functions as `main.py` rather than duplicating logic. It's a scratchpad for demos, not the entry point.

## Project structure

```
.
├── config.py             # Shared constants (models, paths, chunking/retrieval params)
├── ingest.py             # Load PDFs -> chunk -> embed -> persist to Chroma
├── retrieve.py           # Hybrid (BM25 + semantic) retrieval + cross-encoder reranking
├── generate.py           # Prompt construction + Groq LLM call
├── main.py               # Thin CLI wiring retrieve.py + generate.py together
├── compliance_rag.ipynb  # Exploration / demo notebook (not the entry point)
├── requirements.txt      # Python dependencies
├── .env                  # GROQ_API_KEY (not committed)
├── difc_chroma_db/       # Persisted vector store (not committed)
├── difc_chunks.pkl       # Persisted chunk corpus for BM25 (not committed)
└── *.pdf                 # Source regulatory documents
```
