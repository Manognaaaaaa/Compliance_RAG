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

Open `compliance_rag.ipynb` and run the cells **top to bottom**:

1. **Install/import cell** — installs `langchain-huggingface` / `langchain-chroma` and imports everything needed.
2. **Ingestion cell** — loads all PDFs, chunks them (500 chars, 50 overlap), and tags each chunk with its source document name.
3. **Embedding cell** — embeds all chunks and persists them to `./difc_chroma_db`. *Only needs to be run once* — skip it on future runs to avoid re-embedding.
4. **Query cell** — loads the existing vector store, sets up hybrid retrieval + the Groq LLM, and answers a sample question with cited sources.

To ask your own question, edit the `query` variable in the last cell and re-run it.

> **Note:** Cells 2–4 depend on variables defined in earlier cells (`embeddings`, `all_chunks`, `vectorstore`). If you restart the kernel, re-run from the top rather than jumping straight to the query cell.

## Project structure

```
.
├── compliance_rag.ipynb      # Main notebook — ingestion + retrieval + generation
├── requirements.txt          # Python dependencies
├── .env                      # GROQ_API_KEY (not committed)
├── difc_chroma_db/           # Persisted vector store (not committed)
└── *.pdf                     # Source regulatory documents
```
