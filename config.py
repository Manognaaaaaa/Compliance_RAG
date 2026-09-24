"""Shared configuration for the Compliance RAG pipeline."""

# Source documents: filename -> display name used in citations
DOCUMENTS_MAP = {
    "DIFC_Court_Rules.pdf": "DIFC Court Rules",
    "DFSA_Mkt_Rules_24-25.pdf": "DFSA Markets Rules",
    "DFSA_Gen_module.pdf": "DFSA General Module",
    "DFSA_Gen_Appendix_1.pdf": "DFSA GEN Appendix 1",
    "DFSA_Gen_Appendix_2.pdf": "DFSA GEN Appendix 2",
    "DFSA_Gen_Appendix_3.pdf": "DFSA GEN Appendix 3",
    "DFSA_Gen_Appendix_4.pdf": "DFSA GEN Appendix 4",
    "UAE_Federal_Aml LAW.pdf": "UAE Federal AML Law",
    "DIFC_Data_Protection_Law.pdf": "DIFC Data Protection Law",
}

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ["\nArticle", "\nSection", "\n\n", "\n"]

# Prebuilt index (written by ingest.py, committed to git, read at serve time).
# Plain JSON + a NumPy matrix instead of Chroma: the serving path has to fit
# in a Vercel Function (500MB bundle), which rules out Chroma/torch. At this
# corpus size, exact brute-force cosine search over the matrix is a few ms.
INDEX_DIR = "./index"
CHUNKS_PATH = "./index/chunks.json"
EMBEDDINGS_PATH = "./index/embeddings.npy"

# Same models as before, run as ONNX via fastembed instead of torch via
# sentence-transformers.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Hybrid retrieval
# Bumped from 20: on a vocabulary-mismatch query ("filing" vs the DFSA's
# "submit"/"notify"), the genuinely relevant chunk (DFSA GEN Appendix 3, page
# 13) ranked outside the top 20 pre-rerank candidates and was invisible to
# the reranker. Confirmed via direct comparison: top reranker score went
# from 2.52 (K=20) to 5.04 (K=40) with two additional relevant chunks
# entering the pool. Costs some latency (more candidates to cross-encode).
RETRIEVER_K = 40
# Favor semantic over BM25: BM25 contributes noise (not just "no signal") when
# the user's wording doesn't literally appear in the source document (e.g.
# "filing" vs the regulator's actual "submit"/"notify"), so a straight 50/50
# split lets that noise compete with genuinely relevant semantic matches.
ENSEMBLE_WEIGHTS = [0.3, 0.7]

# Reranking
# fastembed's ONNX export of cross-encoder/ms-marco-MiniLM-L-6-v2
CROSS_ENCODER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 5

# Generation
# llama-3.1-8b-instant was deprecated by Groq (404s as of 2026-08-17); switched
# to the closest available small/fast model.
GROQ_MODEL = "openai/gpt-oss-20b"
