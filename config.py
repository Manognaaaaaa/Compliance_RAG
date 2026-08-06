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

# Vector store
PERSIST_DIRECTORY = "./difc_chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk corpus, persisted separately for BM25 (Chroma's SQLite backend chokes
# on pulling back tens of thousands of rows via a single .get() call)
CHUNKS_PATH = "./difc_chunks.pkl"

# Hybrid retrieval
RETRIEVER_K = 20
ENSEMBLE_WEIGHTS = [0.5, 0.5]

# Reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 5

# Generation
GROQ_MODEL = "llama-3.1-8b-instant"
