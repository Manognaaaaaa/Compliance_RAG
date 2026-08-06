"""Hybrid (BM25 + semantic) retrieval with cross-encoder reranking.

Requires the vector store built by ingest.py to already exist on disk.
"""

import pickle
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from config import (
    CHUNKS_PATH,
    CROSS_ENCODER_MODEL,
    EMBEDDING_MODEL,
    ENSEMBLE_WEIGHTS,
    PERSIST_DIRECTORY,
    RERANK_TOP_N,
    RETRIEVER_K,
)


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_MODEL)


def load_vectorstore(embeddings=None, persist_directory=PERSIST_DIRECTORY):
    """Load the persisted Chroma vector store (see ingest.py to build it)."""
    embeddings = embeddings or get_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def load_all_chunks(path=CHUNKS_PATH):
    """Load the chunk corpus persisted by ingest.py, for BM25."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_hybrid_retriever(vectorstore, all_chunks):
    """Combine BM25 keyword search with semantic search. Build once and reuse."""
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = RETRIEVER_K

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=ENSEMBLE_WEIGHTS,
    )


def rerank(query, candidates, top_n=RERANK_TOP_N):
    """Rerank candidate chunks with a cross-encoder and return the top N documents."""
    cross_encoder = get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = cross_encoder.predict(pairs)

    reranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    for score, doc in reranked[:top_n]:
        print(
            f"Reranker score: {score:.4f} | "
            f"{doc.metadata.get('source_doc', 'Unknown Document')} | "
            f"Page {doc.metadata.get('page', 'N/A')}"
        )

    return [doc for _, doc in reranked[:top_n]]


def search(retriever, query, top_n=RERANK_TOP_N):
    """Full retrieval pipeline: hybrid search -> rerank -> top N chunks."""
    candidates = retriever.invoke(query)
    return rerank(query, candidates, top_n=top_n)
