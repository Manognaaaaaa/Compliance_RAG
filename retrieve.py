"""Hybrid (BM25 + semantic) retrieval with cross-encoder reranking.

Requires the index built by ingest.py (index/chunks.json + index/embeddings.npy).

Deliberately free of LangChain/Chroma/torch so the serving path fits in a
Vercel Function: semantic search is a dot product over a NumPy matrix, BM25
is rank_bm25 directly, and both models run as ONNX via fastembed. The
behaviour mirrors the previous LangChain EnsembleRetriever setup (weighted
reciprocal rank fusion, whitespace-tokenized BM25).
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

from config import (
    CHUNKS_PATH,
    CROSS_ENCODER_MODEL,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    ENSEMBLE_WEIGHTS,
    RERANK_TOP_N,
    RETRIEVER_K,
)
from generate import complete

# Reciprocal rank fusion constant - same default LangChain's EnsembleRetriever used.
RRF_C = 60

QUERY_REWRITE_PROMPT = """You are helping search a regulatory rulebook. Rewrite the following \
user question into 1-2 alternative phrasings using formal regulatory/compliance terminology, \
keeping the original question's intent. Return ONLY the rewritten query text, nothing else.

Original question: {query}
"""


@dataclass
class Chunk:
    """Same shape as a LangChain Document (page_content + metadata), so callers
    that read doc.page_content / doc.metadata keep working."""

    page_content: str
    metadata: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def get_embedder():
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_cross_encoder():
    return TextCrossEncoder(model_name=CROSS_ENCODER_MODEL)


def load_all_chunks(path=CHUNKS_PATH):
    """Load the chunk corpus persisted by ingest.py."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [
        Chunk(r["text"], {"source_doc": r["source_doc"], "page": r["page"]})
        for r in records
    ]


def load_embeddings(path=EMBEDDINGS_PATH):
    """Load the L2-normalized chunk embedding matrix (row i = chunk i)."""
    return np.load(path)


def rewrite_query(query):
    """Rewrite a natural-language question into the regulator's likely vocabulary
    before retrieval, so wording mismatches (e.g. "filing" vs "submission") don't
    tank BM25/semantic recall. One LLM call."""
    rewritten = complete(QUERY_REWRITE_PROMPT.format(query=query)).strip()
    print(f"Rewritten query: {rewritten}")
    return rewritten


class HybridRetriever:
    """Combine BM25 keyword search with semantic search. Build once and reuse."""

    def __init__(self, chunks, embeddings, k=RETRIEVER_K, weights=ENSEMBLE_WEIGHTS):
        self.chunks = chunks
        self.embeddings = embeddings
        self.k = k
        self.weights = weights
        # Whitespace split, no lowercasing - matches LangChain BM25Retriever's default.
        self.bm25 = BM25Okapi([c.page_content.split() for c in chunks])

    def bm25_search(self, query):
        scores = self.bm25.get_scores(query.split())
        return np.argsort(scores)[::-1][: self.k].tolist()

    def semantic_search(self, query):
        q = np.array(next(iter(get_embedder().embed([query]))), dtype=np.float32)
        q /= np.linalg.norm(q)
        scores = self.embeddings @ q
        return np.argsort(scores)[::-1][: self.k].tolist()

    def invoke(self, query):
        """Weighted reciprocal rank fusion of both result lists, deduped by text."""
        fused = {}
        for weight, ranked in zip(self.weights, (self.bm25_search(query), self.semantic_search(query))):
            for rank, idx in enumerate(ranked, start=1):
                key = self.chunks[idx].page_content
                score, _ = fused.get(key, (0.0, idx))
                fused[key] = (score + weight / (rank + RRF_C), idx)
        ordered = sorted(fused.values(), key=lambda x: x[0], reverse=True)
        return [self.chunks[idx] for _, idx in ordered]


def build_hybrid_retriever(chunks=None, embeddings=None):
    chunks = chunks if chunks is not None else load_all_chunks()
    embeddings = embeddings if embeddings is not None else load_embeddings()
    return HybridRetriever(chunks, embeddings)


def rerank(query, candidates, top_n=RERANK_TOP_N):
    """Rerank candidate chunks with a cross-encoder and return the top N documents."""
    scores = list(get_cross_encoder().rerank(query, [doc.page_content for doc in candidates]))

    reranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    for score, doc in reranked[:top_n]:
        print(
            f"Reranker score: {score:.4f} | "
            f"{doc.metadata.get('source_doc', 'Unknown Document')} | "
            f"Page {doc.metadata.get('page', 'N/A')}"
        )

    return [doc for _, doc in reranked[:top_n]]


def search(retriever, query, top_n=RERANK_TOP_N, rewritten_query=None):
    """Full retrieval pipeline: hybrid search -> rerank -> top N chunks.

    If rewritten_query is given, candidates are retrieved for both the
    original and rewritten phrasing and merged (deduped) before reranking,
    so a regulator-vocabulary rewrite can surface chunks the user's literal
    wording would miss. Reranking and the final answer still use the
    original query - only retrieval sees the rewrite.
    """
    candidates = retriever.invoke(query)

    if rewritten_query and rewritten_query != query:
        seen = {
            (doc.metadata.get("source_doc"), doc.metadata.get("page"), doc.page_content)
            for doc in candidates
        }
        for doc in retriever.invoke(rewritten_query):
            key = (doc.metadata.get("source_doc"), doc.metadata.get("page"), doc.page_content)
            if key not in seen:
                seen.add(key)
                candidates.append(doc)

    return rerank(query, candidates, top_n=top_n)
