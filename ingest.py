"""Load, chunk, and embed the compliance PDFs into the prebuilt index.

Run this module directly to (re)build the index from scratch:
    python ingest.py

This only needs to be run once (or whenever the source PDFs change). The
output in index/ is committed to git, so the deployed app never ingests -
it just loads the files. Needs requirements-dev.txt (PDF loading/splitting).
"""

import json
import os

import numpy as np
from fastembed import TextEmbedding
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    CHUNKS_PATH,
    DOCUMENTS_MAP,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    INDEX_DIR,
)


def load_and_chunk(documents_map=DOCUMENTS_MAP):
    """Load each source PDF and split it into overlapping, source-tagged chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )

    all_chunks = []
    for filename, doc_name in documents_map.items():
        print(f"Loading {doc_name}...")
        pages = PyPDFLoader(filename).load()
        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            chunk.metadata["source_doc"] = doc_name

        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks, path=CHUNKS_PATH):
    """Persist the chunk text + citation metadata (used for BM25 and for
    mapping semantic-search rows back to documents)."""
    records = [
        {
            "text": chunk.page_content,
            "source_doc": chunk.metadata["source_doc"],
            "page": chunk.metadata.get("page", -1),
        }
        for chunk in chunks
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def build_embeddings(chunks, path=EMBEDDINGS_PATH):
    """Embed every chunk and save the L2-normalized matrix, so a dot product
    at query time is cosine similarity. Row i matches chunk i in chunks.json."""
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    vectors = np.array(list(model.embed([c.page_content for c in chunks], batch_size=64)), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(path, vectors)
    print(f"Embedded {len(vectors)} chunks -> {path}")


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    chunks = load_and_chunk()
    save_chunks(chunks)
    build_embeddings(chunks)


if __name__ == "__main__":
    main()
