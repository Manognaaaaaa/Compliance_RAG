"""Load, chunk, and embed the compliance PDFs into the Chroma vector store.

Run this module directly to (re)build the vector store from scratch:
    python ingest.py

This only needs to be run once (or whenever the source PDFs change) -
retrieve.py reads the persisted store on every subsequent run.
"""

import pickle

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    CHUNKS_PATH,
    DOCUMENTS_MAP,
    EMBEDDING_MODEL,
    PERSIST_DIRECTORY,
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
    """Persist the chunk corpus so retrieve.py can build a BM25 index from it
    without pulling tens of thousands of rows back out of Chroma."""
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


def build_vectorstore(chunks, persist_directory=PERSIST_DIRECTORY):
    """Embed chunks and persist them to a Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print("All documents embedded and stored.")
    return vectorstore


def main():
    chunks = load_and_chunk()
    save_chunks(chunks)
    build_vectorstore(chunks)


if __name__ == "__main__":
    main()
