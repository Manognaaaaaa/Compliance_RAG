"""Thin CLI entry point wiring retrieval and generation together.

Usage:
    python main.py "What are the KYC requirements for a licensed firm?"
    python main.py                      # interactive prompt loop

Requires the index to already exist - run `python ingest.py` first.
"""

import sys

from generate import generate_answer
from retrieve import build_hybrid_retriever, rewrite_query, search


def answer(query, retriever):
    rewritten = rewrite_query(query)
    chunks = search(retriever, query, rewritten_query=rewritten)
    return generate_answer(query, chunks)


def main():
    print("Loading index and retrievers...")
    retriever = build_hybrid_retriever()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(answer(query, retriever))
        return

    print("Compliance RAG ready. Type a question (or 'exit' to quit).")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        print(answer(query, retriever))


if __name__ == "__main__":
    main()
