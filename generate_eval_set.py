"""Generate a RAGAS evaluation set from the source documents.

Samples chunks spread across every document in DOCUMENTS_MAP, asks the LLM to
write a factual question each chunk answers plus a grounded answer, and
writes the result to eval_set.json for evaluate_rag.py to consume.

Run standalone:
    python generate_eval_set.py
"""

import json
import random

from generate import get_llm
from ingest import load_and_chunk

SAMPLE_SIZE = 25
OUTPUT_PATH = "eval_set.json"

GEN_PROMPT = """Here is a passage from a compliance document.
Write ONE specific factual question this passage directly answers, and write the answer using ONLY this passage.
Respond as JSON: {{"question": "...", "answer": "..."}}

Passage:
{passage}
"""


def sample_chunks(chunks, n=SAMPLE_SIZE, seed=None):
    """Sample chunks spread across all source documents, not just the first alphabetically."""
    by_doc_indices = {}
    for i, chunk in enumerate(chunks):
        by_doc_indices.setdefault(chunk.metadata.get("source_doc", "Unknown"), []).append(i)

    rng = random.Random(seed)
    docs = list(by_doc_indices.keys())
    per_doc = max(1, n // len(docs))

    selected = set()
    for doc in docs:
        pool = by_doc_indices[doc]
        selected.update(rng.sample(pool, min(per_doc, len(pool))))

    if len(selected) < n:
        remaining = [i for i in range(len(chunks)) if i not in selected]
        selected.update(rng.sample(remaining, min(n - len(selected), len(remaining))))

    indices = list(selected)
    rng.shuffle(indices)
    return [chunks[i] for i in indices[:n]]


def parse_llm_json(text):
    """Extract the first {...} JSON object from the LLM response (tolerates code fences)."""
    start = text.find("{")
    end = text.rfind("}")
    return json.loads(text[start : end + 1])


def generate_question_answer(llm, chunk):
    prompt = GEN_PROMPT.format(passage=chunk.page_content)
    response = llm.invoke(prompt)
    data = parse_llm_json(response.content)
    return data["question"], data["answer"]


def main():
    print("Loading and chunking source documents...")
    chunks = load_and_chunk()

    print(f"Sampling {SAMPLE_SIZE} chunks across all documents...")
    sampled = sample_chunks(chunks, SAMPLE_SIZE)

    llm = get_llm()
    eval_set = []
    for i, chunk in enumerate(sampled, 1):
        doc_name = chunk.metadata.get("source_doc", "Unknown")
        page = chunk.metadata.get("page", "N/A")
        print(f"[{i}/{len(sampled)}] Generating Q&A from {doc_name} | Page {page}...")
        try:
            question, answer = generate_question_answer(llm, chunk)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  -> skipped (could not parse LLM output: {e})")
            continue

        eval_set.append(
            {
                "question": question,
                "ground_truth": answer,
                "source_doc": doc_name,
                "page": page,
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)

    print(f"\nWrote {len(eval_set)} question/answer pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
