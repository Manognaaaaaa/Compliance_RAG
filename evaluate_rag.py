"""Evaluate the compliance RAG pipeline against eval_set.json using RAGAS.

Runs each question through the existing retrieval pipeline (hybrid BM25 +
semantic search -> cross-encoder rerank -> top 5 -> Groq generation) and
scores the results with RAGAS: faithfulness, answer_relevancy,
context_precision, context_recall.

Run standalone (after generate_eval_set.py has produced eval_set.json):
    python evaluate_rag.py
"""

import json
import sys
import types

# ragas 0.3.0 unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai, a module langchain-community
# removed in favor of the standalone langchain-google-vertexai package. This
# project uses Groq, not Vertex AI, so stub out the missing submodule rather
# than downgrade langchain-community (which ingest.py/retrieve.py depend on)
# or pull in an unused Vertex AI dependency.
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class _ChatVertexAIStub:
    pass


_vertexai_stub.ChatVertexAI = _ChatVertexAIStub
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_stub)

# ragas.executor unconditionally calls nest_asyncio.apply() at import time to
# support re-entrant event loops in notebooks. On Python 3.14, that global
# monkeypatch breaks asyncio.timeout()'s task-context tracking even in a
# plain script that never needs re-entrancy, causing every metric coroutine
# to fail with "Timeout should be used inside a task". We run from a script
# with no event loop already running, so re-entrancy is never needed here -
# neutralize it before ragas.executor imports and applies it.
import nest_asyncio  # noqa: E402

nest_asyncio.apply = lambda *args, **kwargs: None

from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig  # noqa: E402

from generate import build_prompt, get_llm  # noqa: E402
from retrieve import (  # noqa: E402
    build_hybrid_retriever,
    get_embeddings,
    load_all_chunks,
    load_vectorstore,
    rewrite_query,
    search,
)

EVAL_SET_PATH = "eval_set.json"
RESULTS_CSV_PATH = "eval_results.csv"

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# ChatGroq isn't an OpenAI client, so ragas can't apply its OpenAI-specific
# rate-limit handling to it - it falls back to generic tenacity retries
# (default: 10 attempts, exponential backoff up to 60s). faithfulness and
# context_recall each issue several sequential LLM calls per sample, and on
# a free-tier Groq key those calls can get rate-limited badly enough that
# every retry attempt fails too, exhausting the retry budget. Concurrency
# doesn't cause this (it still happens at max_workers=1) - it's the sustained
# rate limit itself. If you see NaN scores for these two metrics, you've hit
# your Groq rate/quota limit; wait a bit and rerun, or use a paid-tier key.
RUN_CONFIG = RunConfig(max_workers=2, max_retries=15, max_wait=90)


def run_pipeline(retriever, query):
    """Run the existing retrieval + generation pipeline for one question."""
    rewritten = rewrite_query(query, get_llm())
    top_chunks = search(retriever, query, rewritten_query=rewritten)
    prompt = build_prompt(query, top_chunks)
    response = get_llm().invoke(prompt)
    contexts = [doc.page_content for doc in top_chunks]
    return response.content, contexts


def main():
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    print("Loading vector store and retrievers...")
    vectorstore = load_vectorstore()
    all_chunks = load_all_chunks()
    retriever = build_hybrid_retriever(vectorstore, all_chunks)

    samples = []
    for i, item in enumerate(eval_set, 1):
        print(f"[{i}/{len(eval_set)}] {item['question']}")
        answer, contexts = run_pipeline(retriever, item["question"])
        samples.append(
            {
                "user_input": item["question"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
            }
        )

    dataset = EvaluationDataset.from_list(samples)

    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RUN_CONFIG,
    )

    print("\nAggregate scores:")
    print(result)

    df = result.to_pandas()
    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nPer-question breakdown written to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
