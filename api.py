"""FastAPI backend for the Compliance RAG assistant.

Loads the vector store, chunk corpus, and hybrid retriever once at startup
(embedding/reranking models and the BM25 index are expensive to build) and
reuses them across requests.

Run:
    uvicorn api:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from generate import generate_answer, get_llm
from retrieve import build_hybrid_retriever, load_all_chunks, load_vectorstore, rewrite_query, search

state = {}

MAX_QUESTION_LENGTH = 2000


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading vector store and retrievers...")
    vectorstore = load_vectorstore()
    all_chunks = load_all_chunks()
    state["retriever"] = build_hybrid_retriever(vectorstore, all_chunks)
    print("Ready.")
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)

# Every /query hit triggers a paid Groq API call, so a public, unprotected
# endpoint is a real cost/abuse risk, not just a nice-to-have. Per-IP is
# enough for a portfolio project - not meant to withstand a determined
# attacker (trivially bypassed by rotating IPs), just casual abuse/loops.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allow any origin for local development. Narrow this to the real frontend
# origin(s) before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    doc: str
    page: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
def query(request: Request, body: QueryRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question must be {MAX_QUESTION_LENGTH} characters or fewer.",
        )

    rewritten = rewrite_query(question, get_llm())
    chunks = search(state["retriever"], question, rewritten_query=rewritten)
    answer = generate_answer(question, chunks)

    sources = []
    seen = set()
    for chunk in chunks:
        doc = chunk.metadata.get("source_doc", "Unknown Document")
        page = chunk.metadata.get("page", -1)
        key = (doc, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"doc": doc, "page": page})

    return {"answer": answer, "sources": sources}
