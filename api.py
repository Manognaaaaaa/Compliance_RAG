"""FastAPI backend for the Compliance RAG assistant.

Serves the JSON API under /api and the static frontend (Frontend/) at /, so
one deployment - locally or on Vercel - is the whole app, same-origin.

The hybrid retriever (BM25 index + embedding matrix + ONNX models) is built
lazily on the first request and cached for the life of the process, rather
than in a startup hook: serverless platforms don't reliably run ASGI
lifespan events, and a warm instance reuses the cache across requests.

Run locally:
    uvicorn api:app --reload --port 8000     # then open http://localhost:8000
"""

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from generate import generate_answer
from retrieve import build_hybrid_retriever, rewrite_query, search

MAX_QUESTION_LENGTH = 2000


@lru_cache(maxsize=1)
def get_retriever():
    print("Loading index and retrievers...")
    retriever = build_hybrid_retriever()
    print("Ready.")
    return retriever


app = FastAPI()


def client_ip(request: Request):
    # Behind a hosting proxy (Vercel, Render) the socket peer is the proxy
    # itself, so every user would share one rate-limit bucket. The first
    # X-Forwarded-For entry is the original client.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


# Every /query hit triggers a paid Groq API call, so a public, unprotected
# endpoint is a real cost/abuse risk, not just a nice-to-have. Per-IP is
# enough for a portfolio project - not meant to withstand a determined
# attacker (trivially bypassed by rotating IPs), just casual abuse/loops.
# In-memory, so on serverless each warm instance counts separately.
limiter = Limiter(key_func=client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# The bundled frontend is same-origin and needs no CORS. This only matters
# if a frontend on another domain calls the API: set ALLOWED_ORIGINS to a
# comma-separated list of those origins. Defaults to any origin.
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
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

    rewritten = rewrite_query(question)
    chunks = search(get_retriever(), question, rewritten_query=rewritten)
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


# Mounted last so the /api routes above take precedence.
app.mount("/", StaticFiles(directory="Frontend", html=True), name="frontend")
