# python:3.11-slim, not a newer version: we hit real pip wheel availability
# problems with Python 3.14 earlier in this project (a dependency needed
# Microsoft C++ Build Tools to compile from source because no prebuilt wheel
# existed for that Python version). 3.11 has the mature, broadly-available
# wheel ecosystem sentence-transformers/torch/chromadb need.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies before copying app code so Docker's layer cache skips
# the slow reinstall on every rebuild when only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build the vector store from the source PDFs during the image build, rather
# than copying a pre-built difc_chroma_db/ from local disk. GitHub hard-
# rejects files over 100MB (chroma.sqlite3 alone is ~192MB) and Render builds
# from the GitHub repo, not this machine - only the PDFs (small, committed)
# and the ingestion code can actually get there. This runs once per image
# build, not per request or container restart. Placed before the app-code
# COPY below so editing api.py etc. doesn't invalidate this cached layer and
# force re-ingesting from scratch.
COPY config.py ingest.py ./
COPY *.pdf ./
RUN python ingest.py

# Application code that actually serves requests.
COPY api.py retrieve.py generate.py ./

EXPOSE 8000

# Shell form so ${PORT:-8000} expands - Render (planned host) injects its own
# PORT at runtime; this avoids rebuilding the image just to fix a hardcoded port.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
