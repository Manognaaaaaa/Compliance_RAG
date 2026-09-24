# Optional: the app deploys to Vercel without Docker. This image is for
# running the same backend on any container host (or locally).
FROM python:3.12-slim

WORKDIR /app

# Install dependencies before copying app code so Docker's layer cache skips
# the reinstall on every rebuild when only application code changes. These
# are the light runtime deps only (no torch) - see requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The index is prebuilt by ingest.py and committed, so there's no ingestion
# step here. Pre-download the ONNX models so the first request doesn't.
COPY config.py generate.py retrieve.py api.py ./
COPY index/ ./index/
COPY Frontend/ ./Frontend/
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN python -c "from retrieve import get_embedder, get_cross_encoder; get_embedder(); get_cross_encoder()"

EXPOSE 8000

# Shell form so ${PORT:-8000} expands - most hosts inject their own PORT.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
