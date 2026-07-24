# Dockerfile — builds the whole app (FastAPI backend + built-in HTML GUI) as
# one container. Works on Hugging Face Spaces (Docker SDK), Render, Fly.io,
# Railway, or any host that runs a Dockerfile.

FROM python:3.11-slim

# Tesseract is a system program (not a pip package) -- needed for OCR.
# Skip this RUN line if you don't need OCR; the app degrades gracefully
# without it (OCR just reports "unavailable").
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects the container to listen on port 7860.
# (For Render/Railway/Fly.io, they inject a $PORT env var instead --
# this app reads that too, see backend/config.py's api_port setting.)
ENV PORT=7860
EXPOSE 7860

# data/ holds the local registry + query log -- see the note in README
# about this being ephemeral on most free hosts (wiped on redeploy).
RUN mkdir -p /app/data/logs

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
