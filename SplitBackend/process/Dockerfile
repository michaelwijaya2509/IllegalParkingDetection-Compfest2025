# backend/process/Dockerfile  (service: process-api)
FROM python:3.10-slim

# Lib dasar (umumnya cukup untuk google-cloud-*)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps dulu agar cache efektif
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# ENV umum
# Cloud Run akan isi $PORT. App kamu sudah baca PORT (default 5001), jadi ini aman.
ENV PYTHONUNBUFFERED=1 \
    PORT=5001

# Jalankan via gunicorn (cukup 1 worker + gthread utk concurrency ringan)
# timeout agak longgar karena route bisa memanggil infer-api.
CMD exec gunicorn app:app \
  --bind :$PORT \
  --workers 1 \
  --threads 8 \
  --worker-class gthread \
 