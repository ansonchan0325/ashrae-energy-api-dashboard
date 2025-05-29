#!/usr/bin/env bash
set -e

echo "[$(date '+%Y-%m-%dT%H:%M:%S')] ▶️ Starting bootstrap…"

echo "[$(date)] 1) Downloading models…"
python download_models.py

echo "[$(date)] 2) Download complete. Current models folder contents:"
ls -lh models/

echo "[$(date)] 3) Launching Uvicorn on port ${PORT:-8080}…"
exec uvicorn app:app \
     --host 0.0.0.0 \
     --port "${PORT:-8080}"
