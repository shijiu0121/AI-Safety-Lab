#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export AEGIS_DEVICE=cuda
echo "Aegis Lab: http://127.0.0.1:8765"
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
