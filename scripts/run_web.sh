#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
APP="fhad.web.app:app"

echo "Starting web app on ${HOST}:${PORT}"
exec uvicorn "$APP" --host "$HOST" --port "$PORT"
