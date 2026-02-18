#!/usr/bin/env bash
# Dev launcher — kills stale process on PORT before starting uvicorn.
# For production, use deploy/prot.service.
set -euo pipefail

PORT="${PORT:-8000}"

if ! command -v lsof &>/dev/null; then
    echo "Warning: lsof not found, skipping port check" >&2
else
    PIDS=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "Killing PID(s) on port $PORT: $PIDS"
        echo "$PIDS" | xargs kill

        # Poll until port is free (max 3s)
        for _ in $(seq 1 10); do
            lsof -ti :"$PORT" >/dev/null 2>&1 || break
            sleep 0.3
        done

        # SIGKILL fallback
        PIDS=$(lsof -ti :"$PORT" 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            echo "SIGKILL fallback: $PIDS"
            echo "$PIDS" | xargs kill -9
            sleep 0.5
        fi
    fi
fi

exec uv run uvicorn prot.app:app --port "$PORT" "$@"
