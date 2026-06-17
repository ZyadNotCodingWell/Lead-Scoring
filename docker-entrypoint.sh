#!/bin/sh
set -e

# Bootstrap data and model on the very first run.
# Subsequent starts skip this because the bind-mounted files already exist.
if [ ! -f "models/lead_scorer.pkl" ]; then
    echo "[entrypoint] No trained model found — running first-time bootstrap..."
    python data/simulate.py
    python models/train.py
    echo "[entrypoint] Bootstrap complete."
fi

exec "$@"
