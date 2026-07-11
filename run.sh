#!/usr/bin/env bash
# Launch the Breakout Board dashboard.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/streamlit run src/app.py "$@"
