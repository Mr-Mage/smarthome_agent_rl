#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$PROJECT_ROOT/scripts/run-single-h100.sh" \
  --config configs/p1-qwen35-9b-retrieval.json --suite configs/p1-dev-frozen.json \
  --harnesses upstream-react structured-schema "$@"
