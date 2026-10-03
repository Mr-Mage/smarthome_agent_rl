#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_ROOT="${AGENT_ROOT:-/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT"
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
# Own dedicated process; do not attach to or reset another user's simulator.
(cd deps/SimuHome; exec "$PROJECT_ROOT/.venv-simuhome/bin/python" -m uvicorn src.simulator.api.app:app --host 127.0.0.1 --port 18780) > logs/adapter-check-simulator.log 2>&1 &
SIM_PID=$!
trap 'kill "$SIM_PID" 2>/dev/null || true; wait "$SIM_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
  kill -0 "$SIM_PID" 2>/dev/null || { echo 'Owned simulator failed to start'; exit 1; }
  if curl -sf --noproxy '*' http://127.0.0.1:18780/api/__health__ >/dev/null; then break; fi
  sleep 1
done
kill -0 "$SIM_PID"
"$AGENT_ROOT/../../anaconda3/envs/agent-lightning/bin/python" scripts/check_environment.py http://127.0.0.1:18780/api | tee logs/final-adapter-checks.json
