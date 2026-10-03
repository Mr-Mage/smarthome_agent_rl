#!/usr/bin/env bash
# Historical custom ReAct V0, retained only for reproducing the first smoke run.
set -euo pipefail
AGENT_ROOT="${AGENT_ROOT:-/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$AGENT_ROOT/activate-agent-lightning.sh"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$AGENT_ROOT/agent-lightning:${PYTHONPATH:-}"
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec python scripts/run_stack.py "$@"
