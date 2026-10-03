#!/usr/bin/env bash
set -euo pipefail
AGENT_ROOT="${AGENT_ROOT:-/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$AGENT_ROOT/activate-agent-lightning.sh"
cd "$PROJECT_ROOT"
exec python scripts/reproduce_official_cli.py "$@"
