#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_ROOT="${AGENT_ROOT:-/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent}"
UV="${UV_BIN:-$AGENT_ROOT/../../anaconda3/envs/agent-lightning/bin/uv}"
cd "$PROJECT_ROOT"
mkdir -p logs
exec > >(tee -a logs/setup-baseline.log) 2>&1
[[ "$(git -C deps/SimuHome rev-parse HEAD)" == 83d28837b69f0cbf1bc02ed5334cb8b561a9d54d ]]
[[ -z "$(git -C deps/SimuHome status --porcelain)" ]]
"$UV" export --project deps/SimuHome --frozen --no-emit-project --no-dev --format requirements-txt > requirements-baseline.lock.txt
if [[ ! -d .venv-baseline ]]; then
    "$UV" venv --python .python/cpython-3.13.14-linux-x86_64-gnu/bin/python3.13 --seed .venv-baseline
fi
PROXY_ARGS=()
if [[ -n "${BASELINE_PROXY_URL:-${HTTPS_PROXY:-}}" ]]; then
    PROXY_ARGS+=(--proxy "${BASELINE_PROXY_URL:-$HTTPS_PROXY}")
fi
if [[ -n "${BASELINE_WHEELHOUSE:-}" ]]; then
    .venv-baseline/bin/python -m pip --isolated install --no-deps --require-hashes \
        --no-index --find-links "$BASELINE_WHEELHOUSE" -r requirements-baseline.lock.txt
else
    .venv-baseline/bin/python -m pip --isolated install --no-deps --require-hashes \
        --index-url "${BASELINE_INDEX_URL:-https://pypi.org/simple}" "${PROXY_ARGS[@]}" -r requirements-baseline.lock.txt
fi
.venv-baseline/bin/python -m pip check
.venv-baseline/bin/python -m pip freeze > logs/baseline-freeze.txt
