#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_ROOT="${AGENT_ROOT:-/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent}"
UV="$AGENT_ROOT/../../anaconda3/envs/agent-lightning/bin/uv"
UV="${UV_BIN:-$UV}"
PIN=83d28837b69f0cbf1bc02ed5334cb8b561a9d54d
cd "$PROJECT_ROOT"
mkdir -p deps logs
exec > >(tee -a logs/setup-simulator.log) 2>&1
if [[ ! -d deps/SimuHome ]]; then
  git clone https://github.com/holi-lab/SimuHome.git deps/SimuHome
  git -C deps/SimuHome checkout "$PIN"
fi
[[ "$(git -C deps/SimuHome rev-parse HEAD)" == "$PIN" ]] || { echo 'SimuHome commit differs; use a separate pinned checkout'; exit 1; }
[[ -z "$(git -C deps/SimuHome status --porcelain)" ]] || { echo 'SimuHome checkout has changes; preserve it and use a pristine checkout'; exit 1; }
"$UV" python install 3.13.14 --install-dir .python --no-bin
if [[ ! -d .venv-simuhome ]]; then
  "$UV" venv --python .python/cpython-3.13.14-linux-x86_64-gnu/bin/python3.13 --seed .venv-simuhome
fi
"$UV" pip install --python .venv-simuhome/bin/python -r requirements-simulator.lock.txt
.venv-simuhome/bin/python -m pip freeze > logs/simulator-freeze.txt
git -C deps/SimuHome rev-parse HEAD > logs/simulator-commit.txt
