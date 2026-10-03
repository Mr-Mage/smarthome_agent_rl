"""Project B0 worker: valid failed tasks are rollout data, not service failures."""
import os
from pathlib import Path
import subprocess


class BaselineAgent:
    def run(self):
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root) + os.pathsep + str(root / "deps/SimuHome")
        # The child still uses the external ReAct loop; only task-failure exit
        # handling differs. Reward/event delivery failures still exit nonzero.
        subprocess.run([str(root / ".venv-baseline/bin/python"),
            str(root / "scripts/run_official_episode.py"), "--mode", "lightning",
            "--task-failure-is-data"], cwd=root, env=environment, check=True, timeout=570)
