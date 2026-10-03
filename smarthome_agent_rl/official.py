"""Lightning v1 local-runner entry point for the unchanged upstream agent."""
import os
from pathlib import Path
import subprocess


class OfficialBaselineAgent:
    def run(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) + os.pathsep + str(root / "deps/SimuHome")
        # Agent and simulator imports remain in upstream's supported Python 3.13.
        subprocess.run([str(root / ".venv-baseline/bin/python"),
                        str(root / "scripts/run_official_episode.py"), "--mode", "lightning"],
                       cwd=root, env=env, check=True, timeout=570)
