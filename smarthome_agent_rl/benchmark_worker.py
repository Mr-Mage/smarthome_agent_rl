"""Lightning delegates official evaluation to the preserved baseline interpreter."""
import os
from pathlib import Path
import subprocess


class BenchmarkAgent:
    def run(self):
        root = Path(__file__).resolve().parents[1]
        subprocess.run([str(root / '.venv-baseline/bin/python'),
            str(root / 'scripts/run_benchmark_episode.py'), '--mode', 'lightning'],
            cwd=root, env={**os.environ, 'PYTHONPATH': str(root) + os.pathsep + str(root / 'deps/SimuHome'),
                'PYTHONNOUSERSITE': '1'}, check=True, timeout=1750)
