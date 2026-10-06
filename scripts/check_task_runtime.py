"""Reproducible offline runtime gates, with UTF-8 evidence and exit status."""
import argparse
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_frozen_baseline import verify

TEST_MODULES = ('tests.test_device_contract', 'tests.test_execution_runtime',
                'tests.test_task_runtime', 'tests.test_runtime_contract_adapter',
                'tests.test_scheduler_runtime', 'tests.test_runtime_harness', 'tests.test_runtime_simuhome',
                'tests.test_task_conflicts', 'tests.test_homebench_adapter', 'tests.test_public_benchmark_runner',
                'tests.test_instruction_serialization', 'tests.test_homebench_dialect')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    baseline = verify(ROOT / 'configs/1007-baseline.json')
    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    with (output / 'tests.log').open('w', encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {'baseline': baseline, 'tests': result.testsRun, 'passed': result.wasSuccessful(),
              'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
              'seconds': time.monotonic() - start, 'actor_calls': 0, 'judge_calls': 0,
              'gpu_seconds': 0, 'scope': 'unit implementation checks; not official benchmark benefit'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
