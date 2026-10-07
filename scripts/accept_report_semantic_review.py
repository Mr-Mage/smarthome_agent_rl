"""Offline engineering acceptance in the existing SimuHome environment."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from scripts.replay_report_semantic_review import replay
from scripts.run_action_semantic_diagnosis import sha
from smarthome_agent_rl.benchmarks.runner import save
from tests.test_report_semantic_native import ReportSemanticNativeTests
from tests.test_report_semantic_review import ReportSemanticReviewTests


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True);args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report, records = replay(args.parent)
    save(args.output / 'replay.json', report);save(args.output / 'records.json', records)
    cases = [case for cls in (ReportSemanticReviewTests, ReportSemanticNativeTests)
             for case in unittest.defaultTestLoader.loadTestsFromTestCase(cls)]
    stream = io.StringIO();native = []
    with redirect_stdout(stream), redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite(cases))
    (args.output / 'specialized.log').write_text(stream.getvalue(), encoding='utf-8')
    for case in cases:
        native.extend(getattr(case, 'native_evidence', []))
    save(args.output / 'native-evidence.json', native)
    full = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], cwd=ROOT,
                          capture_output=True, text=True)
    (args.output / 'full.log').write_text(full.stdout + full.stderr, encoding='utf-8')
    save(args.output / 'acceptance.json', {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'specialized': {'tests': result.testsRun, 'failures': len(result.failures),
                        'errors': len(result.errors), 'skips': len(result.skipped)},
        'native_cases': len(native), 'full_exit_code': full.returncode,
        'accepted': result.wasSuccessful() and not result.skipped and len(native) == 9 and full.returncode == 0,
        'new_model_requests': 0, 'new_tokens': 0, 'reserved_gpu_seconds': 0,
        'scope': 'Scripted native engineering and immutable HTTP replay;no new inference/benchmark/SR'})
    save(args.output / 'artifact_manifest.json', {str(p.relative_to(args.output)).replace('\\', '/'): sha(p)
         for p in args.output.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})
    if not result.wasSuccessful() or result.skipped or len(native) != 9 or full.returncode:
        raise SystemExit(1)
    print('Report-only engineering acceptance passed')


if __name__ == '__main__':
    main()
