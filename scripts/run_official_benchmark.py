"""One dispatch entry for pinned static and dynamic official single-turn runs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def command(benchmark, config_path, output, *, source=None, stage='calibration', calibration=None,
            launch_actors=False, python=sys.executable):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if benchmark == 'HomeBench':
        if source is None:
            raise ValueError('Pinned HomeBench source is required')
        argv = [python, str(ROOT / 'scripts/run_public_benchmark.py'), '--config', str(config_path),
                '--source', str(source), '--output', str(output), '--stage', stage]
        if calibration is not None:
            argv += ['--calibration', str(calibration)]
        if launch_actors:
            argv += ['--launch-actors']
        return argv
    if benchmark != 'SimuHome':
        raise ValueError('This round permits only official single-turn HomeBench and SimuHome')
    if source is not None or calibration is not None or stage != 'calibration' or launch_actors:
        raise ValueError('SimuHome uses its frozen node stages and existing service supervisor')
    if not config.get('driver_python') or not config.get('node_experiment', {}).get('stages'):
        raise ValueError('A frozen SimuHome node config and existing Lightning interpreter are required')
    return [config['driver_python'], str(ROOT / 'scripts/run_node_experiment.py'),
            '--config', str(config_path), '--run-dir', str(output)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['HomeBench', 'SimuHome'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--stage', choices=['calibration', 'full'], default='calibration')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--launch-actors', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Freeze dispatch without starting services')
    args = parser.parse_args()
    config = args.config.resolve()
    argv = command(args.benchmark, config, args.output.resolve(), source=args.source, stage=args.stage,
                   calibration=args.calibration, launch_actors=args.launch_actors)
    dispatch_dir = args.output.parent / (args.output.name + '-dispatch')
    dispatch_dir.mkdir(parents=True, exist_ok=False)
    receipt = {'benchmark': args.benchmark, 'argv': argv,
               'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
               'prepare_only': args.prepare_only, 'status': 'prepared',
               'scope': 'native protocols, evaluators and resource budgets retained; no aggregate cross-benchmark SR'}
    path = dispatch_dir / 'entrypoint.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    if args.prepare_only:
        print(json.dumps(receipt))
        return
    started = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT)
    receipt.update(status='complete' if result.returncode == 0 else 'failed', exit_code=result.returncode,
                   seconds_including_child=time.monotonic() - started)
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
