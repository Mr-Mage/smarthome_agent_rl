"""Compact metrics/identity receipts; unique raw episode evidence stays in runs/."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest


def summarize(run):
    protocol = json.loads((run / 'protocol.json').read_text())
    result = {'run': str(run.relative_to(ROOT)), 'commit': protocol['commit'],
        'config_sha256': protocol['config_sha256'], 'stages': {},
        'selection': json.loads((run / 'selection.json').read_text()),
        'resource_cost': json.loads((run / 'resource-cost.json').read_text()), 'evidence_sha256': {}}
    for stage in protocol['stages']:
        phase = stage['phase']
        directory = run / phase
        report = json.loads((directory / 'report.json').read_text())
        result['stages'][phase] = {k: report[k] for k in ('verified', 'reference', 'arms', 'paired',
            'repeated_seed_diagnostics', 'primary_actor_seed', 'artifact_files', 'statistical_unit')}
        completion = json.loads((directory / 'completion.json').read_text())
        result['stages'][phase]['timing'] = {k: completion[k] for k in ('episodes', 'elapsed_seconds', 'startup_seconds')}
        for filename in ('report.json', 'artifact_manifest.json', 'protocol.json', 'source_identity.json', 'simulator_identity.json'):
            result['evidence_sha256'][str((directory / filename).relative_to(ROOT))] = digest(directory / filename)
    for filename in ('protocol.json', 'selection.json', 'resource-cost.json', 'commands.json'):
        result['evidence_sha256'][str((run / filename).relative_to(ROOT))] = digest(run / filename)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'nodes': [summarize(ROOT / r) for r in args.runs],
        'limits': ['Development comparisons only; no reuse of final for tuning',
            'Live simulator clock and stochastic generation preserved',
            'HTTP includes queue, network, Gateway and inference',
            'Tool time includes nested retrieval; do not add those durations twice',
            'GPU resource seconds are reserved H100 wall time, not active compute; A800 shared allocation excluded']},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
