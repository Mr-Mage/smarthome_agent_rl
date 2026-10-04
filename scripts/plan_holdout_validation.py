"""Admission decision from development discordance and unused-task metadata only."""
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest


def required_tasks(discordance, delta, power=.8, alpha=.025):
    q = max(discordance, abs(delta))
    normal = NormalDist()
    za, zb = normal.inv_cdf(1 - alpha / 2), normal.inv_cdf(power)
    return math.ceil((za * math.sqrt(q) + zb * math.sqrt(q - delta * delta)) ** 2 / (delta * delta))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--guard-baseline-report', default='runs/harness-v2/primary-v2/integration/report.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config_path = ROOT / 'configs/harness-validation-protocol.json'
    config = json.loads(config_path.read_text())
    report = json.loads((ROOT / args.report).read_text())
    selection = json.loads((ROOT / args.selection).read_text())
    inventory = json.loads((ROOT / args.inventory).read_text())
    guard_report = json.loads((ROOT / args.guard_baseline_report).read_text())
    if guard_report['phase'] != 'dev' or guard_report['reference'] != 'B0' or not guard_report['verified']:
        raise ValueError('G-B0 power planning must use verified dev evidence')
    pair = report['paired']['TimePlan']
    independent = report['arms']['G']['unique_tasks']
    q = (pair['wins'] + pair['losses']) / independent
    n = required_tasks(q, config['minimum_meaningful_sr_delta'], config['desired_power'], config['familywise_alpha'] / 2)
    guard_pair = guard_report['paired']['G']
    guard_q = (guard_pair['wins'] + guard_pair['losses']) / guard_report['arms']['G']['unique_tasks']
    guard_n = required_tasks(guard_q, config['minimum_meaningful_sr_delta'], config['desired_power'], config['familywise_alpha'] / 2)
    comparison_estimates = {'TimePlan-G': {'dev_discordance': q, 'required_tasks': n},
                            'G-B0': {'dev_discordance': guard_q, 'required_tasks': guard_n}}
    n = max(n, guard_n)
    enhancement = selection['winner'] != 'G'
    enough = inventory['remaining_tasks'] >= n
    result = {'admitted': enhancement and enough, 'selected_candidate': selection['winner'],
        'development_gate_passed': enhancement, 'unused_pool_sufficient': enough,
        'available_tasks': inventory['remaining_tasks'], 'estimated_required_tasks': n,
        'balanced_twelve_class_max': min(inventory['remaining_by_category'].values()) * 12,
        'dev_primary_discordance': q, 'dev_independent_tasks': independent,
        'target_delta': config['minimum_meaningful_sr_delta'], 'desired_power': config['desired_power'],
        'comparison_estimates': comparison_estimates,
        'reasons': (["No development enhancement passed frozen gates"] if not enhancement else []) +
                   (["Unused official pool is too small for the predeclared sample-size estimate"] if not enough else []),
        'decision': 'Freeze selected dev strategy; no final run or extra modules when admission fails; proceed to N20 archive',
        'limitations': 'Normal approximation for planning, not an exact achieved-power calculation. Historical dev G-B0 discordance is an estimate; both comparisons use conservative Holm alpha=0.025. Shared official generator does not prove independent template families.',
        'evidence_sha256': {name: digest(ROOT / name) for name in
            (args.report, args.selection, args.inventory, args.guard_baseline_report, str(config_path.relative_to(ROOT)))}}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Never overwrite a validation admission decision')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
