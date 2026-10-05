"""Audit on-wire public-ID provenance and report the two frozen within-model pairs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.binding import catalog_prompt, visible_catalog
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.paired_stats import bootstrap_ci, mcnemar


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identifiers(arguments):
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if key in ('room_id', 'device_id') and isinstance(value, str):
                yield key, value
            elif isinstance(value, (dict, list)):
                yield from identifiers(value)
    elif isinstance(arguments, list):
        for value in arguments:
            yield from identifiers(value)


def episode(directory, variant):
    summary = read(directory / 'summary.json')
    audit = read(directory / 'harness_audit.json')
    config = read(directory / 'contract.json')['config']
    expected_binding = config['variant_policies'][variant]['identifier_binding']
    expected_model = config.get('variant_runtime', {}).get(variant, {}).get('model', config['actor_model'])
    calls = read(directory / 'model_calls.json')
    binding = audit['binding']
    assert len(binding) == (len(calls) if expected_binding else 0)
    counts = Counter()
    examples = []
    for index, call in enumerate(calls):
        assert call['request']['model'] == call['response']['model'] == expected_model
        messages = call['request']['messages']
        hints = [i for i, m in enumerate(messages) if m['content'].startswith('OBSERVED IDENTIFIER CANDIDATES\n')]
        if hints:
            assert expected_binding and hints == [len(messages) - 1]
            original = messages[:-1]
        else:
            original = messages
        catalog = visible_catalog(original)
        prompt = catalog_prompt(catalog)
        if expected_binding:
            record = binding[index]
            assert record['catalog'] == catalog
            assert record['prompt'] == prompt
            assert record['used'] == bool(hints)
            if prompt:
                assert messages[-1]['content'] == prompt
                counts['hint_calls'] += 1
        else:
            assert not hints
        counts['model_calls'] += 1
        usage = call['response'].get('usage') or {}
        counts['input_tokens'] += usage.get('prompt_tokens', 0)
        counts['output_tokens'] += usage.get('completion_tokens', 0)
        raw = call['response']['choices'][0]['message'].get('content') or ''
        try:
            action = json.loads(raw)['call']
            references = list(identifiers(action['arguments']))
        except (ValueError, TypeError, KeyError):
            continue
        for kind, value in references:
            counts[kind + '_references'] += 1
            if kind == 'room_id' and catalog['complete_rooms']:
                counts['room_references_with_complete_catalog'] += 1
                if value not in catalog['rooms']:
                    counts['room_mismatches'] += 1
                    examples.append({'call': index + 1, 'tool': action['tool'], 'wrong_room_id': value,
                                     'visible_room_ids': sorted(catalog['rooms'])})
            elif kind == 'device_id':
                if value not in catalog['devices']:
                    counts['unseen_device_references'] += 1
                if catalog['complete_devices']:
                    counts['device_references_with_complete_catalog'] += 1
                    if value not in catalog['devices']:
                        counts['device_mismatches_complete_catalog'] += 1
    assert counts['input_tokens'] + counts['output_tokens'] == summary['actor_tokens']
    errors = Counter(row['tool'] for row in audit['proposals']
                     if row.get('reached_executor') and row.get('simulator_error'))
    return {'directory': str(directory.relative_to(ROOT)), 'success': summary['success'],
            'actor_tokens': summary['actor_tokens'], 'unfinished': summary['task_failure'],
            'counts': dict(counts), 'executor_errors_by_tool': dict(errors),
            'room_mismatch_examples': examples}


def report(stage, gates):
    protocol = read(stage / 'protocol.json')
    official = read(stage / 'report.json')
    assert official['verified'] and len({s['task']['id'] for s in protocol['schedule']}) == 24
    results, aggregates = {}, {}
    for variant in protocol['variants']:
        rows = []
        for item in protocol['schedule']:
            path = episode_directory(stage, item, variant)
            row = episode(path, variant)
            row.update(task_id=item['task']['id'], actor_seed=item['actor_seed'],
                       category=item['task']['query_type'] + ':' + item['task']['case'])
            rows.append(row)
            results[(row['task_id'], row['actor_seed'], variant)] = row
        counts = Counter()
        errors = Counter()
        for row in rows:
            counts.update(row['counts'])
            errors.update(row['executor_errors_by_tool'])
        arm = official['arms'][variant]
        assert sum(row['success'] for row in rows) == arm['successes']
        assert sum(row['actor_tokens'] for row in rows) == arm['all_totals']['actor_tokens']
        assert sum(errors.values()) == arm['all_totals']['invalid_reached_executor']
        assert sum(row['unfinished'] for row in rows) == arm['unfinished']
        aggregates[variant] = {'episodes': len(rows), 'successes': arm['successes'],
            'sr': arm['success_rate'], 'counts': dict(counts), 'executor_errors_by_tool': dict(errors),
            'invalid_reached_executor': sum(errors.values()), 'unfinished': arm['unfinished'],
            'actor_tokens': arm['all_totals']['actor_tokens'],
            'actor_tokens_per_success': arm['cost_latency']['actor_tokens_per_success'],
            'agent_seconds': arm['cost_latency']['agent_seconds'],
            'room_mismatch_rate_eligible': counts['room_mismatches'] / counts['room_references_with_complete_catalog']
                if counts['room_references_with_complete_catalog'] else None}
    pairs = {}
    seeds = protocol['actor_seeds']
    tasks = {item['task']['id']: item['task'] for item in protocol['schedule']}
    categories = sorted({t['query_type'] + ':' + t['case'] for t in tasks.values()})
    for control, candidate in [('G', 'GB'), ('SFT9B', 'SFT9B_B')]:
        for item in protocol['schedule']:
            controls = [episode_directory(stage, item, v) for v in (control, candidate)]
            configs = [read(p / 'contract.json')['config'] for p in controls]
            for setting in ('model_endpoint', 'served_model', 'model_seed', 'generation', 'max_steps',
                            'recovery_per_action', 'extra_queries_max', 'judge_endpoint'):
                assert configs[0][setting] == configs[1][setting], (control, candidate, setting)
            first = [read(p / 'model_calls.json')[0]['request']['messages'] for p in controls]
            assert first[0] == first[1], 'Initial task prompt must be identical before observed IDs exist'
        a, b = aggregates[control], aggregates[candidate]
        mismatches_a, mismatches_b = (r['counts'].get('room_mismatches', 0) for r in (a, b))
        gate = {'sufficient_control_errors': mismatches_a >= gates['control_room_mismatch_min'],
            'room_mismatch_reduction': mismatches_a > 0 and mismatches_b <= mismatches_a * (1 - gates['room_mismatch_reduction_min']),
            'success_rate': b['sr'] >= a['sr'],
            'illegal_execution': b['invalid_reached_executor'] <= a['invalid_reached_executor'],
            'actor_tokens': b['actor_tokens'] <= a['actor_tokens'] * gates['actor_token_ratio_max']}
        task_deltas = {task: {
            'sr_delta': sum(int(results[(task, seed, candidate)]['success']) - int(results[(task, seed, control)]['success']) for seed in seeds) / len(seeds),
            'room_mismatch_delta': sum(results[(task, seed, candidate)]['counts'].get('room_mismatches', 0) - results[(task, seed, control)]['counts'].get('room_mismatches', 0) for seed in seeds) / len(seeds),
        } for task in tasks}
        ci = {metric: bootstrap_ci([[task_deltas[t][metric] for t in tasks
            if tasks[t]['query_type'] + ':' + tasks[t]['case'] == category] for category in categories])
            for metric in ('sr_delta', 'room_mismatch_delta')}
        by_seed = {str(seed): mcnemar([results[(task, seed, control)]['success'] for task in tasks],
                                     [results[(task, seed, candidate)]['success'] for task in tasks]) for seed in seeds}
        cases = [{'task_id': task, 'actor_seed': seed,
                  'control': results[(task, seed, control)], 'candidate': results[(task, seed, candidate)]}
                 for task in tasks for seed in seeds]
        pairs[candidate + '-' + control] = {
            'checks': gate, 'all_checks_pass': all(gate.values()),
            'sr_delta': b['sr'] - a['sr'], 'actor_token_ratio': b['actor_tokens'] / a['actor_tokens'],
            'room_mismatch_reduction': 1 - mismatches_b / mismatches_a if mismatches_a else None,
            'task_clustered_ci95': ci, 'by_seed': by_seed, 'paired_cases': cases,
        }
    decision = ('Warrants a larger G/GB development ablation; no deployment claim' if pairs['GB-G']['all_checks_pass'] else
                'Primary G mechanism inconclusive; retain G' if not pairs['GB-G']['checks']['sufficient_control_errors'] else
                'Primary G gate failed; retain G')
    return {'passed_provenance_audit': True, 'within_model_pair_contracts_equal': True,
        'source_commit': protocol['commit'], 'manifest_sha256': protocol['manifest_sha256'],
        'independent_tasks': len(tasks), 'seeds': seeds, 'arms': aggregates, 'pairs': pairs, 'decision': decision,
        'primary_pair': 'GB-G', 'secondary_pair': 'SFT9B_B-SFT9B',
        'metric_definitions': {
            'room_mismatches': 'Proposed room_id references outside an already visible successful complete get_rooms/get_home_state catalog, including blocked/nested workflow proposals',
            'unseen_device_references': 'Device IDs absent from seen public observations; incomplete discovery means these are not necessarily illegal',
            'executor_errors': 'Immediate errors after dispatch, excluding guard blocks and auxiliary queries',
            'confidence_intervals': 'Stratified bootstrap of task means across repeated seeds; room delta in errors/task/run',
        },
        'limitations': ['Historically exposed development tasks', 'Dynamic trajectories and reference opportunities differ',
            'Prompt tests exact-ID salience only, not all observation binding or planning',
            'Secondary rejected-adapter improvement cannot establish benefit to retained G',
            'Gate alone does not establish significance; generic report pairs use G reference'],
        'source_identity': {name: digest(ROOT / name) for name in ('smarthome_agent_rl/binding.py', 'scripts/report_binding_diagnosis.py')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    args = parser.parse_args()
    run = ROOT / args.run_dir
    output = run / 'binding-diagnostics.json'
    if output.exists():
        raise FileExistsError('Preserve the existing diagnosis')
    config = read(run / 'protocol.json')['config']
    value = report(run / 'dev', config['node_experiment']['gates'])
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in value.items() if k not in ('pairs', 'arms')}, ensure_ascii=False, indent=2))
    for name, arm in value['arms'].items():
        print(name, json.dumps(arm, ensure_ascii=False))
    for name, pair in value['pairs'].items():
        print(name, json.dumps({k: v for k, v in pair.items() if k != 'paired_cases'}, ensure_ascii=False))
