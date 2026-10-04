"""Verify complete paired artifacts and summarize all failures/costs; no selective reruns."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import STRATA, digest
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.paired_stats import mcnemar, bootstrap_ci, holm
from smarthome_agent_rl.profiling import cost_summary


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def invalid_observation(response):
    if not isinstance(response, dict):
        return False
    code = response.get('status', {}).get('code')
    return code in (400, 404, 409, 422) or (code is None and response.get('error') is not None)


def baseline_action_metrics(events):
    # The pristine ReAct loop also emits "observation" for parser/unknown-tool
    # failures. Only an action followed by its observation reached run_tool.
    pending_action = False
    executed = reached_invalid = 0
    preexecution_rejections = sum(row['event'] == 'consecutive_failure' for row in events)
    for row in events:
        if row['event'] == 'action':
            pending_action = True
        elif row['event'] == 'observation':
            if pending_action:
                executed += 1
                reached_invalid += invalid_observation(json.loads(row['payload']))
            pending_action = False
    return {'invalid_proposed': reached_invalid + preexecution_rejections,
        'invalid_reached_executor': reached_invalid, 'executed_tool_calls': executed,
        'structured_rejections': preexecution_rejections}


def episode_metrics(directory):
    summary = read(directory / 'summary.json')
    calls = read(directory / 'model_calls.json')
    judges = read(directory / 'judge_calls.json') if (directory / 'judge_calls.json').exists() else []
    retrieval = read(directory / 'retrieval_calls.json') if (directory / 'retrieval_calls.json').exists() else []
    tokenization = read(directory / 'tokenization_calls.json') if (directory / 'tokenization_calls.json').exists() else []
    audit_path = directory / 'harness_audit.json'
    if audit_path.exists():
        audit = read(audit_path)
        blocked = sum(row['blocked'] for row in audit['proposals'])
        budget_blocked = sum(row['blocked'] and row.get('layer') == 'recovery' for row in audit['proposals'])
        structured_rejections = sum('validation_error' in row for row in audit['structured'])
        actual_invalid = sum(invalid_observation(row['response']) for row in audit['actual_observations'] if not row['extra_query'])
        invalid_proposed = blocked - budget_blocked + structured_rejections + actual_invalid
        verification_failures = sum(row.get('verification', {}).get('verified') is False for row in audit['proposals'])
        recoveries = sum(row.get('recovered', False) for row in audit['proposals'])
        extra_queries = audit['extra_queries']
        extra_query_latency = sum(row['duration_seconds'] for row in audit['actual_observations'] if row['extra_query'])
        executed = sum(not row['extra_query'] for row in audit['actual_observations'])
    else:
        events = read(directory / 'agent_events.json')
        baseline = baseline_action_metrics(events)
        actual_invalid = baseline['invalid_reached_executor']
        invalid_proposed = baseline['invalid_proposed']
        structured_rejections = baseline['structured_rejections']
        blocked = verification_failures = recoveries = extra_queries = 0
        budget_blocked = 0
        extra_query_latency = 0
        executed = baseline['executed_tool_calls']
    token_total = sum(row['response'].get('usage', {}).get('total_tokens', 0) for row in calls)
    judge_total = sum(row['response'].get('usage', {}).get('total_tokens', 0) for row in judges)
    if token_total != summary['actor_tokens'] or judge_total != summary['judge_tokens']:
        raise ValueError('Token accounting differs from HTTP evidence')
    if summary['infrastructure_error'] or summary['official_score'] == -1:
        raise ValueError(f'Infrastructure failure cannot be included as a completed primary run: {directory}')
    phases = {}
    if (directory / 'phase_profile.json').exists():
        profile = read(directory / 'phase_profile.json')
        spans = profile['spans']
        def seconds(kind, phase=None):
            return sum(s['end_seconds'] - s['start_seconds'] for s in spans
                       if s['kind'] == kind and (phase is None or s['phase'] == phase))
        agent_spans = [s for s in spans if s['kind'] == 'agent']
        phases = {'agent_seconds': seconds('agent') if agent_spans else None,
            'post_agent_seconds': max(0, profile['episode_seconds'] -
                (agent_spans[-1]['end_seconds'] - profile['episode_start_seconds'])) if agent_spans else None,
            'evaluator_seconds': seconds('evaluator'), 'tool_seconds': seconds('tool_dispatch'),
            'agent_wait_seconds': seconds('waiting', 'agent'),
            'evaluation_wait_seconds': seconds('waiting', 'post_agent'),
            'evaluation_client_seconds': seconds('simulator_client', 'post_agent'),
            'retrieval_seconds': sum(r.get('duration_seconds', 0) for r in retrieval)}
        # Agent residual includes parsing, trace persistence and framework work, not GPU compute.
        phases['agent_residual_seconds'] = max(0, phases['agent_seconds'] - phases['tool_seconds'] -
            phases['agent_wait_seconds'] - sum(r['duration_seconds'] for r in calls)) if agent_spans else None
    return {**summary, **phases, 'invalid_proposed': invalid_proposed, 'invalid_reached_executor': actual_invalid,
        'guard_blocked': blocked, 'structured_rejections': structured_rejections,
        'recovery_budget_blocked': budget_blocked,
        'verification_failures': verification_failures, 'recovered_actions': recoveries,
        'executed_tool_calls': executed, 'extra_queries': extra_queries,
        'extra_query_latency': extra_query_latency, 'retrieval_calls': len(retrieval),
        'retrieval_tokens': sum(row['input_tokens'] for row in retrieval),
        'actor_latency': sum(row['duration_seconds'] for row in calls),
        'judge_latency': sum(row['duration_seconds'] for row in judges),
        'tokenization_calls': len(tokenization),
        'tokenization_latency': sum(row['duration_seconds'] for row in tokenization)}


def report(run):
    protocol = read(run / 'protocol.json')
    completion = read(run / 'completion.json')
    if not completion['complete']:
        raise ValueError('Incomplete run')
    if protocol['phase'] == 'align':
        raise ValueError('Alignment is a diagnostic, not a benchmark report')
    variants = protocol['variants']
    default_seed = protocol['config']['model_seed']
    seeds = protocol.get('actor_seeds', [default_seed])
    primary_seed = protocol['config'].get('primary_actor_seed', seeds[0])
    if primary_seed not in seeds:
        raise ValueError('Primary actor seed must be included in this run')
    def key(item, variant):
        return item['task']['id'], variant, item.get('actor_seed', default_seed)
    metrics = {}
    for item in protocol['schedule']:
        for variant in item['variants']:
            directory = episode_directory(run, item, variant)
            record = episode_metrics(directory)
            if record['task_id'] != item['task']['id'] or record['variant'] != variant:
                raise ValueError('Artifact task/variant identity mismatch')
            if record.get('actor_seed', default_seed) != key(item, variant)[2] or key(item, variant) in metrics:
                raise ValueError('Duplicate episode or actor seed identity mismatch')
            metrics[key(item, variant)] = record
    if len(metrics) != protocol['expected_episodes'] or len(metrics) != completion['episodes']:
        raise ValueError('Missing or duplicate episode evidence')
    summary = {}
    for variant in variants:
        records = [metrics[key(item, variant)] for item in protocol['schedule']]
        success_records = [r for r in records if r['success']]
        numeric = ['actor_tokens', 'judge_tokens', 'actor_model_calls', 'judge_model_calls',
            'invalid_proposed', 'invalid_reached_executor', 'structured_rejections',
            'executed_tool_calls', 'guard_blocked', 'extra_queries',
            'verification_failures', 'recovered_actions', 'recovery_budget_blocked', 'duration_seconds', 'retrieval_tokens',
            'actor_latency', 'judge_latency', 'extra_query_latency', 'tokenization_calls', 'tokenization_latency']
        summary[variant] = {'episodes': len(records), 'successes': len(success_records),
            'cost_latency': cost_summary(records),
            'success_rate': len(success_records) / len(records), 'unfinished': sum(r['task_failure'] for r in records),
            'unique_tasks': len({i['task']['id'] for i in protocol['schedule']}),
            'by_actor_seed': {str(seed): {'episodes': sum(key(i, variant)[2] == seed for i in protocol['schedule']),
                'successes': sum(metrics[key(i, variant)]['success'] for i in protocol['schedule'] if key(i, variant)[2] == seed),
                'all_totals': {m: sum(metrics[key(i, variant)][m] for i in protocol['schedule'] if key(i, variant)[2] == seed)
                               for m in numeric}} for seed in seeds},
            'evaluator_errors': sum(r['official_score'] == -1 for r in records),
            'all_totals': {m: sum(r[m] for r in records) for m in numeric},
            'successful_means': {m: sum(r[m] for r in success_records) / len(success_records)
                                 if success_records else None for m in numeric},
            'categories': {qt + ':' + case: {'episodes': sum((item['task']['query_type'], item['task']['case']) == (qt, case)
                for item in protocol['schedule']), 'successes': sum(metrics[key(item, variant)]['success']
                for item in protocol['schedule'] if (item['task']['query_type'], item['task']['case']) == (qt, case))}
                for qt, case in STRATA}}
    def compare(reference, variant, seed=primary_seed):
        items = [i for i in protocol['schedule'] if key(i, variant)[2] == seed]
        baseline = [metrics[key(item, reference)] for item in items]
        comparison = [metrics[key(item, variant)] for item in items]
        result = mcnemar([r['success'] for r in baseline], [r['success'] for r in comparison])
        differences = [[int(metrics[key(item, variant)]['success']) -
            int(metrics[key(item, reference)]['success']) for item in items
            if (item['task']['query_type'], item['task']['case']) == (qt, case)] for qt, case in STRATA]
        result['success_delta_ci95'] = bootstrap_ci(differences)
        result['actor_token_delta'] = sum(b['actor_tokens'] - a['actor_tokens'] for a, b in zip(baseline, comparison)) / len(baseline)
        return result
    reference = variants[0]
    pairs = {variant: compare(reference, variant) for variant in variants[1:]}
    adjusted = holm({v: result['p_exact'] for v, result in pairs.items()})
    for variant, result in pairs.items():
        result['p_holm'] = adjusted[variant]
    repeated = {}
    if len(seeds) > 1:
        unique = {i['task']['id']: i for i in protocol['schedule']}
        for variant in variants[1:]:
            differences = [[sum(int(metrics[(item['task']['id'], variant, seed)]['success']) -
                int(metrics[(item['task']['id'], reference, seed)]['success']) for seed in seeds) / len(seeds)
                for item in unique.values() if (item['task']['query_type'], item['task']['case']) == (qt, case)] for qt, case in STRATA]
            repeated[variant] = {'task_clustered_delta_ci95': bootstrap_ci(differences),
                'by_actor_seed': {str(seed): compare(reference, variant, seed) for seed in seeds},
                'independent_tasks': len(unique), 'repeated_runs_per_task': len(seeds)}
    mechanism = {}
    for baseline_variant, comparison_variant in [('G', 'GC'), ('GV', 'Full')]:
        if baseline_variant in variants and comparison_variant in variants:
            mechanism[comparison_variant + '-' + baseline_variant] = compare(baseline_variant, comparison_variant)
    manifest = {str(path.relative_to(run)): digest(path) for path in run.rglob('*')
                if path.is_file() and path.name not in ('report.json', 'report.md', 'artifact_manifest.json')}
    (run / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    result = {'verified': True, 'phase': protocol['phase'], 'commit': protocol['commit'],
        'reporter_version': 'executor-attribution-v2',
        'reporter_sha256': digest(ROOT / 'scripts/report_benchmark.py'),
        'metric_definitions': {'invalid_reached_executor': 'Immediate errors after run_tool dispatch; excludes parser/unknown-tool rejections and auxiliary queries',
            'invalid_proposed': 'Preexecution rejections plus immediate errors from proposed calls; recovery-budget blocks excluded'},
        'reference': reference, 'arms': summary, 'paired': pairs, 'artifact_files': len(manifest),
        'primary_actor_seed': primary_seed, 'repeated_seed_diagnostics': repeated,
        'statistical_unit': 'task; primary McNemar uses one predeclared actor seed; repeat CI resamples task clusters within categories',
        'exploratory_context_pairs': mechanism,
        'judge_panel': 'Three seeds from one local model/service, not three independent judges',
        'limitations': 'Preserves original live virtual-time behavior; task failures stay in denominator.'}
    (run / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Official harness results', '', '| Arm | Success | Actor tokens | Invalid proposed/reached | Extra queries |',
             '|---|---:|---:|---:|---:|']
    for v, row in summary.items():
        total = row['all_totals']
        lines.append(f"| {v} | {row['successes']}/{row['episodes']} | {total['actor_tokens']} | {total['invalid_proposed']}/{total['invalid_reached_executor']} | {total['extra_queries']} |")
    lines += ['', f'Paired reference: {reference}. Exact McNemar, stratified bootstrap 95% CI, Holm adjustment.',
              'Full numbers, categories, failures and successful-task costs: report.json.',
              f'Artifact SHA256 inventory: artifact_manifest.json ({len(manifest)} files).']
    (run / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'verified': True, 'arms': {v: [r['successes'], r['episodes']] for v, r in summary.items()}}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    report(ROOT / parser.parse_args().run_dir)
