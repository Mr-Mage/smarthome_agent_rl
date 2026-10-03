"""Verify complete paired artifacts and summarize all failures/costs; no selective reruns."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import STRATA, digest
from smarthome_agent_rl.paired_stats import mcnemar, bootstrap_ci, holm


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
    return {**summary, 'invalid_proposed': invalid_proposed, 'invalid_reached_executor': actual_invalid,
        'guard_blocked': blocked, 'structured_rejections': structured_rejections,
        'recovery_budget_blocked': budget_blocked,
        'verification_failures': verification_failures, 'recovered_actions': recoveries,
        'executed_tool_calls': executed, 'extra_queries': extra_queries,
        'extra_query_latency': extra_query_latency, 'retrieval_calls': len(retrieval),
        'retrieval_tokens': sum(row['input_tokens'] for row in retrieval),
        'actor_latency': sum(row['duration_seconds'] for row in calls),
        'judge_latency': sum(row['duration_seconds'] for row in judges)}


def report(run):
    protocol = read(run / 'protocol.json')
    completion = read(run / 'completion.json')
    if not completion['complete']:
        raise ValueError('Incomplete run')
    if protocol['phase'] == 'align':
        raise ValueError('Alignment is a diagnostic, not a benchmark report')
    variants = protocol['variants']
    metrics = {}
    for item in protocol['schedule']:
        for variant in item['variants']:
            directory = run / f"worker{item['workflow']}" / item['task']['id'] / variant / 'lightning'
            record = episode_metrics(directory)
            if record['task_id'] != item['task']['id'] or record['variant'] != variant:
                raise ValueError('Artifact task/variant identity mismatch')
            metrics[(item['task']['id'], variant)] = record
    if len(metrics) != protocol['expected_episodes'] or len(metrics) != completion['episodes']:
        raise ValueError('Missing or duplicate episode evidence')
    summary = {}
    for variant in variants:
        records = [metrics[(item['task']['id'], variant)] for item in protocol['schedule']]
        success_records = [r for r in records if r['success']]
        numeric = ['actor_tokens', 'judge_tokens', 'actor_model_calls', 'judge_model_calls',
            'invalid_proposed', 'invalid_reached_executor', 'structured_rejections',
            'executed_tool_calls', 'guard_blocked', 'extra_queries',
            'verification_failures', 'recovered_actions', 'recovery_budget_blocked', 'duration_seconds', 'retrieval_tokens',
            'actor_latency', 'judge_latency', 'extra_query_latency']
        summary[variant] = {'episodes': len(records), 'successes': len(success_records),
            'success_rate': len(success_records) / len(records), 'unfinished': sum(r['task_failure'] for r in records),
            'evaluator_errors': sum(r['official_score'] == -1 for r in records),
            'all_totals': {m: sum(r[m] for r in records) for m in numeric},
            'successful_means': {m: sum(r[m] for r in success_records) / len(success_records)
                                 if success_records else None for m in numeric},
            'categories': {qt + ':' + case: {'episodes': sum((item['task']['query_type'], item['task']['case']) == (qt, case)
                for item in protocol['schedule']), 'successes': sum(metrics[(item['task']['id'], variant)]['success']
                for item in protocol['schedule'] if (item['task']['query_type'], item['task']['case']) == (qt, case))}
                for qt, case in STRATA}}
    def compare(reference, variant):
        baseline = [metrics[(item['task']['id'], reference)] for item in protocol['schedule']]
        comparison = [metrics[(item['task']['id'], variant)] for item in protocol['schedule']]
        result = mcnemar([r['success'] for r in baseline], [r['success'] for r in comparison])
        differences = [[int(metrics[(item['task']['id'], variant)]['success']) -
            int(metrics[(item['task']['id'], reference)]['success']) for item in protocol['schedule']
            if (item['task']['query_type'], item['task']['case']) == (qt, case)] for qt, case in STRATA]
        result['success_delta_ci95'] = bootstrap_ci(differences)
        result['actor_token_delta'] = sum(b['actor_tokens'] - a['actor_tokens'] for a, b in zip(baseline, comparison)) / len(baseline)
        return result
    reference = variants[0]
    pairs = {variant: compare(reference, variant) for variant in variants[1:]}
    adjusted = holm({v: result['p_exact'] for v, result in pairs.items()})
    for variant, result in pairs.items():
        result['p_holm'] = adjusted[variant]
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
