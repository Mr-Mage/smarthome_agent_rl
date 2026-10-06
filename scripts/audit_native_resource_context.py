"""Audit public claim provenance and actual actor HTTP disclosure without inference."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.execution.conflicts import resource_claims, detect_conflicts
from smarthome_agent_rl.execution.resource_context import fixed_claims
from scripts.analyze_native_task_runtime import audit_episode
from scripts.verify_benchmark import verify

PREFIX = 'PUBLIC NATIVE JOB / RESOURCE CONTEXT\n'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_resources(data, calls, adapter):
    problems, structures = [], {}
    jobs = {j['job_id']: j for j in data['jobs']}
    trace = sorted(data['trace'], key=lambda r: r['timestamp'])
    for row in trace:
        response = row['response']
        if row['tool'] == 'get_device_structure' and isinstance(response, dict) and not row['error'] and \
                response.get('status', {}).get('code') == 200 and isinstance(response.get('data'), dict) and \
                response['data'].get('device_id') == row['arguments']['device_id']:
            structures[row['arguments']['device_id']] = response['data']
        elif row['tool'] == 'schedule_workflow' and row['parent_step'] in jobs:
            job = jobs[row['parent_step']]
            proposed = fixed_claims(adapter, row['arguments']['steps'], structures)
            if proposed != job['payload'].get('resource_intention'):
                problems.append('Resource intention differs from preceding public schema and actual scheduled arguments')
            expected = resource_claims(proposed['claims'], namespace='isolated-episode',
                                       target_time=job['target_time'], tolerance=job['tolerance'])
            if expected != job['resource_claims']:
                problems.append('Durable claims differ from fixed targets or closed verification window')
    for conflict in data['resource_conflicts']['historical']:
        selected = [copy.deepcopy(jobs[side['job_id']]) for side in conflict['sides']]
        for job in selected:
            job['status'] = 'SCHEDULED'
        detected, _ = detect_conflicts(selected)
        if conflict not in detected:
            problems.append('Historical conflict does not follow from actual public resource claims')
    current, uncovered = detect_conflicts(data['jobs'])
    if data['resource_conflicts']['current']['conflicts'] != current or \
            data['resource_conflicts']['current']['uncovered'] != uncovered:
        problems.append('Exported live conflict report differs from durable final lifecycle')
    exposed = [e for e in data['events'] if e['kind'] == 'runtime_context_exposed']
    http_contexts = [m['content'] for c in calls for m in c['request']['messages'] if m['content'].startswith(PREFIX)]
    expected_texts = [PREFIX + json.dumps(e['context'], ensure_ascii=False, sort_keys=True) for e in exposed]
    if expected_texts != http_contexts:
        problems.append('Context exposure differs from actual actor HTTP messages')
    registered = set()
    for event in data['events']:
        if event['kind'] == 'native_registration':
            registered.add(event['job_id'])
        if event['kind'] != 'runtime_context_exposed':
            continue
        context = event['context']
        content = PREFIX + json.dumps(context, ensure_ascii=False, sort_keys=True)
        if hashlib.sha256(content.encode()).hexdigest() != event['content_sha256']:
            problems.append('Exposed content SHA differs from recorded projection')
        if context['jobs_total'] != len(registered) or len(context['jobs']) > 8 or \
                len(context['conflicts']) > 4 or len(content) - len(PREFIX) > 12000:
            problems.append('Context contains future registrations or exceeds frozen disclosure budget')
        for row in context['jobs']:
            if row['job_id'] not in registered:
                problems.append('Exposed job was not registered yet')
                continue
            job = jobs[row['job_id']]
            if row['claims'] != job['resource_claims'][:8] or row['target_time'] != job['target_time'] or \
                    row['window_end'] != job['target_time'] + job['tolerance'] or \
                    row['uncovered_steps'] != job['payload']['resource_intention']['uncovered'][:8]:
                problems.append('Exposed fixed intention differs from actual proposal')
        allowed = {'schema', 'policy', 'semantics', 'task_status', 'jobs', 'jobs_total', 'jobs_truncated',
                   'conflicts', 'conflicts_total', 'conflicts_truncated', 'uncovered_overlaps',
                   'uncovered_overlaps_total', 'uncovered_overlaps_truncated', 'size_truncated'}
        if set(context) != allowed or context['task_status'] in ('COMPLETED', 'FAILED'):
            problems.append('Resource context contains extra fields or unjustified Task conclusion')
    return {'problems': problems, 'jobs': len(jobs),
        'fixed_claims': sum(len(j['resource_claims']) for j in jobs.values()),
        'uncovered_steps': sum(len(j['payload']['resource_intention']['uncovered']) for j in jobs.values()),
        'historical_conflicts': len(data['resource_conflicts']['historical']),
        'context_requests': len(exposed), 'contexts_with_conflicts': sum(e['context']['conflicts_total'] > 0 for e in exposed)}


def audit(run, stage='calibration'):
    run, stage = Path(run), Path(run) / stage
    verification = verify(stage)
    protocol = read(stage / 'protocol.json')
    config = protocol['config']
    exp = config['node_experiment']
    reference, candidate = exp['reference'], exp['candidate']
    if protocol['variants'] != [reference, candidate] or reference != 'GTME' or candidate != 'GTMEC':
        raise ValueError('Resource comparison differs from frozen variant identities')
    if {k: v for k, v in config['variant_policies'][candidate].items() if k != 'task_runtime_context'} != \
            config['variant_policies'][reference] or not config['variant_policies'][candidate]['task_runtime_context']:
        raise ValueError('Candidate must isolate resource claims/context on the same event runtime')
    from smarthome_agent_rl.execution.simuhome_contract import SimuHomeContractAdapter
    from smarthome_agent_rl.guard import command_contracts, public_power_rules
    signatures, _ = command_contracts(ROOT / 'deps/SimuHome/src/simulator/domain/clusters')
    power, _ = public_power_rules(ROOT / 'deps/SimuHome/src/simulator/domain/devices')
    adapter = SimuHomeContractAdapter(signatures, power)
    rows, issues, cost = [], [], Counter()
    pairs = 0
    for item in protocol['schedule']:
        initial = []
        for arm in (reference, candidate):
            directory = episode_directory(stage, item, arm)
            issues.extend(audit_episode(directory)['problems'])
            calls = read(directory / 'model_calls.json')
            initial.append(calls[0]['request'])
            for role, receipts in [('actor', calls), ('judge', read(directory / 'judge_calls.json')
                                   if (directory / 'judge_calls.json').exists() else [])]:
                for call in receipts:
                    req, response = call['request'], call.get('response') or {}
                    usage = response.get('usage') or {}
                    cost[role + '_requests'] += 1
                    cost[role + '_tokens'] += usage.get('total_tokens', 0)
                    cost['missing_usage'] += 'total_tokens' not in usage
                    cost['http_errors'] += call['status'] != 200
                    model = config[role + '_model']
                    params = config['generation' if role == 'actor' else 'judge_generation']
                    expected = {k: v for k, v in params.items() if k != 'extra_body'}
                    expected.update(params.get('extra_body', {}))
                    if req.get('model') != model or response.get('model') != model or \
                            any(req.get(k) != v for k, v in expected.items()) or \
                            req.get('seed') not in ([item['actor_seed']] if role == 'actor' else config['judge_seeds']):
                        issues.append('HTTP model/seed/generation differs from frozen protocol')
            data = read(directory / 'task_runtime.json')
            if arm == candidate:
                result = audit_resources(data, calls, adapter)
                issues.extend(result['problems'])
                rows.append({'task_id': item['task']['id'], 'seed': item['actor_seed'], **result})
                db = sqlite3.connect((directory / 'task-runtime.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
                try:
                    stored = [json.loads(r[0]) for r in db.execute("SELECT data FROM records WHERE kind='conflict' ORDER BY id")]
                    if stored != data['resource_conflicts']['historical']:
                        issues.append('Historical conflicts differ from durable database')
                finally:
                    db.close()
            elif any(m['content'].startswith(PREFIX) for c in calls for m in c['request']['messages']):
                issues.append('Reference unexpectedly receives resource context')
        if initial[0] != initial[1]:
            issues.append('Initial paired public actor inputs differ')
        pairs += 1
    for call in read(run / 'services/inference-probe.json')['results']:
        cost['probe_requests'] += 1
        usage = (call.get('response') or {}).get('usage') or {}
        cost['probe_tokens'] += usage.get('total_tokens', 0)
        cost['missing_usage'] += 'total_tokens' not in usage
        cost['http_errors'] += call.get('status', 200) != 200
    totals = {key: sum(r[key] for r in rows) for key in ('jobs', 'fixed_claims', 'uncovered_steps',
               'historical_conflicts', 'context_requests', 'contexts_with_conflicts')}
    gates = exp['gates']
    checks = {'evidence_consistent': not issues, 'complete_episodes': verification['expected_episodes'] == gates['complete_episodes'],
        'public_tasks': len({r['task_id'] for r in rows}) == gates['public_tasks'],
        'native_jobs_exercised': totals['jobs'] >= gates['minimum_native_jobs'],
        'fixed_claims_exercised': totals['fixed_claims'] >= gates['minimum_fixed_claims'],
        'actual_context_exercised': totals['context_requests'] >= gates['minimum_context_requests'],
        'http_errors': cost['http_errors'] == 0, 'usage_complete': cost['missing_usage'] == 0}
    result = {'source_commit': protocol['commit'], 'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks': checks, 'engineering_accepted': all(checks.values()), 'problems': dict(Counter(issues)),
        'totals': totals, 'cost': dict(cost), 'initial_pairs': pairs, 'episodes': rows, 'default': 'G',
        'scope': 'Exposed official single-turn engineering ablation; fixed action intentions,not goal verification or persistent occupancy. No training,new benchmark,holdout benefit or automatic adoption.'}
    (run / 'resource-context-acceptance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', default='calibration')
    args = parser.parse_args()
    result = audit(args.run, args.stage)
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}))
    raise SystemExit(0 if result['engineering_accepted'] else 1)
