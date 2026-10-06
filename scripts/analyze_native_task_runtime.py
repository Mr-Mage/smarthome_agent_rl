"""Audit frozen native runtime lifecycle, public receipts and official scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.execution.episode import virtual_seconds
from scripts.verify_benchmark import verify


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_episode(directory):
    data = read(directory / 'task_runtime.json')
    contract = read(directory / 'contract.json')
    native_path = directory / 'official_result.json'
    native = read(native_path) if native_path.exists() else None
    audit = read(directory / 'harness_audit.json')
    summary = read(directory / 'summary.json')
    problems = []
    tasks, jobs, workflows = data['tasks'], data['jobs'], data['workflows']
    if len(tasks) != 1 or tasks[0]['goal'] != contract['public_context']['query'] or \
            tasks[0]['expected_postconditions'] != [] or tasks[0]['status'] != 'WAITING':
        problems.append('Public task boundary or conservative completion violated')
    if native is None and not summary.get('task_failure'):
        problems.append('Native result missing without retained ordinary task failure')
    elif native is not None and native['evaluation_result']['score'] != summary['official_score']:
        problems.append('Runtime replaced the native official score')
    # JSON export must agree with durable database, not merely an in-memory sidecar.
    db = sqlite3.connect((directory / 'task-runtime.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
    try:
        for kind, exported in [('task', tasks), ('job', jobs), ('workflow', workflows), ('trace', data['trace'])]:
            stored = [json.loads(row[0]) for row in db.execute('SELECT data FROM records WHERE kind=? ORDER BY id', (kind,))]
            if stored != exported:
                problems.append(f'Durable {kind} differs from exported evidence')
    finally:
        db.close()
    trace = {r['invocation_id']: r for r in data['trace']}
    task_ids = {t['task_id'] for t in tasks}
    if any(r['task_id'] not in task_ids for r in trace.values()):
        problems.append('Dispatched invocation has no public task linkage')
    fingerprint = lambda r: json.dumps([r['tool'], r.get('arguments'), r['response']], sort_keys=True)
    if Counter(map(fingerprint, trace.values())) != Counter(map(fingerprint, audit['actual_observations'])):
        problems.append('Runtime/actual tool evidence differs or supervisor cost is missing')
    if data['supervisor_queries'] > audit['extra_queries'] or audit['extra_queries'] > contract['config']['extra_queries_max']:
        problems.append('Supervisor escaped shared public query budget')
    wf_by_id = {wf['workflow_id']: wf for wf in workflows}
    registration_ids = []
    for job in jobs:
        wf = wf_by_id[job['workflow_id']]
        registration = next((e for e in wf['evidence'] if e.get('kind') == 'registration'), None)
        if registration is None:
            problems.append('Native registration intent has no completed receipt linkage')
            continue
        invocation = trace[registration['invocation_id']]
        registration_ids.append(invocation['invocation_id'])
        if invocation['tool'] != 'schedule_workflow' or invocation['workflow_id'] != wf['workflow_id'] or \
                invocation['parent_step'] != job['job_id'] or job['task_id'] not in task_ids or \
                job['target_time'] != virtual_seconds(invocation['arguments']['start_time']):
            problems.append('Native registration/job identity differs')
        if job['mode'] != 'device_native':
            problems.append('Unexpected mutation/wakeup outside the frozen native scope')
        if job['status'] in ('DONE', 'FAILED'):
            result = job['evidence'][-1]
            observed = result['observed_at']
            if type(observed) not in (int, float) or not job['target_time'] <= observed <= job['target_time'] + job['tolerance']:
                problems.append('Timed outcome lacks an in-window public observation')
            clocks = [e for e in result['evidence'] if e.get('kind') == 'public_observation_clock']
            if clocks:
                row = trace.get(clocks[-1].get('invocation_id'))
                try:
                    valid_clock = row is not None and row['tool'] == 'get_current_time' and not row['error'] and \
                        row['response']['status']['code'] == 200 and \
                        virtual_seconds(row['response']['data']['now']) == observed and \
                        clocks[-1].get('observed_at') == observed
                except (KeyError, TypeError, ValueError):
                    valid_clock = False
                if not valid_clock:
                    problems.append('Timed outcome clock does not match its public receipt')
            elif contract['config'].get('variant_policies', {}).get('GTME', {}).get('task_runtime_clock') == 'public_events':
                problems.append('Event runtime timed outcome lacks clock receipt provenance')
        if job['status'] == 'DONE' and (job['payload']['uncovered'] or not job['payload']['conditions']):
            problems.append('Uncovered action intention incorrectly verified')
    if len(registration_ids) != len(set(registration_ids)):
        problems.append('Same native registration adopted twice')
    successful = {r['invocation_id'] for r in trace.values() if r['tool'] == 'schedule_workflow'
                  and not r['error'] and r['response']['status']['code'] == 200}
    if not successful.issubset(set(registration_ids)):
        problems.append('Successful native registration missing durable job')
    return {'task_id': summary['task_id'], 'problems': problems, 'jobs': len(jobs),
        'job_statuses': dict(Counter(j['status'] for j in jobs)),
        'supervisor_queries': data['supervisor_queries'], 'trace_calls': len(trace),
        'native_clock_callbacks': data.get('event_supervision', {}).get('native_clock_callbacks',
            sum(e['kind'] == 'public_clock_supervision' and
                e.get('phase') == 'native_virtual_time_advanced' for e in data['events']))}


def analyze(run, stage='calibration'):
    run = Path(run)
    directory = run / stage
    verification = verify(directory)
    protocol = read(directory / 'protocol.json')
    report = read(directory / 'report.json')
    config = protocol['config']
    gates = config['node_experiment']['gates']
    candidate = config['node_experiment'].get('candidate', 'GTM')
    if protocol['variants'] != ['G', candidate] or protocol['expected_episodes'] != gates['complete_episodes']:
        raise ValueError('Native task-runtime experiment differs from frozen protocol')
    policies = config['variant_policies']
    if {k: v for k, v in policies[candidate].items()
            if k not in ('task_runtime', 'task_runtime_tolerance', 'task_runtime_clock')} != policies['G']:
        raise ValueError('Native task-runtime must isolate unchanged G')
    rows = [audit_episode(episode_directory(directory, item, candidate)) for item in protocol['schedule']]
    problems = [p for row in rows for p in row['problems']]
    jobs = sum(r['jobs'] for r in rows)
    callbacks = sum(r['native_clock_callbacks'] for r in rows)
    checks = {'complete_episodes': verification['expected_episodes'] == gates['complete_episodes'],
        'public_tasks': len({r['task_id'] for r in rows}) == gates['public_tasks'], 'evidence_consistent': not problems,
        'native_jobs_exercised': jobs >= gates['minimum_native_jobs'],
        'native_time_callback_exercised': callbacks >= gates['minimum_native_time_callbacks']}
    verified_jobs = sum(r['job_statuses'].get('DONE', 0) for r in rows)
    if 'minimum_verified_native_jobs' in gates:
        checks['native_state_verification_exercised'] = verified_jobs >= gates['minimum_verified_native_jobs']
    if 'candidate_executions' in gates:
        checks['candidate_executions'] = len(rows) == gates['candidate_executions']
    result = {'source_commit': protocol['commit'], 'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks': checks, 'engineering_accepted': all(checks.values()),
        'problems': problems, 'episodes': rows, 'native_jobs': jobs, 'verified_native_jobs': verified_jobs,
        'native_clock_callbacks': callbacks,
        'supervisor_queries': sum(r['supervisor_queries'] for r in rows),
        'official_successes': {a: r['successes'] for a, r in report['arms'].items()},
        'paired': report['paired'], 'default': 'G',
        'conclusion': 'Engineering attachment only; exposed single-turn diagnostics do not establish stable SR benefit',
        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (directory/'protocol.json', directory/'report.json', directory/'artifact_manifest.json')}}
    (run / 'native-runtime-acceptance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', default='calibration')
    args = parser.parse_args()
    result = analyze(args.run, args.stage)
    print(json.dumps({k: v for k, v in result.items() if k not in ('episodes', 'paired', 'input_sha256')}))
    if not result['engineering_accepted']:
        raise SystemExit(1)
