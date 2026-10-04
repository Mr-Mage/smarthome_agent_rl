"""Offline dev attribution: separate deterministic replay from model interpretation."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.guard import GuardError
from smarthome_agent_rl.time_plan import TimePlan, RELATIVE


def audit(run):
    report = json.loads((run / 'report.json').read_text())
    if not report['verified'] or report['phase'] != 'dev':
        raise ValueError('Complete verified dev run required')
    protocol = json.loads((run / 'protocol.json').read_text())
    totals, unique, details = Counter(), {'T': set(), 'R': set()}, []
    for item in protocol['schedule']:
        directory = episode_directory(run, item, 'TimePlan')
        data = json.loads((directory / 'harness_audit.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        task = json.loads((ROOT / 'deps/SimuHome/data/benchmark' / item['task']['path']).read_text())
        ledger = data['time_plan']
        expected = len(list(RELATIVE.finditer(task['query'])))
        totals['episodes'] += 1
        totals['recognized_phrases'] += expected
        totals['table_rows'] += len(ledger['table'])
        totals['covered_episodes'] += bool(ledger['table'])
        totals['table_missing_episodes'] += bool(expected and not ledger['table'])
        totals['unfinished'] += summary['task_failure']
        totals['query_budget_exhausted'] += sum('prequery_budget_exhausted' in p.get('uncovered', []) for p in data['proposals'])
        if summary.get('task_failure_kind'):
            totals['failure_' + summary['task_failure_kind']] += 1
        entry = {'task_id': item['task']['id'], 'actor_seed': item['actor_seed'],
            'evidence': str(directory.relative_to(ROOT)), 'success': summary['success'],
            'actor_tokens': summary['actor_tokens'], 'actor_model_calls': summary['actor_model_calls'],
            'public_query': task['query'], 'public_start_time': task['initial_home_config'].get('base_time'),
            'model_interpretation': ledger['table'], 'events': []}
        replay = TimePlan()
        replay.rows, replay.table = ledger.get('phrases', []), copy.deepcopy(ledger['table'])
        # Receipts used in finish are actual tool responses, not model claims.
        for proposal in data['proposals']:
            if proposal['tool'] != 'schedule_workflow' or proposal.get('layer') == 'schema':
                continue
            replay.metadata = {'refs': proposal.get('time_refs', []), 'dispositions': []}
            failure = None
            try:
                replay.check('schedule_workflow', proposal['arguments'])
            except GuardError as exc:
                failure = exc.detail
            recorded = proposal.get('layer') == 'time_plan'
            totals['T_checked_proposals'] += 1
            totals['T_triggered'] += recorded
            totals['T_replay_disagreements'] += recorded != bool(failure)
            totals['uncovered_step_refs'] += proposal.get('time_refs', []).count('uncovered')
            if recorded:
                unique['T'].add(item['task']['id'])
                entry['events'].append({'module': 'T', 'turn': proposal['turn'], 'arguments': proposal['arguments'],
                    'refs': proposal.get('time_refs', []), 'deterministic_result': failure,
                    'online_record': proposal.get('detail'), 'replay_agrees': bool(failure) == recorded,
                    'semantic_false_rejection': 'Unresolved unless model-to-intent mapping is independently reviewed'})
        for row in data['structured']:
            error = row.get('validation_error', '')
            receipt_error = any(s in error for s in ('Finish must account', 'no actual receipt', 'needs an explanation'))
            totals['R_triggered'] += receipt_error
            totals['metadata_rejections'] += bool(error and not receipt_error)
            if receipt_error:
                unique['R'].add(item['task']['id'])
                prior_receipts = [o for o in data['actual_observations'] if o['turn'] <= row['turn'] and
                                  o['tool'] in ('schedule_workflow', 'cancel_workflow') and not o['extra_query']]
                entry['events'].append({'module': 'R', 'turn': row['turn'], 'deterministic_result': error,
                                       'prior_public_receipts': prior_receipts})
        if expected or entry['events']:
            details.append(entry)
    if totals['episodes'] != report['arms']['TimePlan']['episodes']:
        raise ValueError('Audit coverage mismatch')
    return {'run': str(run.relative_to(ROOT)), 'report_sha256': digest(run / 'report.json'),
        'counts': dict(totals), 'triggered_unique_tasks': {k: len(v) for k, v in unique.items()}, 'details': details,
        'attribution': 'Both checks share planning prompt and metadata. N16 bundled arm cannot distinguish prompt, T and R causal effects.',
        'limits': 'Replay checks consistency with the recorded model table, not correct intent parsing. Wrong infeasibility, missed requirements and semantic false rejection remain unresolved without manual public-evidence review. No hidden evaluator goals used.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(ROOT / args.run)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Never overwrite mechanism evidence')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'counts': result['counts'], 'triggered_unique_tasks': result['triggered_unique_tasks']}))
