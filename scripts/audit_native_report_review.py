"""Native report-only coverage/cost audit from original pre-dispatch receipts."""
import argparse
from collections import Counter
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from scripts.report_benchmark import read, report
from scripts.verify_benchmark import verify
from smarthome_agent_rl.benchmarks.runner import save, valid_usage
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.profiling import percentile
from smarthome_agent_rl.report_semantic_review import attach_report_reviewer
from smarthome_agent_rl.semantic_context import digest, MUTATIONS
from smarthome_agent_rl.semantic_context_enriched import build_enriched_context
from smarthome_agent_rl.workflow_semantics import load_workflow_semantics


def audit_episode(directory, rules):
    directory = Path(directory);contract = read(directory / 'contract.json')
    config = contract['config'];summary = read(directory / 'summary.json')
    audit = read(directory / 'harness_audit.json');public = contract['public_context']
    proposals = [p for p in audit['proposals'] if p['tool'] in MUTATIONS and
                 p.get('reached_executor') and not p['blocked']]
    candidate = config['variant_policies'][summary['variant']].get('report_semantic_review', False)
    path = directory / 'semantic_review_calls.json'
    rows = read(path) if path.exists() else []
    if not candidate:
        if rows or summary.get('semantic_review') is not None or any('semantic_verification' in p for p in audit['proposals']):
            raise ValueError('Reference G unexpectedly contains review intervention')
    elif len(rows) != len(proposals):
        raise ValueError('Guard-admitted mutation/review coverage differs')
    _, reviewer = attach_report_reviewer(config, summary['variant'])
    for proposal, row in zip(proposals, rows):
        cutoff = proposal['actual_calls_before'] + proposal['actual_calls'] - 1
        observations = audit['actual_observations'];dispatch = observations[cutoff]
        if dispatch['tool'] != proposal['tool'] or dispatch['arguments'] != proposal['arguments']:
            raise ValueError('Review cutoff is not the exact original tool dispatch')
        context = build_enriched_context(public['query'], {'tool': proposal['tool'], **proposal['arguments']},
            observations[:cutoff], user_location=public['user_location'], initial_time=public['current_time'],
            contract=proposal.get('contract'), references=True, workflow_rules=rules)
        if asdict(context) != row['context'] or asdict(context) != proposal['semantic_context'] or \
                digest(asdict(context)) != proposal['semantic_context_sha256']:
            raise ValueError('Review context does not follow from original public pre-dispatch evidence')
        call = row['call']
        if call.get('error') is None:
            raw = json.loads(call['raw_response']);choice = raw['choices'][0]
            if raw != call['response'] or choice['message']['content'] != call['text'] or \
                    raw.get('usage') != call['usage'] or choice.get('finish_reason') != call['finish_reason']:
                raise ValueError('Review raw HTTP/text/usage/finish differs')
        reviewer.transport = lambda endpoint, body, timeout: copy.deepcopy(call)
        result = reviewer.verify(context)
        if reviewer.records[-1] != row or result.as_dict() != proposal['semantic_verification']:
            raise ValueError('Original review receipt/result/cost or sequence differs from replay')
    costs = reviewer.costs() if reviewer is not None else {'requests': 0, 'tokens': 0,
        'missing_usage': 0, 'request_seconds': 0, 'accepted_model_outputs': 0, 'fallbacks': 0}
    if candidate and costs != summary['semantic_review']:
        raise ValueError('Review cost differs from retained HTTP evidence')
    calls = read(directory / 'model_calls.json')
    judges = read(directory / 'judge_calls.json') if (directory / 'judge_calls.json').exists() else []
    for values, field in ((calls, 'actor'), (judges, 'judge')):
        if len(values) != summary[field + '_model_calls'] or sum(r['response'].get('usage', {}).get('total_tokens', 0)
                for r in values) != summary[field + '_tokens']:
            raise ValueError(field + ' costs differ from HTTP evidence')
    spans = read(directory / 'phase_profile.json')['spans']
    review_spans = [s for s in spans if s['kind'] == 'semantic_review']
    if len(review_spans) != len(rows) or any(s['phase'] != 'agent' or s['outcome'] != 'returned' for s in review_spans):
        raise ValueError('Review wall spans/phase/outcome differ')
    wall = sum(s['end_seconds'] - s['start_seconds'] for s in review_spans)
    if wall + .001 < costs['request_seconds']:
        raise ValueError('Review HTTP time exceeds inclusive wall span')
    return {'task_id': summary['task_id'], 'variant': summary['variant'], 'actor_seed': summary['actor_seed'],
        'success': summary['success'], 'guard_admitted_mutations': len(proposals),
        'zero_mutation': not proposals, 'review': costs, 'review_wall_seconds': wall,
        'fallbacks': dict(Counter(r['fallback_reason'] for r in rows if r['fallback_reason'])),
        'report_verdicts': dict(Counter(r['result']['verdict'] for r in rows)),
        'model_verdicts': dict(Counter(r['model_decision']['verdict'] if r['model_decision'] else 'INVALID' for r in rows)),
        'actor_tokens': summary['actor_tokens'], 'judge_tokens': summary['judge_tokens'],
        'actor_calls': len(calls), 'judge_calls': len(judges),
        'missing_actor_judge_usage': sum(not valid_usage(r['response'].get('usage')) for r in calls + judges),
        'actor_requests': [r['request'] for r in calls],
        'tool_sequence': [(r['tool'], r['arguments']) for r in audit['actual_observations']],
        'agent_seconds': sum(s['end_seconds'] - s['start_seconds'] for s in spans if s['kind'] == 'agent'),
        'episode_seconds': summary['duration_seconds']}


def audit(run, stage='calibration', source_root=None):
    run = Path(run);directory = run / stage
    verified = verify(directory);protocol = read(directory / 'protocol.json')
    config = protocol['config'];exp = config['node_experiment'];gates = exp['gates']
    if protocol['variants'] != ['G', 'Candidate'] or exp['node'] != 'N78':
        raise ValueError('Native report review differs from frozen stage identities')
    if config != read(ROOT / 'configs/native-report-semantic-review.json'):
        raise ValueError('Stage configuration differs from frozen repository configuration')
    rules = load_workflow_semantics(source_root=source_root)
    episodes = [audit_episode(episode_directory(directory, item, variant), rules)
                for item in protocol['schedule'] for variant in protocol['variants']]
    main_report = read(directory / 'report.json');arms = {}
    for name in protocol['variants']:
        rows = [r for r in episodes if r['variant'] == name]
        fallback = Counter();verdict = Counter();model_verdict = Counter()
        for row in rows:
            fallback.update(row['fallbacks']);verdict.update(row['report_verdicts']);model_verdict.update(row['model_verdicts'])
        arms[name] = {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
            'guard_admitted_mutations': sum(r['guard_admitted_mutations'] for r in rows),
            'zero_mutation_episodes': sum(r['zero_mutation'] for r in rows),
            'review': {k: sum(r['review'][k] for r in rows) for k in rows[0]['review']},
            'review_wall_seconds': sum(r['review_wall_seconds'] for r in rows),
            'fallbacks': dict(fallback), 'report_verdicts': dict(verdict), 'model_verdicts': dict(model_verdict),
            'actor_tokens': sum(r['actor_tokens'] for r in rows), 'judge_tokens': sum(r['judge_tokens'] for r in rows),
            'actor_calls': sum(r['actor_calls'] for r in rows), 'judge_calls': sum(r['judge_calls'] for r in rows),
            'missing_actor_judge_usage': sum(r['missing_actor_judge_usage'] for r in rows),
            'latency': {k: {'p50': percentile([r[k] for r in rows], .5),
                           'p95': percentile([r[k] for r in rows], .95)} for k in
                        ('review_wall_seconds', 'agent_seconds', 'episode_seconds')}}
        arms[name]['actor_plus_review_tokens'] = arms[name]['actor_tokens'] + arms[name]['review']['tokens']
        if arms[name]['successes'] != main_report['arms'][name]['successes']:
            raise ValueError('Native success denominator differs from original official report')
    candidate = arms['Candidate'];pairs = Counter();mapped = {(r['task_id'], r['actor_seed'], r['variant']): r for r in episodes}
    pair_cases = []
    for item in protocol['schedule']:
        key = (item['task']['id'], item['actor_seed'])
        base, new = (mapped[(*key, v)] for v in ('G', 'Candidate'))
        group = 'win' if new['success'] and not base['success'] else 'loss' if base['success'] and not new['success'] else 'same'
        pair = {'task_id': key[0], 'actor_seed': key[1], 'result': group,
                'actor_requests_same': base['actor_requests'] == new['actor_requests'],
                'tool_sequence_same': base['tool_sequence'] == new['tool_sequence'],
                'review_requests': new['review']['requests']}
        pair_cases.append(pair);pairs[group] += 1
    checks = {'complete_episodes': len(episodes) == gates['complete_episodes'],
              'public_tasks': len({r['task_id'] for r in episodes}) == gates['public_tasks'],
              'seeds': protocol['actor_seeds'] == [42, 43, 44],
              'coverage': candidate['review']['requests'] == candidate['guard_admitted_mutations'],
              'nonempty_review': candidate['review']['requests'] >= gates['minimum_reviews'],
              'usage_complete': candidate['review']['missing_usage'] == 0 and all(a['missing_actor_judge_usage'] == 0 for a in arms.values()),
              'no_reference_review': arms['G']['review']['requests'] == 0, 'artifact_verified': verified['verified']}
    result = {'checks': checks, 'engineering_passed': all(checks.values()), 'arms': arms,
              'paired_outcomes': dict(pairs), 'pairs': pair_cases,
              'native_report': 'calibration/report.json', 'artifact_files': verified['files'],
              'scope': 'Exposed official single-turn coverage/failure/cost diagnosis. Reviews are synchronous and may change live virtual time. No blocking/actor feedback,semantic accuracy,independent holdout,SR improvement claim or default adoption.'}
    save(run / (stage + '-review-audit.json'), result)
    print(json.dumps({'engineering_passed': result['engineering_passed'], 'checks': checks,
                      'candidate_reviews': candidate['review']['requests']}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', default='calibration');parser.add_argument('--source-root', type=Path)
    args = parser.parse_args();result = audit(args.run, args.stage, args.source_root)
    if not result['engineering_passed']:
        raise SystemExit(1)
