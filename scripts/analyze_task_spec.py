"""Independent N33 provenance and coverage audit; official scores remain unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.action_state import MUTATIONS
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.concurrency import episode_directory


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = ROOT / args.run
    config = read(run / 'protocol.json')['config']
    gates = config['node_experiment']['gates']
    target_ids = set(config['node_experiment']['manual_review']['task_ids'])
    totals = {'episodes': 0, 'extracted': 0, 'goals': 0, 'metadata_errors': 0,
              'mutations': 0, 'associated_mutations': 0, 'goals_without_action_evidence': 0}
    reviews = []
    sources = []
    for stage_name in ('smoke', 'dev'):
        stage = run / stage_name
        assert read(stage / 'report.json')['verified']
        protocol = read(stage / 'protocol.json')
        for item in protocol['schedule']:
            task = item['task']
            path = episode_directory(stage, item, 'GTS')
            audit = read(path / 'harness_audit.json')
            summary = read(path / 'summary.json')
            query = read(ROOT / 'deps/SimuHome/data/benchmark' / task['path'])['query']
            spec = audit['task_spec']
            assert spec['query_sha256'] == hashlib.sha256(query.encode()).hexdigest()
            assert spec['coverage'] == 'model_extraction_unverified'
            ids = {goal['goal_id'] for goal in spec['goals']}
            for goal in spec['goals']:
                span = goal['source_span']
                assert query[span['start']:span['end']] == goal['source_text']
                assert goal['satisfaction'] == 'unverified'
            for turn in spec['turns']:
                assert set(turn['goal_refs']) <= ids
                for binding in turn['bindings'].values():
                    assert binding['semantic_match'] == 'model_proposed_unverified'
                    for source in binding['sources']:
                        observation = audit['actual_observations'][source['observation_index'] - 1]
                        assert source['response_sha256'] == hashlib.sha256(
                            json.dumps(observation['response'], sort_keys=True).encode()).hexdigest()
                        assert observation['response']['status']['code'] == 200
                        assert observation['response']['error'] is None
            actions = audit['action_lifecycle']['actions']
            for action in actions:
                assert set(action['goal_ids']) <= ids
            if stage_name == 'dev':
                mutations = [action for action in actions if not action['extra_query'] and
                             action['tool'] in MUTATIONS and any(t['to'] == 'dispatched' for t in action['transitions'])]
                totals['episodes'] += 1
                totals['extracted'] += int(spec['status'] == 'extracted')
                totals['goals'] += len(spec['goals'])
                totals['metadata_errors'] += len(spec['errors'])
                totals['mutations'] += len(mutations)
                totals['associated_mutations'] += sum(bool(action['goal_ids']) for action in mutations)
                totals['goals_without_action_evidence'] += sum(not goal['action_evidence'] for goal in spec['goals'])
                if task['id'] in target_ids:
                    reviews.append({'task_id': task['id'], 'actor_seed': summary['actor_seed'],
                        'query': query, 'success': summary['success'], 'goals': spec['goals'],
                        'path': str(path.relative_to(ROOT)), 'audit_sha256': digest(path / 'harness_audit.json')})
            sources.append({'path': str(path.relative_to(ROOT)), 'audit_sha256': digest(path / 'harness_audit.json')})
    report = read(run / 'dev/report.json')
    g, candidate = report['arms']['G'], report['arms']['GTS']
    ratio = candidate['all_totals']['actor_tokens'] / g['all_totals']['actor_tokens']
    extracted = totals['extracted'] / totals['episodes']
    associated = totals['associated_mutations'] / totals['mutations'] if totals['mutations'] else None
    checks = {'success_rate': candidate['success_rate'] - g['success_rate'] >= gates['sr_delta_min'],
              'illegal_execution': candidate['all_totals']['invalid_reached_executor'] <= g['all_totals']['invalid_reached_executor'],
              'actor_tokens': ratio <= gates['actor_token_ratio_max'],
              'extracted_episodes': extracted >= gates['extracted_episode_ratio_min'],
              'associated_mutations': associated is not None and associated >= gates['associated_mutation_ratio_min'],
              'manual_semantics_review': False}
    (run / 'task-spec-case-review.json').write_text(json.dumps({'criteria': config['node_experiment']['manual_review']['criteria'],
        'cases': reviews, 'status': 'pending'}, indent=2)+'\n')
    output = {'totals': totals, 'extracted_episode_ratio': extracted, 'associated_mutation_ratio': associated,
        'actor_token_ratio': ratio, 'checks': checks, 'winner': 'G', 'provenance_verified': True,
        'sources': sources, 'manual_review': 'pending',
        'limits': 'Model extraction and association are not semantic correctness or goal satisfaction. Exposed developer tasks only.'}
    (run / 'selection.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: output[k] for k in ('totals', 'actor_token_ratio', 'checks', 'winner')}))


if __name__ == '__main__':
    main()
