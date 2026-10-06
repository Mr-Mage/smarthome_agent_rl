"""Read-only acceptance of frozen all-600 evidence; never select or revise a policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.concurrency import episode_directory
from verify_benchmark import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = ROOT / args.run
    stage = run / 'full600'
    evidence = verify(stage)
    protocol = json.loads((stage / 'protocol.json').read_text())
    frozen = json.loads((run / 'protocol.json').read_text())
    config = protocol['config']
    report = json.loads((stage / 'report.json').read_text())
    manifest_path = ROOT / 'configs/benchmark-all/full600.json'
    manifest = json.loads(manifest_path.read_text())
    assert config['node_experiment']['node'] == 'N37'
    assert config == frozen['config'] and protocol['commit'] == frozen['commit']
    assert protocol['manifest_sha256'] == digest(manifest_path) == frozen['stages'][0]['manifest_sha256']
    assert protocol['variants'] == ['B0', 'G'] and protocol['actor_seeds'] == [42]
    assert protocol['expected_episodes'] == 1200 and evidence['expected_episodes'] == 1200
    rows = manifest['tasks']
    assert len(rows) == len({r['id'] for r in rows}) == 600
    counts = Counter((r['query_type'], r['case']) for r in rows)
    assert len(counts) == 12 and set(counts.values()) == {50}
    assert {i['task']['id'] for i in protocol['schedule']} == {r['id'] for r in rows}
    expected_rows = {r['id']: r for r in rows}
    calls = Counter()
    for item in protocol['schedule']:
        assert item['task'] == expected_rows[item['task']['id']]
        task_path = ROOT / 'deps/SimuHome/data/benchmark' / item['task']['path']
        assert digest(task_path) == item['task']['sha256']
        endpoints = []
        for variant in item['variants']:
            directory = episode_directory(stage, item, variant)
            actual = json.loads((directory / 'contract.json').read_text())['config']
            assert actual['variant_policies']['G'] == config['variant_policies']['G']
            endpoints.append(actual['model_endpoint'])
            for label, filename, options, model in (
                ('actor', 'model_calls.json', config['generation'], config['actor_model']),
                ('judge', 'judge_calls.json', config['judge_generation'], config['judge_model'])):
                path = directory / filename
                for row in json.loads(path.read_text()) if path.exists() else []:
                    request = row['request']
                    assert request['model'] == model
                    assert request['seed'] == 42 if label == 'actor' else request['seed'] in config['judge_seeds']
                    for key, value in options.items():
                        if key == 'extra_body':
                            assert all(request[k] == v for k, v in value.items())
                        else:
                            assert request[key] == value
                    if row.get('status') == 200:
                        assert row['response']['model'] == model
                        assert 'usage' in row['response']
                    calls[label] += 1
        assert endpoints[0] == endpoints[1]
    for variant in ('B0', 'G'):
        arm = report['arms'][variant]
        assert arm['episodes'] == arm['unique_tasks'] == 600 and arm['evaluator_errors'] == 0
        assert len(arm['categories']) == 12
        assert all(c['episodes'] == 50 for c in arm['categories'].values())
    result = {'node': 'N37', 'verified': True, 'commit': protocol['commit'],
        'episodes': 1200, 'tasks': 600, 'categories': 12,
        'manifest_sha256': digest(manifest_path), 'stage_files': evidence['files'],
        'requests': dict(calls), 'model_parameters_pairing_and_task_hashes_verified': True,
        'candidate_selection': False, 'default_variant': 'G',
        'formerly_sealed_tasks': len(manifest['historical_exposure']['previously_unused_ids']),
        'scope': 'Full official snapshot, including exposed tasks; not 600 independent holdout tasks. N13 formal result remains separate.'}
    (run / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
