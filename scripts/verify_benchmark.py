"""Independent immutable artifact verification, suitable for checking a copied archive."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.variant_runtime import runtime


def verify(run):
    run = Path(run).resolve()
    manifest = json.loads((run / 'artifact_manifest.json').read_text(encoding='utf-8'))
    failures = []
    for name, expected in manifest.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or not path.is_file():
            failures.append({'path': name, 'problem': 'missing or outside artifact root'})
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append({'path': name, 'problem': 'SHA256 mismatch'})
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    completion = json.loads((run / 'completion.json').read_text(encoding='utf-8'))
    expected_summaries = []
    for item in protocol['schedule']:
        for variant in item['variants']:
            expected_summaries.append(str((episode_directory(run, item, variant) / 'summary.json').relative_to(run)).replace('\\', '/'))
            if protocol.get('scheduler') == 'actor-affine-queue-v2':
                worker = item.get('variant_workflows', {}).get(variant)
                slot = next((s for s in protocol['execution_slots'] if s['id'] == worker), None)
                if slot is None or slot['actor_id'] != item['workflow']:
                    failures.append({'task': item['task']['id'], 'variant': variant, 'problem': 'paired actor affinity violated'})
                    continue
                contract_path = episode_directory(run, item, variant) / 'contract.json'
                if contract_path.exists():
                    config = json.loads(contract_path.read_text())['config']
                    if 'actor_seed' in item and config['model_seed'] != item['actor_seed']:
                        failures.append({'task': item['task']['id'], 'variant': variant, 'problem': 'actor seed isolation mismatch'})
                    calls_path = contract_path.parent / 'model_calls.json'
                    if calls_path.exists() and any(call['request'].get('seed') != config['model_seed']
                            for call in json.loads(calls_path.read_text())):
                        failures.append({'task': item['task']['id'], 'variant': variant, 'problem': 'HTTP actor seed differs from contract'})
                    expected=runtime(protocol['config'],variant,slot) if 'config' in protocol else {'model_endpoint':f"http://127.0.0.1:{slot['actor_port']}/v1"}
                    if config['simulator_url'] != f"http://127.0.0.1:{slot['simulator_port']}/api" or (
                        config['model_endpoint'] != expected['model_endpoint']):
                        failures.append({'task': item['task']['id'], 'variant': variant, 'problem': 'episode endpoint isolation mismatch'})
                    if 'config' in protocol and (config['served_model']!=expected['served_model'] or config['generation']!=expected['generation']):
                        failures.append({'task':item['task']['id'],'variant':variant,'problem':'variant runtime differs from frozen protocol'})
                    if 'config' in protocol and calls_path.exists():
                        for call in json.loads(calls_path.read_text()):
                            if call['request']['model']!=expected['served_model'] or (
                                call.get('status')==200 and call['response'].get('model')!=expected['served_model']):
                                failures.append({'task':item['task']['id'],'variant':variant,'problem':'HTTP served model identity mismatch'})
                else:
                    failures.append({'task': item['task']['id'], 'variant': variant, 'problem': 'missing isolation contract'})
    if set(expected_summaries) != {name for name in manifest if name.endswith('/summary.json')}:
        failures.append({'problem': 'missing/extra episode summary'})
    if len(expected_summaries) != len(set(expected_summaries)) or len(expected_summaries) != protocol['expected_episodes']:
        failures.append({'problem': 'duplicate schedule path or expected episode mismatch'})
    if completion['episodes'] != protocol['expected_episodes'] or not completion['complete']:
        failures.append({'problem': 'episode coverage mismatch'})
    result = {'verified': not failures, 'files': len(manifest),
              'expected_episodes': len(expected_summaries), 'failures': failures}
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise ValueError('Artifact verification failed')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    verify(parser.parse_args().run_dir)
