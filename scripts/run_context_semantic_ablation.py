"""One frozen114-request diagnosis; reuse N72 baseline instead of rerunning it."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_action_semantic_diagnosis import audit as audit_parent
from scripts.run_action_semantic_diagnosis import read, sha, request
from scripts.run_public_benchmark import actors
from smarthome_agent_rl.benchmarks.runner import completion, save
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_context_ablation import ARMS, enrich, evaluate
from smarthome_agent_rl.semantic_diagnosis import parse, schema
from smarthome_agent_rl.workflow_semantics import load_workflow_semantics


def prepare(config, parent=None):
    parent = Path(parent or ROOT / config['parent_run'])
    if sha(parent / 'artifact_manifest.json') != config['parent_artifact_sha256']:
        raise ValueError('Frozen N72 manifest differs')
    checked = audit_parent(parent)
    if not checked['verified']:
        raise ValueError('Parent audit failed: ' + str(checked['problems']))
    protocol = read(parent / 'protocol.json')
    if digest(protocol['inputs']) != config['parent_inputs_sha256']:
        raise ValueError('Frozen N72 inputs differ')
    if protocol['schema'] != schema():
        raise ValueError('Do not change original response schema')
    for key in ('model', 'model_seed', 'generation', 'actors', 'request_timeout'):
        if config[key] != protocol['config'][key]:
            raise ValueError('Frozen review setting changed: ' + key)
    labels = read(parent / 'developer-semantic-audit.json')
    if sha(parent / 'developer-semantic-audit.json') != config['developer_labels_sha256']:
        raise ValueError('Frozen developer readings differ')
    rules = load_workflow_semantics()
    inputs, baseline = [], []
    for index, item in enumerate(protocol['inputs']):
        episode = parent / 'source-evidence' / item['source_episode']
        contract, trace = read(episode / 'contract.json'), read(episode / 'harness_audit.json')
        proposal = next(r for r in trace['proposals'] if r['action_id'] == item['proposal_action_id'])
        inputs.append(enrich(item, contract['public_context'], proposal, trace['actual_observations'], rules))
        baseline.append(read(parent / 'records' / f'{index:03d}' / '9b_public.json'))
    if len(inputs) != config['proposals'] or len(protocol['coverage']) != config['episodes']:
        raise ValueError('Frozen real source coverage differs')
    return inputs, baseline, labels, protocol['coverage'], checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); config = read(args.config)
    inputs, baseline, labels, coverage, checked = prepare(config)
    if args.prepare_only:
        print(json.dumps({'inputs_sha256': digest(inputs), 'proposals': len(inputs), 'parent_audit': checked,
            'affected': {arm: sum(item['contexts'][arm.split('_', 1)[1]] !=
                read(ROOT/config['parent_run']/'protocol.json')['inputs'][i]['contexts']['public']
                for i, item in enumerate(inputs)) for arm in ARMS}}))
        return
    if digest(inputs) != config['inputs_sha256']:
        raise ValueError('Inputs changed after freeze')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit source/config before inference')
    if len(config['actors']) != 4 or config['slots_per_actor'] != 16:
        raise ValueError('Four actors and64 slots required')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT/config['parent_run'], args.output/'n72-reference')
    save(args.output/'protocol.json', {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=ROOT, text=True).strip(), 'config': config, 'inputs': inputs, 'coverage': coverage, 'schema': schema(),
        'parent_audit': checked})
    records = []; started = time.monotonic()
    limits = {a['id']: threading.Semaphore(config['slots_per_actor']) for a in config['actors']}
    def one(index, item, arm):
        actor = config['actors'][index % len(config['actors'])]
        with limits[actor['id']]:
            call = completion(actor['endpoint'], request(config, item, arm), config['request_timeout'])
        decision, error = None, None
        if call['error'] is None:
            try: decision = parse(call['text'])
            except (ValueError, TypeError, KeyError) as exc: error = str(exc)
        row = {'id': item['id'], 'arm': arm, 'actor_id': actor['id'], 'call': call,
               'decision': decision, 'parse_error': error}
        path = args.output/'records'/f'{index:03d}';path.mkdir(parents=True, exist_ok=True)
        save(path/(arm+'.json'), row)
        return row
    def interrupted(*unused): raise KeyboardInterrupt('Interrupted; completed requests retained')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with actors(config, args.output/'services'):
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures = [pool.submit(one, index, item, arm) for index, item in enumerate(inputs) for arm in ARMS]
                for future in as_completed(futures):
                    records.append(future.result())
                    if len(records) % 20 == 0: print(json.dumps({'completed': len(records), 'expected':114}), flush=True)
        report = evaluate(config, records, baseline, labels)
        report['wall_seconds'] = time.monotonic()-started
        report['resources'] = read(args.output/'services/lifecycle.json')
        save(args.output/'report.json', report)
        save(args.output/'state.json', {'stage':'awaiting_independent_audit', 'records':len(records)})
        print(json.dumps({'stage':'awaiting_independent_audit', 'checks':report['checks'],
                          'developer_case_gate':report['developer_case_gate']}), flush=True)
    except BaseException as exc:
        save(args.output/'failure.json', {'type':type(exc).__name__, 'message':str(exc), 'completed':len(records)})
        raise
    finally:
        save(args.output/'artifact_manifest.json', {str(p.relative_to(args.output)).replace('\\','/'):sha(p)
            for p in args.output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})


if __name__ == '__main__': main()
