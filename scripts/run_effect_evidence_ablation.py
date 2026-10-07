"""Frozen N80 representation ablation; immutable parent runs are read only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_evidence_consistency import audit as audit_parent
from scripts.run_action_semantic_diagnosis import read, sha
from scripts.run_evidence_consistency_audit import prepare as prepare_parent, check_manifest
from scripts.run_public_benchmark import actors
from smarthome_agent_rl.benchmarks.runner import completion, save
from smarthome_agent_rl.effect_evidence_ablation import ARMS, PROMPT_BY_ARM, request, schema, evaluate
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_diagnosis import parse


def prepare(config, parent=None, n76=None, n78=None):
    parent = Path(parent or ROOT/config['parent_run'])
    check_manifest(parent, 'collection-manifest.json', config['parent_collection_sha256'])
    checked = audit_parent(parent, n76, n78, write_receipt=False)
    old = read(parent/'protocol.json')['config']
    inputs, coverage = prepare_parent(old, n76, n78)
    if digest(inputs) != config['inputs_sha256'] or len(inputs) != config['records']:
        raise ValueError('Frozen cohort differs')
    original = Path(n76 or ROOT/old['parents']['n76']['run'])
    previous = read(original/'protocol.json')['config']
    for key in ('model', 'model_seed', 'generation', 'actors', 'request_timeout'):
        if config[key] != previous[key]:
            raise ValueError('Frozen generation/resources differ: '+key)
    labels_path = original/'n75-reference/n74-reference/n72-reference/developer-semantic-audit.json'
    if sha(labels_path) != config['developer_labels_sha256']:
        raise ValueError('Original developer evidence differs')
    labels = [{'id': 'n76:'+r['id'], 'category': r['category']} for r in read(labels_path)['cases']]
    labels += [{'id': r['id'], 'category': 'UNLABELED'} for r in inputs if r['origin'] == 'n78']
    schemas = {arm: [schema(r['context'], arm) for r in inputs] for arm in ARMS}
    if list(ARMS) != config['arms'] or digest(PROMPT_BY_ARM) != config['prompts_sha256'] or digest(schemas) != config['schemas_sha256']:
        raise ValueError('Frozen representation differs')
    if len(config['actors']) != 4 or config['slots_per_actor'] != 16 or config['new_requests'] != len(inputs)*len(ARMS):
        raise ValueError('Four actors/64 slots and complete paired arms required')
    evaluate(config, inputs, [], labels)  # Check label coverage/groups before inference.
    return inputs, labels, coverage, schemas, checked


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); config = read(args.config); began = time.monotonic()
    inputs, labels, coverage, schemas, checked = prepare(config)
    preparation = time.monotonic()-began
    if args.prepare_only:
        print(json.dumps({'records': len(inputs), 'new_requests': config['new_requests'], 'parent_audit': checked,
                          'preparation_seconds': preparation})); return
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit source/config before inference')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'protocol.json', {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'config': config, 'inputs': inputs, 'labels': labels, 'coverage': coverage, 'schemas': schemas,
         'prompts': PROMPT_BY_ARM, 'parent_audit': checked, 'preparation_seconds': preparation})
    started = time.monotonic(); records = []
    limits = {a['id']: threading.Semaphore(config['slots_per_actor']) for a in config['actors']}
    def one(index, item, arm):
        actor = config['actors'][index % 4]
        with limits[actor['id']]:
            call = completion(actor['endpoint'], request(config, item, arm), config['request_timeout'])
        decision, error = None, None
        if call['error'] is None:
            try: decision = parse(call['text'])
            except (ValueError, TypeError, KeyError) as exc: error = str(exc)
        row = {'id': item['id'], 'arm': arm, 'actor_id': actor['id'], 'call': call, 'decision': decision, 'parse_error': error}
        directory = args.output/'records'/f'{index:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        save(directory/(arm+'.json'), row)
        return row
    def interrupted(*unused): raise KeyboardInterrupt('Interrupted;completed requests retained')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with actors(config, args.output/'services'):
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures = [pool.submit(one, i, r, arm) for i, r in enumerate(inputs) for arm in ARMS]
                for future in as_completed(futures):
                    records.append(future.result())
                    if len(records) % 10 == 0:
                        print(json.dumps({'completed': len(records), 'expected': config['new_requests']}), flush=True)
        report = evaluate(config, inputs, records, labels)
        report.update(wall_seconds=time.monotonic()-started, resources=read(args.output/'services/lifecycle.json'))
        save(args.output/'report.json', report)
        save(args.output/'state.json', {'stage': 'awaiting_independent_audit', 'records': len(records)})
        print(json.dumps({'complete': report['complete_records'], 'screen': report['diagnostic_screen_passed']}), flush=True)
    except BaseException as exc:
        save(args.output/'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'completed': len(records)}); raise
    finally:
        save(args.output/'artifact_manifest.json', {str(p.relative_to(args.output)).replace('\\', '/'): sha(p)
             for p in args.output.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})


if __name__ == '__main__': main()
