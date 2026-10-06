"""Frozen G prompt audit with the deployed tokenizer; no simulator or model IO."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.evidence_context import READS, render, restore


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def freeze(config_path, output):
    config = read(config_path)
    source = ROOT / config['source_run']
    assert read(source / 'protocol.json')['commit'].startswith(config['source_commit'])
    files = []
    for stage, expected in config['stages'].items():
        stage_root = source / stage
        assert read(stage_root / 'report.json')['verified']
        audits = sorted(stage_root.glob('**/G/lightning/harness_audit.json'))
        assert len(audits) == expected, (stage, len(audits), expected)
        for audit in audits:
            paths = [audit, audit.with_name('model_calls.json'), audit.with_name('contract.json'),
                     audit.with_name('summary.json')]
            summary = read(paths[-1])
            assert summary['variant'] == 'G'
            files.append({'stage': stage, 'episode': str(audit.parent.relative_to(ROOT)),
                          'files': {str(p.relative_to(ROOT)): digest(p) for p in paths}})
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'config_sha256': digest(config_path), 'config': config, 'episodes': files,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(output / 'source-manifest.json', manifest)
    print(json.dumps({'frozen_episodes': len(files), 'manifest_sha256': digest(output/'source-manifest.json')}))


def analyze(output, model):
    from transformers import AutoTokenizer
    began, cpu = time.monotonic(), time.process_time()
    manifest = read(output / 'source-manifest.json')
    config = manifest['config']
    assert not (output / 'report.json').exists(), 'Do not overwrite completed audit'
    assert sorted(READS) == sorted(config['readonly_tools'])
    for episode in manifest['episodes']:
        for name, expected in episode['files'].items():
            assert digest(ROOT/name) == expected, name
    tokenizer_start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    load_seconds = time.monotonic() - tokenizer_start
    identity = {p.name: digest(p) for p in Path(model).iterdir()
                if p.name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json')}
    stages, episodes, cases, mismatch = {}, [], [], []
    token_time, token_calls = 0.0, 0
    for episode in manifest['episodes']:
        folder = ROOT / episode['episode']
        audit, contract = read(folder / 'harness_audit.json'), read(folder / 'contract.json')
        stage = stages.setdefault(episode['stage'], {'episodes': 0, 'requests': 0, 'raw_tokens': 0,
            'selected_tokens': 0, 'candidate_tokens': 0, 'requests_with_references': 0,
            'selected_requests': 0, 'references': 0, 'roundtrips': 0, 'source_matches': 0,
            'protected_messages': 0, 'protected_unchanged': 0, 'usage_tokens': 0})
        totals = {'episode': episode['episode'], 'requests': 0, 'raw_tokens': 0, 'selected_tokens': 0,
                  'references': 0, 'roundtrips': 0}
        for call_index, call in enumerate(read(folder / 'model_calls.json')):
            if call['provider'] != 'actor':
                continue
            raw = call['request']['messages']
            candidate, refs = render(raw, audit['actual_observations'])
            assert restore(candidate) == raw
            stage['roundtrips'] += 1
            # Independent checks of retained plans, first full sources, real receipts and indices.
            referenced = {r['message_index'] for r in refs}
            for i, message in enumerate(raw):
                if i not in referenced:
                    stage['protected_messages'] += 1
                    assert candidate[i] == message
                    stage['protected_unchanged'] += 1
            for ref in refs:
                current = audit['actual_observations'][ref['observation_index'] - 1]
                source = audit['actual_observations'][ref['source_observation_index'] - 1]
                assert current['turn'] == ref['observed_turn']
                assert source['turn'] == ref['source_turn']
                assert not current['extra_query'] and not source['extra_query']
                assert current['tool'] == source['tool'] == ref['tool']
                assert current['arguments'] == source['arguments'] == ref['args']
                assert current['response'] == source['response']
                assert raw[ref['source_message_index']]['content'] == raw[ref['message_index']]['content']
                assert candidate[ref['source_message_index']] == raw[ref['source_message_index']]
                stage['source_matches'] += 1
            kwargs = call['request'].get('chat_template_kwargs',
                config.get('chat_template_kwargs', {'enable_thinking': False}))
            started = time.monotonic()
            raw_count = len(tokenizer.apply_chat_template(raw, tokenize=True, add_generation_prompt=True, **kwargs))
            candidate_count = (len(tokenizer.apply_chat_template(candidate, tokenize=True,
                add_generation_prompt=True, **kwargs)) if refs else raw_count)
            token_calls += 1 + bool(refs)
            token_time += time.monotonic() - started
            selected = bool(refs) and candidate_count < raw_count
            usage = call.get('response', {}).get('usage', {}).get('prompt_tokens')
            if usage is not None:
                stage['usage_tokens'] += usage
                if raw_count != usage:
                    mismatch.append({'episode': episode['episode'], 'call_index': call_index,
                                     'local_tokens': raw_count, 'http_prompt_tokens': usage})
            else:
                mismatch.append({'episode': episode['episode'], 'call_index': call_index,
                                 'reason': 'missing HTTP prompt usage'})
            stage['requests'] += 1
            stage['raw_tokens'] += raw_count
            stage['candidate_tokens'] += candidate_count
            stage['selected_tokens'] += candidate_count if selected else raw_count
            stage['requests_with_references'] += bool(refs)
            stage['selected_requests'] += selected
            stage['references'] += len(refs)
            totals['requests'] += 1
            totals['raw_tokens'] += raw_count
            totals['selected_tokens'] += candidate_count if selected else raw_count
            totals['references'] += len(refs)
            totals['roundtrips'] += 1
            if refs:
                name = f'case-{len(cases):04d}.json'
                write(output / name, {'source_episode': episode['episode'], 'call_index': call_index,
                    'refs': refs, 'original_messages': raw, 'managed_messages': candidate,
                    'original_tokens': raw_count, 'managed_tokens': candidate_count,
                    'used': selected, 'roundtrip': True})
                cases.append({'path': name, 'sha256': digest(output/name)})
        stage['episodes'] += 1
        episodes.append(totals)
    gates = config['offline_gates']
    dev = stages['dev']
    reference_count = sum(s['references'] for s in stages.values())
    checks = {'lossless_roundtrip': all(s['roundtrips'] == s['requests'] for s in stages.values()),
        'public_source_match': (all(s['source_matches'] == s['references'] for s in stages.values())
                                if reference_count else None),
        'protected_messages': all(s['protected_messages'] == s['protected_unchanged'] for s in stages.values()),
        'exact_deployed_tokenizer': not mismatch,
        'input_token_savings': dev['selected_tokens'] / dev['raw_tokens'] <= gates['dev_input_token_ratio_max']}
    for name, expected in ((p, h) for e in manifest['episodes'] for p,h in e['files'].items()):
        assert digest(ROOT/name) == expected
    report = {'verified': True, 'checks': checks, 'online_eligible': all(checks.values()),
        'winner': 'G', 'stages': stages, 'episodes': episodes, 'reference_cases': cases,
        'tokenizer_identity': identity, 'tokenizer_model': str(Path(model).resolve()),
        'tokenizer_mismatches': mismatch, 'source_manifest_sha256': digest(output/'source-manifest.json'),
        'source_sha256': digest(ROOT/'smarthome_agent_rl/evidence_context.py'),
        'cost': {'wall_seconds': time.monotonic()-began, 'process_cpu_seconds': time.process_time()-cpu,
            'tokenizer_load_seconds': load_seconds, 'tokenizer_calls': token_calls,
            'tokenizer_seconds': token_time, 'actor_inference_calls': 0, 'judge_inference_calls': 0,
            'reserved_h100_gpu_seconds': 0, 'a800_resource_cost': 'not attributable; resident judge untouched'},
        'limits': 'A null public-source check means no reference was exercised. '
                  'Lossless restore proves evidence availability, not model understanding or task success. '
                  'Fixed historical G prompts do not predict candidate trajectories or full episode savings.'}
    write(output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('checks','online_eligible','stages','cost')}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze','analyze'))
    parser.add_argument('--config', default='configs/evidence-context.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model')
    args = parser.parse_args()
    output = ROOT / args.output
    if args.mode == 'freeze':
        freeze(ROOT/args.config, output)
    else:
        if not args.model:
            parser.error('--model required for analyze')
        analyze(output, args.model)


if __name__ == '__main__':
    main()
