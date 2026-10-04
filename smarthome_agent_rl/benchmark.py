"""Official benchmark selection and paired ordering; never rewrite episode contents."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

STRATA = [(qt, case) for qt in ('qt1', 'qt2', 'qt3', 'qt4-1', 'qt4-2', 'qt4-3')
          for case in ('feasible', 'infeasible')]
KNOWN_EXPOSURES = {'qt1_feasible_seed_1', 'qt3_feasible_seed_1',
                   'qt3_feasible_seed_2', 'qt3_feasible_seed_3'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_failure_kind(error, calls=()):
    if not error or error['type'] != 'AgentExecutionError':
        return None
    message = error['message'].lower()
    for term, kind in [('explicit finish', 'step_limit'), ('consecutive failures', 'output_rejections'),
                       ('polling budget', 'workflow_poll_limit')]:
        if term in message:
            return kind
    if calls and calls[-1]['status'] in (400, 413) and any(term in message for term in (
            'maximum context length', 'reduce the length of the input messages', 'parameter=input_tokens')):
        return 'actor_context_limit'
    return None


def select(benchmark, *, seed=20261004, dev_count=10, final_count=16, exposed=()):
    groups = defaultdict(list)
    exposed = KNOWN_EXPOSURES | set(exposed)
    for path in sorted(Path(benchmark).glob('*.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))['meta']
        stratum = (meta['query_type'], meta['case'])
        if stratum not in STRATA:
            raise ValueError(f'Unknown official category: {stratum}')
        groups[stratum].append({'id': path.stem, 'path': path.name, 'sha256': digest(path),
            'query_type': stratum[0], 'case': stratum[1], 'seed': meta['seed']})
    rng = random.Random(seed)
    dev, final = [], []
    for stratum in STRATA:
        candidates = groups[stratum][:]
        rng.shuffle(candidates)
        eligible_final = [r for r in candidates if r['id'] not in exposed]
        if len(eligible_final) < final_count or len(candidates) < dev_count + final_count:
            raise ValueError(f'Not enough independent official cases: {stratum}')
        final_rows = eligible_final[:final_count]
        final_ids = {r['id'] for r in final_rows}
        dev_rows = [r for r in candidates if r['id'] not in final_ids][:dev_count]
        final.extend(final_rows)
        dev.extend(dev_rows)
    ids = [r['id'] for r in dev + final]
    hashes = [r['sha256'] for r in dev + final]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError('Duplicate official task or identical content across selected splits')
    return {'seed': seed, 'source_counts': {':'.join(k): len(v) for k, v in groups.items()},
            'excluded_from_final': sorted(exposed), 'dev': dev, 'final': final,
            'smoke': []}


def schedule(rows, variants, actors=2):
    """Each task stays on one actor; adjacent tasks alternate variant order."""
    return [{'task': row, 'workflow': i % actors,
             'variants': list(variants if (i // actors) % 2 == 0 else reversed(variants))}
            for i, row in enumerate(rows)]


def freeze(benchmark, output, *, upstream_commit, exposed=()):
    selection = select(benchmark, exposed=exposed)
    selection['smoke'] = [r for qt, case in STRATA for r in [
        row for row in selection['dev'] if (row['query_type'], row['case']) == (qt, case)][:2]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split in ('dev', 'final', 'smoke'):
        target = output / f'{split}.json'
        if target.exists():
            raise FileExistsError('Frozen manifests cannot be overwritten')
        target.write_text(json.dumps({'schema': 'official-benchmark-manifest-v1', 'split': split,
            'sampling_seed': selection['seed'], 'simuhome_commit': upstream_commit,
            'tasks': selection[split]}, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'selection.json').write_text(json.dumps({k: v for k, v in selection.items()
        if k not in ('dev', 'final', 'smoke')}, indent=2), encoding='utf-8')
    return selection
