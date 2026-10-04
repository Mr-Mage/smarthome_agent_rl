"""Conservative metadata/hash inventory; no unused task text or outcomes exported."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import KNOWN_EXPOSURES, digest


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def inventory(root):
    benchmark = root / 'deps/SimuHome/data/benchmark'
    tasks, query_hashes, template_hashes = {}, {}, {}
    for path in sorted(benchmark.glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        meta = data['meta']
        tasks[path.stem] = {'id': path.stem, 'path': path.name, 'sha256': digest(path),
            'query_type': meta['query_type'], 'case': meta['case']}
        query = data['query'].lower().strip()
        query_hashes[path.stem] = hashlib.sha256(query.encode()).hexdigest()
        normalized = re.sub(r'\d+(?:\.\d+)?', '<number>', query)
        template_hashes[path.stem] = hashlib.sha256(normalized.encode()).hexdigest()
    exposed, evidence = set(KNOWN_EXPOSURES) & set(tasks), []
    # Include incomplete rounds and conservative config selections, not just accepted reports.
    paths = list((root / 'runs').rglob('protocol.json')) + list((root / 'runs').rglob('summary.json'))
    paths += list((root / 'configs').rglob('*.json'))
    ids = re.compile(r'qt(?:[123]|4-[123])_(?:in)?feasible_seed_\d+')
    for path in sorted(set(paths)):
        # Source copies/config snapshots duplicate evidence but are harmless conservative exposure.
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError):
            continue
        found = {name for text in strings(value) for name in ids.findall(text) if name in tasks}
        if found:
            exposed.update(found)
            evidence.append({'path': str(path.relative_to(root)), 'sha256': digest(path), 'matched_tasks': len(found)})
    content_exposed = {tasks[name]['sha256'] for name in exposed}
    query_exposed = {query_hashes[name] for name in exposed}
    template_exposed = {template_hashes[name] for name in exposed}
    remaining = [row for name, row in tasks.items() if name not in exposed and row['sha256'] not in content_exposed
                 and query_hashes[name] not in query_exposed]
    counts = Counter(row['query_type'] + ':' + row['case'] for row in remaining)
    related = [row['id'] for row in remaining if template_hashes[row['id']] in template_exposed]
    return {'purpose': 'N19 metadata/hash exposure audit; no unused query text or results exported',
        'official_tasks': len(tasks), 'conservative_exposed_tasks': len(exposed),
        'remaining_tasks': len(remaining), 'remaining_by_category': dict(sorted(counts.items())),
        'remaining_manifest': remaining, 'exposed_ids': sorted(exposed), 'historical_evidence': evidence,
        'numeric_normalized_query_overlap': len(related),
        'exact_query_duplicate_exclusions': sum(name not in exposed and query_hashes[name] in query_exposed for name in tasks),
        'limitations': 'Numeric-normalized query overlap is a narrow near-duplicate screen, not proof of independent templates. All tasks originate from the same official generator; unseen queries are hashed algorithmically, not exported or manually reviewed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Never overwrite an exposure inventory')
    result = inventory(ROOT)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('official_tasks', 'conservative_exposed_tasks', 'remaining_tasks',
                                          'remaining_by_category', 'numeric_normalized_query_overlap')}))
