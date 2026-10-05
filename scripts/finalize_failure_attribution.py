"""Join complete manual labels to audited episodes; do not adjust official scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = ROOT / args.run
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    config = read(ROOT / 'configs/failure-attribution.json')
    labels_path = ROOT / 'configs/failure-attribution-labels.json'
    review = read(labels_path)
    cards = read(run / 'manual-review-cards.json')
    labels = review['labels']
    indexed = {r['task_id']: r for r in cards}
    assert len(labels) == len({r['task_id'] for r in labels}) == len(cards) == 31
    assert set(indexed) == {r['task_id'] for r in labels}
    rows = []
    for label in labels:
        card = indexed[label['task_id']]
        assert label['primary'] in config['manual_categories']
        assert label['observation'] and label['limitations']
        assert set(label['turns']).issubset({r['turn'] for r in card['actions']})
        rows.append({**label, 'episode': card['path'], 'source_sha256': card['source_sha256'],
            'official_result_sha256': card['official_result_sha256'], 'official_success': False})
    selected = review['selected_mechanism']
    tasks = sorted(r['task_id'] for r in rows if r['mechanism'] == selected)
    minimum = config['n28_admission']['minimum_independent_original_G_tasks']
    selection = {'node': 'N27', 'n28_admitted': len(tasks) >= minimum,
        'mechanism': selected, 'independent_original_G_tasks': tasks,
        'count': len(tasks), 'required_minimum': minimum,
        'intervention_boundary': review['intervention_boundary'],
        'difference_from_previous': 'N10 verifies effects only after proposed commands; N15 rejects poweroff Start. Neither addresses omission of Start after successful On/mode changes. Here only public semantics are explained; no postchecks or added precondition checks.',
        'not_a_success_claim': 'Manual diagnosis on exposed development tasks; mixed feasible/infeasible cases, additional timing/coverage defects remain. No corrected official denominator and no new SR.',
        'n29': 'Conditional on N28 frozen paired mechanism/SR/safety/cost gates, never automatic'}
    output = {'source_cards_sha256': digest(run/'manual-review-cards.json'),
        'labels_sha256': digest(labels_path), 'primary_counts': dict(Counter(r['primary'] for r in rows)),
        'mechanism_counts': dict(Counter(r['mechanism'] for r in rows)), 'reviewed': rows,
        'limits': 'Single predeclared seed42; primary observations are descriptive, not mutually isolated causes. All failures kept; no evaluator details supplied to actor.'}
    for name, value in [('manual-attribution.json', output), ('selection.json', selection)]:
        path = run/name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'primary_counts': output['primary_counts'], 'selection': selection}, ensure_ascii=False))


if __name__ == '__main__':
    main()
