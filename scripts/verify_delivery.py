"""Final smoke chain acceptance, not candidate selection or holdout validation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.episode_review import review_episode, markdown
from smarthome_agent_rl.concurrency import episode_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = ROOT/args.run
    stage = run/'smoke'
    report = json.loads((stage/'report.json').read_text())
    protocol = json.loads((stage/'protocol.json').read_text())
    config = protocol['config']
    if config['node_experiment']['node'] != 'N36' or not report['verified']:
        raise ValueError('Expected verified N36 smoke stage')
    if protocol['expected_episodes'] != 24 or protocol['variants'] != ['B0','G']:
        raise ValueError('Acceptance differs from frozen tasks/arms')
    reviews = []
    for item in protocol['schedule']:
        for variant in item['variants']:
            folder = episode_directory(stage,item,variant)
            summary = json.loads((folder/'summary.json').read_text())
            if summary['infrastructure_error'] or summary['official_score'] == -1:
                raise ValueError('Infrastructure failure invalidates the acceptance stage')
            if variant == 'G':
                review = review_episode(folder,stage/'artifact_manifest.json')
                target = run/'reviews'/summary['task_id']
                target.mkdir(parents=True,exist_ok=False)
                (target/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                (target/'review.md').write_text(markdown(review),encoding='utf-8')
                reviews.append(str(target.relative_to(ROOT)))
    if len(reviews) != 12 or any(report['arms'][v]['episodes'] != 12 for v in ('B0','G')):
        raise ValueError('Incomplete acceptance coverage')
    selection = {'node':'N36','chain_accepted':True,'winner':'G','new_candidate_selected':False,
        'episodes':24,'reviews':reviews,'verified':True,
        'scope':'Already exposed calibration12, one seed; no new formal SR claim. '
                'Ordinary task failures retained. Current G unchanged.'}
    (run/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(selection))


if __name__ == '__main__':
    main()
