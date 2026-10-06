"""Export an existing evidence-linked episode review; never run a model or a tool."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.episode_review import markdown, review_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', help='Directory with original audit, contract, summary and model calls')
    parser.add_argument('--manifest', help='Frozen artifact or receipt SHA manifest; required for authentication')
    parser.add_argument('--output', help='New directory under outputs/ or runs/; omitted prints JSON')
    args = parser.parse_args()
    report = review_episode(args.episode, args.manifest)
    if args.output:
        output = Path(args.output).resolve()
        if not any(output.is_relative_to(ROOT/name) for name in ('runs','outputs')):
            parser.error('Review exports belong in runs/ or outputs/')
        output.mkdir(parents=True, exist_ok=False)
        (output/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (output/'review.md').write_text(markdown(report), encoding='utf-8')
        print(json.dumps({'output': str(output), 'task_id': report['task_id'], 'integrity': report['integrity']}))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
