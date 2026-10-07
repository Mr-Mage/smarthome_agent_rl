import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.lightweight_verifier import LightweightDecisionVerifier, save


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    examples = read_jsonl(args.input)
    model = LightweightDecisionVerifier()
    training = model.fit(examples, epochs=args.epochs)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    save(model, target)
    print(json.dumps({'checkpoint': str(target), **training}))


if __name__ == '__main__':
    main()
