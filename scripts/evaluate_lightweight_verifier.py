import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.lightweight_verifier import evaluate, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    examples = [json.loads(line) for line in Path(args.input).read_text(encoding='utf-8').splitlines() if line.strip()]
    print(json.dumps(evaluate(load(Path(args.checkpoint)), examples)))


if __name__ == '__main__':
    main()
