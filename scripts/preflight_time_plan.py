"""Replay an existing dev request with the finite metadata schema; no tool side effects."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.time_plan import TimePlan, OUTPUT_CONTRACT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    episode = ROOT / config['node_experiment']['preflight_episode']
    contract = json.loads((episode / 'contract.json').read_text())
    task = json.loads((ROOT / 'deps/SimuHome/data/benchmark' / contract['task_identity']['path']).read_text())
    plan = TimePlan()
    plan.initialize(task['query'], task['initial_home_config']['base_time'])
    request = copy.deepcopy(json.loads((episode / 'model_calls.json').read_text())[0]['request'])
    schema = request['response_format']
    for key in ('time_refs', 'time_dispositions', 'time_plan'):
        schema['json_schema']['schema']['properties'].pop(key, None)
        schema['json_schema']['schema']['required'] = [r for r in schema['json_schema']['schema']['required'] if r != key]
    request['response_format'] = plan.augment_schema(schema)
    request['messages'][-1]['content'] = plan.prompt()
    request['messages'][0]['content'] = request['messages'][0]['content'].split(
        '\n[STRUCTURED HARNESS OUTPUT CONTRACT]', 1)[0] + OUTPUT_CONTRACT
    url = f"http://127.0.0.1:{config['workflows'][0]['actor_port']}/v1/chat/completions"
    started = time.monotonic()
    with httpx.Client(trust_env=False, timeout=300) as client:
        response = client.post(url, json=request)
        response.raise_for_status()
    result = response.json()
    error = None
    try:
        body = json.loads(result['choices'][0]['message']['content'])
        plan.consume(body)
        if set(body) != {'thought', 'call'} or len(plan.table) != len(plan.rows):
            raise ValueError('Metadata/body mismatch')
    except (ValueError, KeyError, TypeError) as exc:
        error = str(exc)
    receipt = {'passed': error is None, 'error': error, 'request': request, 'response': result,
        'duration_seconds': time.monotonic() - started, 'table': plan.snapshot(),
        'scope': 'One frozen existing dev actor request; no tools executed; excluded from benchmark SR; tokens/cost retained'}
    (ROOT / args.output).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': receipt['passed'], 'error': error, 'actor_tokens': result.get('usage', {}).get('total_tokens')}))
    if error:
        raise RuntimeError('Finite time-plan schema preflight failed')
