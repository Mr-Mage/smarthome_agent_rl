"""Offline project gates: stdlib core CI, or full tests in the existing server venv."""
import argparse
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.concurrency import execution_slots, external_judge

CORE = ('test_action_state.py', 'test_task_spec.py', 'test_recovery.py', 'test_concurrency.py',
        'test_node_selection.py', 'test_paired_stats.py', 'test_evidence_context.py', 'test_episode_review.py')


def check_release():
    config = json.loads((ROOT/'configs/harness-release.json').read_text(encoding='utf-8'))
    assert len(execution_slots(config)) == 64 and external_judge(config)
    assert [a['gpus'] for a in config['workflows']] == [[0],[1],[2],[3]]
    assert config['variants_dev'] == ['B0','G'] and config['variants_final'] == []
    policy = config['variant_policies']['G']
    assert policy == {'verify':False,'verification_version':1,'context_version':0,
        'dead_front':False,'workflow_all_devices':False,'time_plan':False,
        'identifier_binding':False,'start_semantics':False,'recovery':False,
        'task_spec':False,'evidence_context':False}
    stages = config['node_experiment']['stages']
    assert len(stages) == 1 and stages[0]['manifest'] == 'configs/sft-pilot/calibration.json'
    tasks = json.loads((ROOT/stages[0]['manifest']).read_text())['tasks']
    assert len(tasks) == 12 and len({(t['query_type'],t['case']) for t in tasks}) == 12
    assert config['generation'] == json.loads((ROOT/'configs/task-spec.json').read_text())['generation']
    checked = 0
    for name in ('README.md','PLAN.md','docs/项目架构.md','docs/nodes/N35.md','docs/nodes/N36.md'):
        path = ROOT/name
        if not path.exists():
            raise FileNotFoundError(name)
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            if not (path.parent/target.split('#')[0]).exists():
                raise FileNotFoundError(f'{name}: {target}')
            checked += 1
    return {'release_config':True,'actors':4,'slots':64,'exposed_calibration_tasks':12,'local_links_checked':checked}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Use existing SimuHome/venv; no install or network')
    args = parser.parse_args()
    checks = check_release()
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT/'tests')) if args.full else unittest.TestSuite(
        loader.discover(str(ROOT/'tests'),pattern=name) for name in CORE)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({**checks,'mode':'full' if args.full else 'stdlib-core',
        'tests':result.testsRun,'skipped':len(result.skipped),'passed':result.wasSuccessful()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
