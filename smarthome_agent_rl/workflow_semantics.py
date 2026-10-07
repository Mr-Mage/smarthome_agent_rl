"""Source-bound public workflow facts; never a user-goal oracle."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / 'configs/public-workflow-semantics.json'


def load_workflow_semantics(source_root=None, rules_path=RULES):
    source_root = Path(source_root or ROOT / 'deps/SimuHome').resolve()
    rules = json.loads(Path(rules_path).read_text(encoding='utf-8'))
    for relative, expected in rules['source_sha256'].items():
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Public workflow source differs: ' + relative)
    schemas = ast.parse((source_root / 'src/simulator/api/schemas.py').read_text(encoding='utf-8'))
    step = next(n for n in schemas.body if isinstance(n, ast.ClassDef) and n.name == 'ScheduleWorkflowStep')
    fields = [n.target.id for n in step.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
    if fields != rules['facts']['step_fields']:
        raise ValueError('Workflow step fields differ from the public declaration')
    return copy.deepcopy(rules)


def workflow_context(action, rules):
    if action.get('tool') != 'schedule_workflow':
        return None
    return {'schema': 'public-workflow-execution-semantics-v1',
            'source_sha256': copy.deepcopy(rules['source_sha256']),
            'facts': copy.deepcopy(rules['facts']),
            'declared_start_time': action.get('start_time'),
            'declared_steps': len(action.get('steps', [])),
            'scope': 'Execution API facts only; no predicted device state, validated user goal, '
                     'guaranteed dispatch time or instruction to cancel/replay.'}
