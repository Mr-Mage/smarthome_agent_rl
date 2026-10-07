"""Frozen SimuHome tasks with separate public context and native evaluation.

The author runner still owns reset, virtual time, tool collection and scoring.
This adapter enforces its input boundary; it never invents prompts or task goals.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

from .actors import ReActActor


class SimuHomeAdapter:
    def __init__(self, source, manifest):
        self.source = Path(source)
        self.manifest = copy.deepcopy(manifest)
        self.tasks = {r['id']: copy.deepcopy(r) for r in manifest['tasks']}
        if len(self.tasks) != len(manifest['tasks']):
            raise ValueError('Duplicate frozen task identity')

    def task_ids(self):
        return list(self.tasks)

    def task_path(self, task_id):
        row = self.tasks[task_id]
        path = self.source / row['path']
        if path.name != row['path'] or not path.resolve().is_relative_to(self.source.resolve()):
            raise ValueError('Frozen task path escaped benchmark source')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Frozen official case identity mismatch')
        return path

    def _load(self, task_id):
        episode = json.loads(self.task_path(task_id).read_text(encoding='utf-8'))
        row = self.tasks[task_id]
        if any(episode['meta'].get(k) != row[k] for k in ('query_type', 'case', 'seed') if k in row):
            raise ValueError('Official metadata differs from frozen manifest')
        return episode

    def category(self, task_id):
        row = self.tasks[task_id]
        return row['query_type'] + ':' + row['case']

    def public_input(self, task_id):
        episode = self._load(task_id)
        if not isinstance(episode['query'], str) or not episode['query']:
            raise ValueError('Official query must be a nonempty string')
        return {'query': episode['query'], 'user_location': copy.deepcopy(episode.get('user_location')),
                'current_time': episode['initial_home_config'].get('base_time')}

    def bind_agent(self, task_id, agent):
        return PublicEpisodeAgent(agent, self.public_input(task_id))

    def score(self, task_id, official_result):
        row = self.tasks[task_id]
        public = self.public_input(task_id)
        if (official_result.get('query') != public['query'] or
                any(official_result.get(k) != row[k] for k in ('query_type', 'case'))):
            raise ValueError('Native evaluation is not for this frozen episode')
        score = official_result['evaluation_result']['score']
        if type(score) not in (int, float) or not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError('Native score is invalid')
        return {'official_score': score, 'success': score == 1, 'evaluator_error': score == -1,
                'source': 'unchanged official SimuHome evaluator; not runtime task-state verification'}


class PublicEpisodeAgent:
    def __init__(self, agent, public_input):
        self.wrapped, self.public = agent, copy.deepcopy(public_input)
        self.actor = ReActActor(agent)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def run(self, query, *, user_location=None, current_time=None):
        actual = {'query': query, 'user_location': user_location, 'current_time': current_time}
        if actual != self.public:
            raise ValueError('Actor context drift or evaluator leakage')
        return self.actor.invoke(actual)
