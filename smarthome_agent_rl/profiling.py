"""Process-local timings without changing upstream agent or evaluator behavior."""
from functools import wraps
import time


class PhaseProfile:
    def __init__(self, save, clock=time.monotonic):
        self.clock, self.save = clock, save
        self.origin = clock()
        self.spans, self.phase = [], 'setup'

    def wrap(self, function, kind, label=None):
        @wraps(function)
        def measured(*args, **kwargs):
            start, phase = self.clock(), self.phase
            outcome = 'error'
            try:
                value = function(*args, **kwargs)
                outcome = 'returned'
                return value
            finally:
                self.spans.append({'kind': kind, 'label': label or function.__name__, 'phase': phase,
                    'start_seconds': start - self.origin, 'end_seconds': self.clock() - self.origin,
                    'outcome': outcome})
        return measured

    def agent(self, agent, react_module):
        run = agent.run
        def measured(*args, **kwargs):
            self.phase = 'agent'
            original = react_module.run_tool
            if hasattr(agent, 'executor'):
                dispatch = agent.executor.dispatch
                agent.executor.dispatch = self.wrap(dispatch, 'tool_dispatch')
            else:
                react_module.run_tool = self.wrap(original, 'tool_dispatch')
            try:
                return self.wrap(run, 'agent')(*args, **kwargs)
            finally:
                self.phase = 'post_agent'
                if hasattr(agent, 'executor'):
                    agent.executor.dispatch = dispatch
                react_module.run_tool = original
        agent.run = measured
        return agent

    def flush(self, episode_seconds, episode_start_seconds=0):
        self.save('phase_profile.json', {'version': 1, 'episode_seconds': episode_seconds,
            'episode_start_seconds': episode_start_seconds,
            'spans': self.spans,
            'definitions': {'agent': 'run duration, including tools, model HTTP and local waiting',
                'post_agent': 'after agent returns, including official evaluation; not user-visible latency',
                'tool_dispatch': 'inclusive of retrieval; nested spans must not be added twice',
                'http': 'includes network, Gateway, queue and inference; not pure GPU compute'}})


class TimedTime:
    """Module-local time proxy: avoids replacing global time.sleep."""
    def __init__(self, original, profile):
        self.original = original
        self.sleep = profile.wrap(original.sleep, 'waiting', 'sleep')

    def __getattr__(self, name):
        return getattr(self.original, name)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cost_summary(records):
    successes = sum(r['success'] for r in records)
    result = {'actor_tokens_per_success': sum(r['actor_tokens'] for r in records) / successes if successes else None,
        'judge_tokens_per_success': sum(r['judge_tokens'] for r in records) / successes if successes else None,
        'failed_actor_tokens': sum(r['actor_tokens'] for r in records if not r['success'])}
    for key in ('duration_seconds', 'agent_seconds', 'post_agent_seconds', 'evaluator_seconds',
                'tool_seconds', 'agent_wait_seconds', 'evaluation_wait_seconds', 'retrieval_seconds',
                'actor_latency', 'judge_latency', 'evaluation_client_seconds', 'agent_residual_seconds'):
        values = [r[key] for r in records if r.get(key) is not None]
        result[key] = {'observed': len(values), 'total': sum(values) if values else None,
            'p50': percentile(values, .5), 'p95': percentile(values, .95)}
    return result
