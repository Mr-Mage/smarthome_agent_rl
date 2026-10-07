"""Public action-intention claims and bounded native context, without arbitration."""
import copy
import hashlib
import json

from ..device_contract import _get_path
from .conflicts import has_reference
from .events import EventEpisodeRuntime
from .mutation import resolve


def argument_references_only(value):
    if isinstance(value, str):
        return not value.startswith('$') or value.startswith('$args.')
    if isinstance(value, dict):
        return all(argument_references_only(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(argument_references_only(v) for v in value)
    return True


def fixed_claims(adapter, steps, structures):
    """Compile only fixed argument/literal targets from already observed schemas.

    Last declared write wins within a workflow. An unknown later operation can
    overwrite earlier expectations, so it drops all prior claims for its device.
    Windows are supplied by Scheduler; no persistent occupancy is inferred.
    """
    claims, provenance, uncovered = {}, {}, []
    for index, step in enumerate(steps):
        args, tool = step['args'], step['tool']
        device = args['device_id']
        try:
            structure = structures[device]
            if structure.get('device_id') != device:
                raise ValueError('PUBLIC_STRUCTURE_IDENTITY_MISMATCH')
            contract, name, action_args = adapter.build(tool, args, structure, future=True)
            function = contract.functions.get(name)
            if function is None or not function.postconditions:
                raise ValueError('PUBLIC_POSTCONDITION_UNCOVERED')
            pending = []
            for condition in function.postconditions:
                path = condition['path']
                if condition.get('op', 'eq') != 'eq' or not _get_path(structure, path)[0]:
                    raise ValueError('NON_EQUALITY_OR_UNOBSERVED_TARGET_PATH')
                if not argument_references_only(condition['value']):
                    raise ValueError('NON_ARGUMENT_FUTURE_REFERENCE')
                # Empty state rejects $state references; only $args is accepted.
                value = resolve(condition['value'], action={'args': action_args}, state={})
                if has_reference(value):
                    raise ValueError('UNRESOLVED_TARGET_REFERENCE')
                tolerance = condition.get('tolerance', 0)
                row = {'device_id': device, 'path': path, 'value': value, 'tolerance': tolerance}
                json.dumps(row, allow_nan=False)
                pending.append(row)
            for row in pending:
                key = (device, row['path'])
                claims[key] = row
                provenance[key] = {'device_id': device, 'path': row['path'], 'step': index,
                    'tool': tool, 'arguments': copy.deepcopy(args),
                    'public_structure_sha256': hashlib.sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest()}
        except (KeyError, TypeError, ValueError) as exc:
            for key in [key for key in claims if key[0] == device]:
                del claims[key]
                del provenance[key]
            uncovered.append({'step': index, 'device_id': device, 'reason': str(exc)})
    keys = sorted(claims)
    return {'claims': [claims[key] for key in keys],
            'provenance': [provenance[key] for key in keys], 'uncovered': uncovered}


class ResourceEpisodeRuntime(EventEpisodeRuntime):
    def __init__(self, *args, **kwargs):
        save = kwargs.pop('save', None)
        def save_resource_snapshot(name, data):
            data['resource_conflicts'] = {
                'historical': self.store.list('conflict'),
                'current': self.manager.detect_conflicts(self.task.task_id, self.task.user_id) if self.task else None}
            if save:
                save(name, data)
        kwargs['save'] = save_resource_snapshot if save else None
        super().__init__(*args, **kwargs)

    def _intent(self, steps):
        payload = super()._intent(steps)
        payload['resource_intention'] = fixed_claims(self.adapter, steps, self.structures)
        return payload

    def _resource_claims(self, payload):
        return payload['resource_intention']['claims']

    def actor_context(self, *, max_jobs=8, max_claims=8, max_conflicts=4, max_uncovered=8, max_chars=12000):
        if self.task is None:
            return None
        jobs = [j for j in self.store.list('job') if j['task_id'] == self.task.task_id]
        if not jobs:
            return None
        # Small deterministic projection; raw structure/receipts and user goals
        # stay in their original public history, not this diagnostic summary.
        jobs.sort(key=lambda j: (j['target_time'], j['job_id']))
        workflows = {w['workflow_id']: w for w in self.store.list('workflow')}
        live = self.manager.detect_conflicts(self.task.task_id, self.task.user_id)
        rows = []
        for job in jobs[:max_jobs]:
            claims = job['resource_claims']
            rows.append({'job_id': job['job_id'],
                'native_workflow_id': workflows[job['workflow_id']]['device_workflow_id'],
                'status': job['status'], 'registration_uncertain': job.get('registration_uncertain', False),
                'target_time': job['target_time'], 'window_end': job['target_time'] + job['tolerance'],
                'claims': claims[:max_claims], 'claims_total': len(claims),
                'claims_truncated': len(claims) > max_claims,
                'uncovered_steps': job['payload']['resource_intention']['uncovered'][:max_uncovered],
                'uncovered_steps_total': len(job['payload']['resource_intention']['uncovered']),
                'uncovered_steps_truncated': len(job['payload']['resource_intention']['uncovered']) > max_uncovered})
        context = {'schema': 'public-native-resource-context-v1',
            'policy': 'Report only. No automatic cancellation, replay or arbitration. '
                      'Only change plans when justified by the public user request and observations.',
            'semantics': 'Claims are actor-selected fixed targets in closed verification windows, '
                         'not current state, persistent resource occupancy or validated user goals. '
                         'No reported conflict does not prove compatibility; uncovered actions remain unknown. '
                         'Job DONE does not complete the user Task.',
            'task_status': self.manager.get(self.task.task_id, self.task.user_id).status,
            'jobs': rows, 'jobs_total': len(jobs), 'jobs_truncated': len(jobs) > max_jobs,
            'conflicts': live['conflicts'][:max_conflicts], 'conflicts_total': len(live['conflicts']),
            'conflicts_truncated': len(live['conflicts']) > max_conflicts,
            'uncovered_overlaps': live['uncovered'][:max_conflicts],
            'uncovered_overlaps_total': len(live['uncovered']),
            'uncovered_overlaps_truncated': len(live['uncovered']) > max_conflicts,
            'size_truncated': False}
        # Large literal targets must not silently overflow the actor's history.
        # Retain exact totals and remove whole evidence rows, never alter values.
        for key in ('uncovered_overlaps', 'conflicts', 'jobs'):
            while context[key] and len(json.dumps(context, ensure_ascii=False, sort_keys=True)) > max_chars:
                context[key].pop()
                context[key + '_truncated'] = True
                context['size_truncated'] = True
        return copy.deepcopy(context)


class RuntimeContextProvider:
    def __init__(self, inner, runtime):
        self.inner, self.runtime = inner, runtime

    def generate(self, messages, response_format=None):
        from src.agents.types import ChatMessage
        runtime = self.runtime()
        context = runtime.actor_context() if runtime else None
        if context is None:
            return self.inner.generate(messages, response_format=response_format)
        content = 'PUBLIC NATIVE JOB / RESOURCE CONTEXT\n' + json.dumps(context, ensure_ascii=False, sort_keys=True)
        # Append a fresh projection without mutating or duplicating old history.
        converted = [*messages, ChatMessage(role='user', content=content)]
        runtime._event('runtime_context_exposed', context=context,
                       content_sha256=hashlib.sha256(content.encode()).hexdigest())
        return self.inner.generate(converted, response_format=response_format)
