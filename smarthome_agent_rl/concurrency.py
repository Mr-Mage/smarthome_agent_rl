"""Execution slots share model services, never simulator state."""
from pathlib import Path


def dispatch_items(items, policy='manifest'):
    """Reorder dispatch only; preserve frozen tasks, actor affinity and arm order."""
    if policy == 'manifest':
        return list(items)
    if policy != 'workflow-first':
        raise ValueError('Unknown dispatch policy')
    def priority(item):
        task = item['task']
        workflow = task['query_type'].startswith('qt4-')
        # Public category only: never consult final outcomes or case contents.
        return (0 if workflow and task['case'] == 'feasible' else 1 if workflow else 2,
                0 if task['query_type'] == 'qt4-1' else 1)
    return sorted(items, key=priority)


def execution_slots(config):
    count = config.get('slots_per_actor', 1)
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError('slots_per_actor must be an integer in [1, 32]')
    actors = config['workflows']
    if [actor['id'] for actor in actors] != list(range(len(actors))):
        raise ValueError('Actor ids must be contiguous from zero')
    slots = []
    for actor in actors:
        for ordinal in range(count):
            slot_id = actor['id'] * count + ordinal
            slots.append({**actor, 'actor_id': actor['id'], 'id': slot_id,
                'simulator_port': actor['simulator_port'] if count == 1 else
                    config.get('simulator_port_base', 21000) + slot_id})
    ports = [slot['simulator_port'] for slot in slots] + [
        actor[key] for actor in actors for key in ('actor_port', 'gateway_port')]
    ports += [config['judge_port'], config['embedding_port']]
    if len(ports) != len(set(ports)):
        raise ValueError('Execution simulator/model/gateway ports overlap')
    return slots


def episode_directory(run, item, variant):
    worker = item.get('variant_workflows', {}).get(variant, item['workflow'])
    return Path(run) / f'worker{worker}' / item['task']['id'] / variant / 'lightning'
