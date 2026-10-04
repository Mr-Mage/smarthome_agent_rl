"""Execution slots share model services, never simulator state."""
from pathlib import Path
from urllib.parse import urlsplit


def external_judge(config):
    deployment = config.get('judge_deployment', 'local')
    if deployment not in ('local', 'external'):
        raise ValueError('judge_deployment must be local or external')
    if deployment == 'external':
        endpoint = config.get('judge_endpoint', '')
        parts = urlsplit(endpoint)
        if (parts.scheme not in ('http', 'https') or not parts.hostname or
                parts.path.rstrip('/') != '/v1' or parts.query or parts.fragment or
                parts.username or parts.password):
            raise ValueError('External judge requires an HTTP(S) /v1 endpoint without credentials')
        if config.get('judge_gpus') != []:
            raise ValueError('External judge must not reserve local GPUs')
    return deployment == 'external'


def judge_endpoint(config):
    return (config['judge_endpoint'].rstrip('/') if external_judge(config) else
            f"http://127.0.0.1:{config['judge_port']}/v1")


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
    ports += [config['embedding_port']]
    if not external_judge(config):
        ports.append(config['judge_port'])
    if len(ports) != len(set(ports)):
        raise ValueError('Execution simulator/model/gateway ports overlap')
    return slots


def episode_directory(run, item, variant):
    worker = item.get('variant_workflows', {}).get(variant, item['workflow'])
    base = Path(run) / f"seed{item['actor_seed']}" if 'actor_seed' in item else Path(run)
    return base / f'worker{worker}' / item['task']['id'] / variant / 'lightning'


def seeded_schedule(rows, variants, actor_seeds, actors=2):
    from smarthome_agent_rl.benchmark import schedule
    if not actor_seeds or len(actor_seeds) != len(set(actor_seeds)) or any(type(s) is not int for s in actor_seeds):
        raise ValueError('Distinct integer actor seeds required')
    if len(actor_seeds) == 1:
        return schedule(rows, variants, actors)
    return [{**item, 'actor_seed': seed} for ordinal, seed in enumerate(actor_seeds)
        for item in schedule(rows, variants if ordinal % 2 == 0 else list(reversed(variants)), actors)]
