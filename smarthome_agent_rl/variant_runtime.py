"""Explicit per-arm generation/model routing recorded in each episode contract."""
import copy
from smarthome_agent_rl.generation import generation_options

def runtime(config, variant, workflow):
    options = copy.deepcopy(config.get('variant_runtime',{}).get(variant,{}))
    if set(options)-{'generation','model','endpoint'}:
        raise ValueError('Unsupported variant runtime settings')
    generation = options.get('generation',config['generation'])
    generation_options({'generation':generation})
    return {'generation':generation, 'served_model':options.get('model',config['actor_model']),
            'model_endpoint':options.get('endpoint',f"http://127.0.0.1:{workflow['actor_port']}/v1")}
