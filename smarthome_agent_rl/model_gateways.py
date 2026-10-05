"""Use fixed-model Lightning gateways so upstream routing cannot rewrite an arm."""
from smarthome_agent_rl.variant_runtime import runtime

def gateway(config,workflow,model):
    if model==config['actor_model']:return workflow['gateway_port']
    ports=config.get('model_gateway_ports',{}).get(model)
    if not isinstance(ports,list) or len(ports)!=len(config['workflows']):
        raise ValueError('Explicit gateway ports required for every alternate model')
    return ports[workflow.get('actor_id',workflow['id'])]

def deployments(config,variants,workflow):
    models={}
    for variant in variants:
        r=runtime(config,variant,workflow)
        model=r['served_model']
        row={'model':model,'endpoint':r['model_endpoint'],'gateway_port':gateway(config,workflow,model)}
        if model in models and models[model]!=row:raise ValueError('A model has inconsistent per-arm endpoints')
        models[model]=row
    return list(models.values())
