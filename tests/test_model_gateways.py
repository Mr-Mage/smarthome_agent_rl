import unittest
from smarthome_agent_rl.model_gateways import gateway,deployments

class ModelGatewayTests(unittest.TestCase):
    def test_models_are_routed_through_disjoint_fixed_model_gateways(self):
        config={'actor_model':'base','generation':{'temperature':.7},'workflows':[{'id':0}],
                'variant_runtime':{'SFT9B':{'model':'adapter'},'Teacher':{'model':'teacher','endpoint':'http://external/v1'}},
                'model_gateway_ports':{'adapter':[20185],'teacher':[20189]}}
        workflow={'id':0,'actor_port':20000,'gateway_port':20181}
        rows=deployments(config,['G','SFT9B','Teacher'],workflow)
        self.assertEqual([r['gateway_port'] for r in rows],[20181,20185,20189])
        self.assertEqual([r['endpoint'] for r in rows],['http://127.0.0.1:20000/v1']*2+['http://external/v1'])
        self.assertEqual(gateway(config,{'id':16,'actor_id':0,'gateway_port':20181},'adapter'),20185)
        del config['model_gateway_ports']['adapter']
        with self.assertRaises(ValueError):deployments(config,['SFT9B'],workflow)
