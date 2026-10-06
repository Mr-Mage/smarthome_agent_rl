import json
from pathlib import Path
import unittest
from urllib.parse import urlparse


class SeedReplicationBudgetTests(unittest.TestCase):
    def test_two_seeds_share_four_gpus_and64_slots_with_identical_experiment_rules(self):
        root=Path(__file__).resolve().parents[1]
        base=json.loads((root/'configs/homebench-format-ablation.json').read_text())
        replicas=[json.loads((root/f'configs/homebench-format-seed{s}.json').read_text()) for s in (43,44)]
        resources=[]
        ports=[]
        slots=0
        for seed,config in zip((43,44),replicas):
            self.assertEqual(config['model_seed'],seed)
            self.assertEqual({k:v for k,v in config.items() if k not in ('model_seed','actors','freeze_note')},
                             {k:v for k,v in base.items() if k not in ('model_seed','actors','freeze_note')})
            for actor in config['actors']:
                endpoint=urlparse(actor['endpoint'])
                self.assertEqual(endpoint.hostname,'127.0.0.1')
                resources.append(actor['gpu'])
                ports.append(endpoint.port)
            slots+=len(config['actors'])*config['slots_per_actor']
        self.assertEqual(sorted(resources),[0,1,2,3])
        self.assertEqual(len(set(ports)),4)
        self.assertEqual(slots,64)


if __name__=='__main__':
    unittest.main()
