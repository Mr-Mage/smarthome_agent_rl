"""Integration against existing, unmodified SimuHome domain implementations."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    from src.simulator.domain.devices.dimmable_light import DimmableLight
    from src.simulator.domain.devices.air_conditioner import AirConditioner
    from src.simulator.domain.clusters.temperature_control import TemperatureControlCluster, FeatureFlag
except ModuleNotFoundError as exc:
    if exc.name != 'src':
        raise
    DimmableLight = None

from smarthome_agent_rl.execution.harness import RuntimeExecutor
from smarthome_agent_rl.execution.simuhome_contract import SimuHomeContractAdapter
from smarthome_agent_rl.execution.mutation import check_postconditions
from smarthome_agent_rl.guard import ToolGuard, command_contracts, public_power_rules

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(DimmableLight is None, 'requires existing SimuHome server environment')
class RealSimuHomeRuntimeTests(unittest.TestCase):
    def setUp(self):
        signatures, _ = command_contracts(ROOT / 'deps/SimuHome/src/simulator/domain/clusters')
        power, _ = public_power_rules(ROOT / 'deps/SimuHome/src/simulator/domain/devices')
        self.adapter = SimuHomeContractAdapter(signatures, power)
        schemas = {tool: {'properties': {}, 'additionalProperties': True}
                   for tool in ('execute_command', 'write_attribute', 'get_device_structure')}
        self.guard = ToolGuard(schemas, signatures, power)
        self.device = DimmableLight('lamp')
        self.calls = []
        self.timeout = False

    def dispatch(self, tool, args):
        self.calls.append((tool, copy.deepcopy(args)))
        if tool == 'get_device_structure':
            return {'status': {'code': 200}, 'data': json.loads(json.dumps(self.device.get_structure()))}
        if tool == 'execute_command':
            result = self.device.execute_command(args['endpoint_id'], args['cluster_id'], args['command_id'], **args['args'])
        else:
            result = self.device.write_attribute(args['endpoint_id'], args['cluster_id'], args['attribute_id'], args['value'])
        if self.timeout and result.success:
            raise TimeoutError('test fault after actual SimuHome mutation')
        return {'status': {'code': 200 if result.success else 400}, 'data': result.data or {},
                'error': None if result.success else {'type': result.error_code.value, 'message': result.error_message}}

    def executor(self):
        return RuntimeExecutor(guard=self.guard, adapter=self.adapter, dispatch=self.dispatch,
            invocation_factory=SimpleNamespace)

    def action(self, **updates):
        return {**{'device_id': self.device.device_id, 'endpoint_id': 1, 'cluster_id': 'LevelControl',
                   'command_id': 'MoveToLevelWithOnOff', 'args': {'Level': 100}}, **updates}

    def test_real_light_mutation_and_readback(self):
        result = self.executor().execute('execute_command', self.action())
        self.assertEqual(result['harness_verification']['status'], 'VERIFIED_SUCCESS')
        self.assertEqual(self.device.get_attribute(1, 'LevelControl', 'CurrentLevel'), 100)

    def test_timeout_after_actual_domain_commit_does_not_replay(self):
        self.timeout = True
        result = self.executor().execute('execute_command', self.action())
        self.assertEqual(result['harness_verification']['status'], 'VERIFIED_SUCCESS')
        self.assertEqual(sum(tool == 'execute_command' for tool, _ in self.calls), 1)

    def test_suppressed_light_command_is_not_completed(self):
        self.device.execute_command(1, 'OnOff', 'Off')
        result = self.executor().execute('execute_command', self.action(command_id='MoveToLevel'))
        self.assertTrue(result['data']['suppressed'])
        self.assertEqual(result['harness_verification']['status'], 'VERIFIED_FAILURE')

    def test_live_transition_is_unverified_and_does_not_advance_virtual_time(self):
        self.device.execute_command(1, 'LevelControl', 'MoveToLevelWithOnOff', Level=20)
        result = self.executor().execute('execute_command', self.action(args={'Level': 100, 'TransitionTime': 20}))
        self.assertEqual(result['harness_verification']['reason'], 'PUBLIC_TRANSITION_PENDING')
        self.assertEqual(result['harness_verification']['status'], 'UNVERIFIED')
        self.assertEqual(self.device.get_attribute(1, 'LevelControl', 'CurrentLevel'), 20)

    def test_actual_invalid_range_and_readonly_write_are_rejected(self):
        executor = self.executor()
        rejected = executor.execute('execute_command', self.action(args={'Level': 999}))
        self.assertEqual(rejected['status']['code'], 422)
        readonly = executor.execute('write_attribute', {k: v for k, v in self.action(
            attribute_id='CurrentLevel', value=100).items() if k not in ('command_id', 'args')})
        self.assertEqual(readonly['error']['reason_code'], 'READONLY_FUNCTION')
        self.assertFalse(any(tool in ('execute_command', 'write_attribute') for tool, _ in self.calls))

    def test_actual_ac_power_and_deadband_are_checked_before_dispatch(self):
        self.device = AirConditioner('ac')
        executor = self.executor()
        args = {'device_id': 'ac', 'endpoint_id': 1, 'cluster_id': 'Thermostat',
                'attribute_id': 'OccupiedCoolingSetpoint', 'value': 3000}
        self.device.execute_command(1, 'OnOff', 'Off')
        self.assertEqual(executor.execute('write_attribute', args)['status']['code'], 422)
        self.device.execute_command(1, 'OnOff', 'On')
        heating = self.device.get_attribute(1, 'Thermostat', 'OccupiedHeatingSetpoint')
        self.assertEqual(executor.execute('write_attribute', {**args, 'value': heating})['status']['code'], 422)
        result = executor.execute('write_attribute', args)
        self.assertEqual(result['harness_verification']['status'], 'VERIFIED_SUCCESS')

    def test_temperature_numeric_precedence_and_explicit_null_match_domain(self):
        cluster = TemperatureControlCluster(features=FeatureFlag.TN | FeatureFlag.TL)
        self.device.add_cluster(1, cluster)
        for values in ({'target_temperature': 2500, 'target_temperature_level': 1},
                       {'target_temperature': None, 'target_temperature_level': 2}):
            args = self.action(cluster_id='TemperatureControl', command_id='SetTemperature', args=values)
            before = json.loads(json.dumps(self.device.get_structure()))
            contract, name, contract_args = self.adapter.build('execute_command', args, before)
            self.assertTrue(contract.validate({'function': name, 'args': contract_args}, before).accepted)
            self.assertTrue(self.device.execute_command(1, 'TemperatureControl', 'SetTemperature', **values).success)
            after = json.loads(json.dumps(self.device.get_structure()))
            result = check_postconditions(contract.functions[name], {'args': contract_args}, before, after)
            self.assertEqual(result.status.value, 'VERIFIED_SUCCESS')
            self.assertEqual(len(result.evidence), 1)

    def test_harness_agent_switch_uses_runtime_and_legacy_default_remains(self):
        from smarthome_agent_rl.harness_agent import HarnessAgent, GuardedExecutor
        agent = HarnessAgent(SimpleNamespace(), variant='Candidate', max_steps=20,
            policy={'verify': False, 'verification_version': 1, 'context_version': 0, 'execution_runtime': True})
        self.assertIsInstance(agent.executor, RuntimeExecutor)
        legacy = HarnessAgent(SimpleNamespace(), variant='G', max_steps=20)
        self.assertIsInstance(legacy.executor, GuardedExecutor)
        agent.executor.trace.dispatch = self.dispatch
        result = agent.executor.execute('execute_command', self.action())
        self.assertEqual(result['harness_verification']['status'], 'VERIFIED_SUCCESS')
        with self.assertRaises(ValueError):
            HarnessAgent(SimpleNamespace(), variant='Candidate', max_steps=20,
                policy={'verify': True, 'verification_version': 1, 'context_version': 0, 'execution_runtime': True})


if __name__ == '__main__':
    unittest.main()
