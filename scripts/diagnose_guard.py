"""Public-cluster differential checks; additional diagnostics, never benchmark scores."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deps/SimuHome'))
from src.simulator.domain.clusters.level_control import LevelControlCluster
from src.simulator.domain.clusters.onoff import OnOffCluster
from smarthome_agent_rl.guard import ToolGuard, GuardError, command_contracts, public_power_rules, harness_schemas
from smarthome_agent_rl.structured import tool_schemas

contracts, sources = command_contracts(ROOT / 'deps/SimuHome/src/simulator/domain/clusters')
power_rules, device_sources = public_power_rules(ROOT / 'deps/SimuHome/src/simulator/domain/devices')
sources.update(device_sources)
guard = ToolGuard(harness_schemas(tool_schemas()), contracts, power_rules)
records = []
cases = [{'Level': level, 'TransitionTime': transition} for level in
         [-1, 0, 1, 10, 254, 255, 100.0, '100', True] for transition in [None, 0, 5]]
cases += [{'level': 100}, {}, {'Level': 100, 'Unknown': 2}]
for command in ('MoveToLevel', 'MoveToLevelWithOnOff'):
    for args in cases:
        level, onoff = LevelControlCluster(), OnOffCluster()
        level.attributes['CurrentLevel'] = 15
        onoff.attributes['OnOff'] = True
        level._device = SimpleNamespace(endpoints={1: {'OnOff': onoff}})
        structure = {'endpoints': {'1': {'clusters': {'LevelControl': level.get_structure(),
                                                    'OnOff': onoff.get_structure()}}}}
        arguments = {'device_id': 'public-prototype', 'endpoint_id': 1, 'cluster_id': 'LevelControl',
                     'command_id': command, 'args': args}
        blocked, error = False, None
        try:
            guard.schema('execute_command', arguments)
            guard.capability('execute_command', arguments, structure)
        except GuardError as exc:
            blocked, error = True, str(exc)
        result = level.execute_command(command, **copy.deepcopy(args))
        records.append({'command': command, 'arguments': args, 'guard_blocked': blocked,
            'guard_error': error, 'simulator_accepted': bool(result.success)})
from src.simulator.domain.devices.air_conditioner import AirConditioner
from src.simulator.domain.devices.fan import Fan
for device_cls, cluster, attribute, values in [
    (AirConditioner, 'Thermostat', 'OccupiedCoolingSetpoint', [700, 1500, 2100, 2825, 3000, 3300]),
    (Fan, 'FanControl', 'PercentSetting', [-1, 0, 20, 80, 100, 101])]:
    for powered in (False, True):
        for value in values:
            device = device_cls('public-prototype')
            device.get_cluster(1, 'OnOff').attributes['OnOff'] = powered
            structure = json.loads(json.dumps(device.get_structure()))
            arguments = {'device_id': 'public-prototype', 'endpoint_id': 1,
                'cluster_id': cluster, 'attribute_id': attribute, 'value': value}
            blocked, error = False, None
            try:
                guard.schema('write_attribute', arguments)
                guard.capability('write_attribute', arguments, structure)
            except GuardError as exc:
                blocked, error = True, str(exc)
            result = device.write_attribute(1, cluster, attribute, value)
            records.append({'device_type': structure['device_type'], 'powered': powered,
                'arguments': arguments, 'guard_blocked': blocked, 'guard_error': error,
                'simulator_accepted': bool(result.success)})
output = ROOT / 'work/harness-mvp/guard-differential.json'
output.parent.mkdir(parents=True, exist_ok=True)
report = {'purpose': 'additional_public_api_diagnostic', 'cases': len(records),
    'false_rejections': sum(r['guard_blocked'] and r['simulator_accepted'] for r in records),
    'source_sha256': sources, 'records': records}
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: report[k] for k in ('cases', 'false_rejections')}))
if report['false_rejections']:
    raise RuntimeError('Guard rejected calls accepted by public implementation')
