"""Local action postconditions only; never infer the benchmark's hidden success goals."""
import copy


def expected_effect_v2(tool, arguments, before):
    if tool == 'write_attribute':
        value = arguments['value']
        old = before.get('endpoints', {}).get(str(arguments['endpoint_id']), {}).get('clusters', {}).get(
            arguments['cluster_id'], {}).get('attributes', {}).get(arguments['attribute_id'], {}).get('value')
        # Public setters may coerce values. Without an exact contract, do not
        # declare a false mismatch merely because the accepted types differ.
        if value is None or old is not None and type(value) is not type(old):
            return {}
    original = expected_effect(tool, arguments, before)
    if original or tool != 'execute_command':
        return original
    endpoint, cluster, command, args = (arguments['endpoint_id'], arguments['cluster_id'],
        arguments['command_id'], arguments['args'])
    if cluster == 'OperationalState' and command == 'Start':
        countdown = before.get('endpoints', {}).get(str(endpoint), {}).get('clusters', {}).get(
            cluster, {}).get('attributes', {}).get('CountdownTime', {}).get('value')
        if type(countdown) in (int, float) and countdown <= 0:
            # A successful Start can finish on the next simulator tick.
            return {}
        return {(endpoint, cluster, 'OperationalState'): 1}
    if cluster == 'LaundryWasherMode' and command == 'ChangeToMode':
        return {(endpoint, cluster, 'CurrentMode'): args['new_mode']}
    if cluster == 'FanControl' and command == 'Step':
        # Public cluster code on a detached snapshot; no simulator I/O or time advance.
        from src.simulator.domain.clusters.fan_control import FanControlCluster
        attrs = before.get('endpoints', {}).get(str(endpoint), {}).get('clusters', {}).get(cluster, {}).get('attributes', {})
        if 'FanModeSequence' not in attrs or 'PercentSetting' not in attrs:
            return {}
        shadow = FanControlCluster(fan_mode_sequence=attrs['FanModeSequence']['value'])
        shadow.attributes.update({k: copy.deepcopy(v['value']) for k, v in attrs.items()})
        try:
            result = shadow._step(**args)
        except (TypeError, ValueError):
            return {}
        if result.success:
            return {(endpoint, cluster, 'PercentSetting'): shadow.attributes['PercentSetting']}
    return {}


def expected_effect(tool, arguments, before):
    if tool == 'write_attribute':
        return {(arguments['endpoint_id'], arguments['cluster_id'], arguments['attribute_id']): arguments['value']}
    if tool != 'execute_command':
        return {}
    endpoint, cluster, command, args = (arguments['endpoint_id'], arguments['cluster_id'],
        arguments['command_id'], arguments['args'])
    if cluster == 'OnOff' and command in ('On', 'Off', 'Toggle'):
        if command == 'Toggle':
            value = before['endpoints'][str(endpoint)]['clusters'][cluster]['attributes']['OnOff']['value']
            target = not value
        else:
            target = command == 'On'
        return {(endpoint, cluster, 'OnOff'): target}
    if cluster == 'LevelControl' and command in ('MoveToLevel', 'MoveToLevelWithOnOff'):
        return {(endpoint, cluster, 'CurrentLevel'): args['Level']}
    if cluster == 'TemperatureControl' and command == 'SetTemperature':
        if args.get('target_temperature') is not None:
            return {(endpoint, cluster, 'TemperatureSetpoint'): args['target_temperature']}
        if args.get('target_temperature_level') is not None:
            return {(endpoint, cluster, 'SelectedTemperatureLevel'): args['target_temperature_level']}
    return {}


def verify_effect(expectations, after, result):
    data = result.get('data') or {}
    if data.get('suppressed'):
        return {'status': 'suppressed', 'verified': False, 'reason': data.get('reason')}
    if data.get('duration', 0) > 0:
        return {'status': 'pending', 'verified': None, 'reason': 'Transition still in progress; no virtual time advanced'}
    if not expectations:
        return {'status': 'uncovered', 'verified': None}
    differences = []
    for (endpoint, cluster, attribute), expected in expectations.items():
        observed = after.get('endpoints', {}).get(str(endpoint), {}).get('clusters', {}).get(
            cluster, {}).get('attributes', {}).get(attribute, {}).get('value')
        if observed != expected or type(observed) is bool and type(expected) is not bool:
            differences.append({'endpoint': endpoint, 'cluster': cluster, 'attribute': attribute,
                                'expected': expected, 'observed': observed})
    return {'status': 'mismatch' if differences else 'verified', 'verified': not differences,
            'differences': differences}
