"""Explain power/run distinction only for publicly observed supported commands."""
import hashlib
import json

from smarthome_agent_rl.binding import field


def observed_cycle_devices(messages):
    start = max(i for i, m in enumerate(messages) if 'This is your actual task.' in field(m, 'content'))
    devices, pending = {}, None
    for index, message in enumerate(messages[start + 1:], start + 1):
        role, content = field(message, 'role'), field(message, 'content')
        if role == 'assistant':
            pending = None
            try:
                body = json.loads(content)
                if 'call' in body:
                    pending = body['call']
                else:
                    args = body['action_input']
                    pending = {'tool': body['action'], 'arguments': json.loads(args) if isinstance(args, str) else args}
            except (ValueError, KeyError, TypeError):
                pass
        elif role == 'user' and content.startswith('observation:') and pending:
            tool, args = pending.get('tool'), pending.get('arguments', {})
            pending = None
            try:
                response = json.loads(content.split(':', 1)[1])
            except (ValueError, TypeError):
                continue
            if not isinstance(response, dict) or response.get('status', {}).get('code') != 200 or response.get('error') is not None:
                continue
            data = response.get('data')
            if tool in ('add_device', 'remove_device'):
                devices.clear()
            elif tool == 'get_device_structure' and isinstance(data, dict) and data.get('device_id') == args.get('device_id'):
                device = data['device_id']
                devices.pop(device, None)
                supported = []
                for endpoint, value in data.get('endpoints', {}).items():
                    clusters = value.get('clusters', {})
                    if str(endpoint).isdigit() and 'Start' in clusters.get('OperationalState', {}).get('commands', []) and 'On' in clusters.get('OnOff', {}).get('commands', []):
                        supported.append(int(endpoint))
                if supported:
                    devices[device] = {'device_id': device, 'endpoints': sorted(supported),
                        'message_index': index, 'observation_sha256': hashlib.sha256(content.encode()).hexdigest()}
    return [devices[key] for key in sorted(devices)]


def semantics_prompt(devices):
    if not devices:
        return None
    return ('PUBLIC CYCLE COMMAND SEMANTICS\n'
        'For the observed devices below, OnOff.On powers the device; setting its cycle mode or dryness level also does not start its cycle. '
        'OperationalState.Start is a separate supported command to start a stopped cycle and requires power on. '
        'When planning a requested cycle start, distinguish powering/configuring from starting, including workflow steps. '
        'These capabilities do not establish current running state or task feasibility.\n'
        + json.dumps([{'device_id': d['device_id'], 'endpoints': d['endpoints']} for d in devices], separators=(',', ':')))


class StartSemanticsProvider:
    def __init__(self, inner, executor):
        self.inner, self.executor = inner, executor

    def generate(self, messages, response_format=None):
        from src.agents.types import ChatMessage
        devices = observed_cycle_devices(messages)
        prompt = semantics_prompt(devices)
        self.executor.start_semantics_audit.append({'turn': len(self.executor.start_semantics_audit) + 1,
            'devices': devices, 'prompt': prompt, 'used': prompt is not None})
        converted = [*messages, ChatMessage(role='user', content=prompt)] if prompt else messages
        try:
            return self.inner.generate(converted, response_format=response_format)
        finally:
            self.executor.save_audit()
