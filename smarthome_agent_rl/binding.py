"""Render exact identifiers from successful observations already visible to the actor."""
import hashlib
import json


def field(message, key):
    return message[key] if isinstance(message, dict) else getattr(message, key)


def visible_catalog(messages):
    start = max(i for i, m in enumerate(messages) if 'This is your actual task.' in field(m, 'content'))
    rooms, devices, sources = {}, {}, []
    complete_rooms = complete_devices = False
    pending = None
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
            except (ValueError, TypeError, KeyError):
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
            if not isinstance(data, dict):
                continue
            if tool in ('add_device', 'remove_device'):
                # Invalidate catalog completeness after topology changes; never infer a new ID.
                rooms, devices, sources = {}, {}, []
                complete_rooms = complete_devices = False
            elif tool == 'get_rooms' and isinstance(data.get('rooms'), list):
                rooms = {r['room_id']: r.get('display_name') for r in data['rooms']
                         if isinstance(r, dict) and isinstance(r.get('room_id'), str)}
                complete_rooms = True
            elif tool == 'get_home_state' and isinstance(data.get('rooms'), dict):
                rooms = {room: None for room in data['rooms']}
                devices = {d['device_id']: {'room_id': room, 'device_type': d.get('device_type')}
                           for room, value in data['rooms'].items() for d in value.get('devices', [])
                           if isinstance(d, dict) and isinstance(d.get('device_id'), str)}
                complete_rooms = complete_devices = True
            elif tool == 'get_room_devices' and isinstance(args.get('room_id'), str):
                room = args['room_id']
                rooms.setdefault(room, None)
                devices = {key: value for key, value in devices.items() if value.get('room_id') != room}
                devices.update({key: {'room_id': room, 'device_type': value.get('device_type')}
                                for key, value in data.items() if isinstance(value, dict) and 'device_type' in value})
            elif tool == 'get_room_states' and isinstance(args.get('room_id'), str):
                rooms.setdefault(args['room_id'], None)
            elif tool == 'get_device_structure' and isinstance(data.get('device_id'), str):
                device = data['device_id']
                devices.setdefault(device, {'room_id': None, 'device_type': data.get('device_type')})
            else:
                continue
            sources.append({'message_index': index, 'tool': tool,
                            'observation_sha256': hashlib.sha256(content.encode()).hexdigest()})
    return {'rooms': rooms, 'devices': devices, 'complete_rooms': complete_rooms,
            'complete_devices': complete_devices, 'sources': sources}


def catalog_prompt(catalog):
    if not catalog['rooms'] and not catalog['devices']:
        return None
    payload = {'rooms': [{'room_id': key, 'display_name': value} for key, value in sorted(catalog['rooms'].items())],
               'devices': [{'device_id': key, **value} for key, value in sorted(catalog['devices'].items())]}
    return ('OBSERVED IDENTIFIER CANDIDATES\n'
            'These exact IDs are copied from successful tool observations in this task. '
            'Use the observed room_id/device_id spelling when referring to these entities; do not shorten or translate IDs. '
            'The device list may be incomplete: an unlisted device is not proof of absence. '
            'This list describes identifiers only, not current state or task completion.\n'
            + json.dumps(payload, ensure_ascii=False, separators=(',', ':')))


class BindingProvider:
    def __init__(self, inner, executor):
        self.inner, self.executor = inner, executor

    def generate(self, messages, response_format=None):
        from src.agents.types import ChatMessage
        catalog = visible_catalog(messages)
        prompt = catalog_prompt(catalog)
        self.executor.binding_audit.append({'turn': len(self.executor.binding_audit) + 1,
            'catalog': catalog, 'prompt': prompt, 'used': prompt is not None,
            'original_messages_sha256': hashlib.sha256(json.dumps(
                [{'role': field(m, 'role'), 'content': field(m, 'content')} for m in messages],
                ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
        converted = [*messages, ChatMessage(role='user', content=prompt)] if prompt else messages
        try:
            return self.inner.generate(converted, response_format=response_format)
        finally:
            self.executor.save_audit()
