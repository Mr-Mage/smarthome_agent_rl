"""Explicit relative-time ledger from user text and public episode start time."""
import copy
from datetime import datetime, timedelta
import json
import re

from smarthome_agent_rl.guard import GuardError

RELATIVE = re.compile(r'(\d+(?:\.\d+)?)\s+minutes?\s+(from now|after the previous action)', re.I)
GUIDANCE = """
[TIME PLAN]
The ledger below contains literal relative-time phrases from the USER query, not evaluation goals.
On the first response assign an anchor for each row: 'now' for from-now, or the ID of an earlier
action row for after-previous-action. Resolve which previous action the user means from the query.
The metadata fields EXTEND the earlier two-field output example. time_plan is a fixed object
mapping row IDs to anchors, e.g. {"t0":"now","t1":"t0"}; do not output a list or repeat rows.
Return time_plan only when requested by the schema. time_refs maps EACH workflow step to its row
ID, or 'uncovered' if the intent has no covered relative-time phrase. Different due times require
separate workflows; all steps of one workflow start together.
For ordinary tools return time_refs: [] and time_dispositions: []. For schedule_workflow return
time_dispositions: []; for finish return time_refs: [] and all row dispositions.
Use original task start, not later
get_current_time, for 'from now'. Do not invent delays inside steps. Check absolute times in the
query as well: conflicting user constraints may be infeasible and should be explained.
For finish include one disposition per row: registered (a real registration receipt exists),
infeasible (give a concrete reason), or uncovered (explain uncertainty). Registration is not future
execution success. Today's power state is not a future scheduling precondition.
"""
OUTPUT_CONTRACT = """\n[STRUCTURED HARNESS OUTPUT CONTRACT]
Follow the CURRENT response schema, including its time metadata fields. This replaces
earlier JSON examples. Emit metadata FIRST, followed by thought and call, e.g.:
{"time_plan":{"t0":"now"},"time_refs":[],"time_dispositions":[],
 "thought":"Discover devices","call":{"tool":"get_rooms","arguments":{}}}
Only include time_plan when requested by the current schema. arguments is a JSON object.
Do not end the response before all required fields are present.
"""


class TimePlan:
    def __init__(self):
        self.rows, self.table, self.receipts, self.metadata = [], {}, {}, {}
        self.query, self.base_time = None, None
        self.audit = []

    def initialize(self, query, current_time):
        self.query, self.base_time = query, current_time
        self.rows = [{'id': f't{i}', 'source_quote': m.group(0), 'minutes': float(m.group(1)),
            'relative': m.group(2).lower(), 'span': [m.start(), m.end()]}
            for i, m in enumerate(RELATIVE.finditer(query))]
        try:
            datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            self.rows = []  # Unknown public clock format stays uncovered.

    def prompt(self):
        return GUIDANCE + json.dumps({'public_start_time': self.base_time, 'phrases': self.rows,
            'table': self.table, 'registered_rows': sorted(self.receipts),
            'coverage': 'numeric minute-relative phrases only; intent-to-step mapping is model supplied'}, ensure_ascii=False)

    def augment_schema(self, schema):
        schema = copy.deepcopy(schema)
        body = schema['json_schema']['schema']
        body['properties']['time_refs'] = {'type': 'array', 'maxItems': 32,
            'items': {'type': 'string', 'enum': ['uncovered', *[r['id'] for r in self.rows]]}}
        body['properties']['time_dispositions'] = {'type': 'array', 'items': {'type': 'object',
            'properties': {'id': {'type': 'string', 'enum': [r['id'] for r in self.rows]}, 'status': {'type': 'string',
                'enum': ['registered', 'infeasible', 'uncovered']}, 'reason': {'type': 'string', 'maxLength': 512}},
            'required': ['id', 'status', 'reason'], 'additionalProperties': False}}
        body['properties']['time_dispositions']['maxItems'] = len(self.rows)
        body['required'] += ['time_refs', 'time_dispositions']
        if self.rows and not self.table:
            properties = {}
            earlier = []
            for row in self.rows:
                properties[row['id']] = {'type': 'string', 'enum': ['now'] if row['relative'] == 'from now'
                                         else earlier or ['uncovered']}
                earlier = [*earlier, row['id']]
            body['properties']['time_plan'] = {'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}
            body['required'].append('time_plan')
        # The model naturally stops after thought/call. Put required metadata before that suffix
        # so constrained decoding does not force whitespace when it tries to close early.
        order = [k for k in ('time_plan', 'time_refs', 'time_dispositions', 'thought', 'call') if k in body['properties']]
        body['properties'] = {k: body['properties'][k] for k in order}
        body['required'] = order
        return schema

    def consume(self, body):
        if self.rows and not self.table:
            anchors = body.pop('time_plan')
            if not isinstance(anchors, dict) or set(anchors) != {r['id'] for r in self.rows}:
                raise ValueError('time_plan must cover each literal relative-time phrase once')
            table = {}
            base = datetime.strptime(self.base_time, '%Y-%m-%d %H:%M:%S')
            for row in self.rows:
                anchor = anchors[row['id']]
                if row['relative'] == 'from now' and anchor != 'now':
                    raise ValueError('from-now must anchor to original public start time')
                if row['relative'] != 'from now' and anchor == 'uncovered':
                    table[row['id']] = {**row, 'anchor': anchor, 'due_time': None}
                    continue
                if row['relative'] != 'from now' and anchor not in table:
                    raise ValueError('after-previous-action must anchor to an earlier action row')
                if anchor != 'now' and table[anchor]['due_time'] is None:
                    table[row['id']] = {**row, 'anchor': anchor, 'due_time': None}
                    continue
                at = base if anchor == 'now' else datetime.strptime(table[anchor]['due_time'], '%Y-%m-%d %H:%M:%S')
                table[row['id']] = {**row, 'anchor': anchor,
                    'due_time': (at + timedelta(minutes=row['minutes'])).strftime('%Y-%m-%d %H:%M:%S')}
            self.table = table
        self.metadata = {'refs': body.pop('time_refs'), 'dispositions': body.pop('time_dispositions')}
        self.audit.append({'table': copy.deepcopy(self.table), 'metadata': copy.deepcopy(self.metadata)})
        return body

    def check(self, tool, arguments):
        if not self.rows:
            return
        if tool == 'schedule_workflow':
            refs = self.metadata.get('refs', [])
            if len(refs) != len(arguments['steps']):
                raise GuardError('time_plan', 'Map each workflow step to a time constraint ID')
            for ref in refs:
                if ref == 'uncovered':
                    continue
                if ref not in self.table:
                    raise GuardError('time_plan', 'Unknown time constraint ID', ref=ref)
                if self.table[ref]['due_time'] is not None and arguments['start_time'] != self.table[ref]['due_time']:
                    raise GuardError('time_plan', 'Workflow time conflicts with explicit relative-time table',
                                     ref=ref, expected=self.table[ref]['due_time'])
        elif tool == 'finish':
            dispositions = self.metadata.get('dispositions', [])
            by_id = {r['id']: r for r in dispositions}
            if len(by_id) != len(dispositions) or set(by_id) != set(self.table):
                raise GuardError('time_plan', 'Finish must account for each time constraint')
            for ref, item in by_id.items():
                if item['status'] == 'registered' and ref not in self.receipts:
                    raise GuardError('time_plan', 'Claimed registration has no actual receipt', ref=ref)
                if item['status'] != 'registered' and not item['reason'].strip():
                    raise GuardError('time_plan', 'Infeasible/uncovered constraint needs an explanation', ref=ref)

    def observe(self, tool, arguments, response):
        if response.get('status', {}).get('code') != 200 or response.get('error') is not None:
            return
        if tool == 'schedule_workflow':
            workflow_id = response.get('data', {}).get('workflow_id')
            if workflow_id:
                for ref in self.metadata.get('refs', []):
                    if ref in self.table:
                        self.receipts[ref] = {'workflow_id': workflow_id, 'start_time': arguments['start_time']}
        elif tool == 'cancel_workflow':
            self.receipts = {ref: receipt for ref, receipt in self.receipts.items()
                             if receipt['workflow_id'] != arguments['workflow_id']}

    def snapshot(self):
        return {'phrases': self.rows, 'public_start_time': self.base_time,
            'table': self.table, 'receipts': self.receipts, 'turns': self.audit,
            'coverage': 'Numeric minute-relative phrases; model resolves previous-action links and maps steps. No hidden goals.'}
