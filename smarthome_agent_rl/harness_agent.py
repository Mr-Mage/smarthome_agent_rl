"""Project variants around the upstream ReAct loop, with process-local tool injection."""
import copy
import json
import hashlib
from pathlib import Path
import time

from src.agents.strategies.react_agent import ReActAgent, ReActConfig
import src.agents.strategies.react_agent as react_module
from src.agents.strategies.base import ToolInvocation
from src.agents.tools import run_tool
from src.agents.types import ChatMessage

from smarthome_agent_rl.guard import ToolGuard, GuardError, command_contracts, public_power_rules, harness_schemas
from smarthome_agent_rl.structured import StructuredProvider, tool_schemas
from smarthome_agent_rl.verification import expected_effect, expected_effect_v2, verify_effect, verify_effect_v2
from smarthome_agent_rl.action_state import ActionLedger
from smarthome_agent_rl.recovery import RecoveryPolicy, public_context
from smarthome_agent_rl.semantic_verifier import VerificationContext, feedback as semantic_feedback
from smarthome_agent_rl.device_contract import ContractRegistry

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = {'execute_command', 'write_attribute', 'schedule_workflow', 'cancel_workflow',
             'add_device', 'remove_device', 'set_tick_interval'}


def ok(response):
    return isinstance(response, dict) and response.get('status', {}).get('code') == 200 and response.get('error') is None


class GuardedExecutor:
    def __init__(self, *, verify=False, verification_version=1, repair_limit=2, query_limit=40,
                 dead_front=False, workflow_all_devices=False, audit_fn=None, dispatch=run_tool,
                 recovery=False, recovery_total_limit=6, semantic_verifier=None,
                 reflection_verifier=None, semantic_blocking=False, contract_registry=None,
                 contract_blocking=False):
        contracts, sources = command_contracts(ROOT / 'deps/SimuHome/src/simulator/domain/clusters')
        power_rules, device_sources = public_power_rules(ROOT / 'deps/SimuHome/src/simulator/domain/devices', dead_front=dead_front)
        sources.update(device_sources)
        sources['api/schemas.py'] = hashlib.sha256((ROOT / 'deps/SimuHome/src/simulator/api/schemas.py').read_bytes()).hexdigest()
        self.guard = ToolGuard(harness_schemas(tool_schemas()), contracts, power_rules)
        self.sources, self.verify = sources, verify
        self.verification_version = verification_version
        self.workflow_all_devices = workflow_all_devices
        self.repair_limit, self.query_limit, self.audit_fn, self.dispatch = repair_limit, query_limit, audit_fn, dispatch
        self.audit, self.actual, self.failures, self.observations = [], [], {}, []
        self.actions = ActionLedger()
        self.active_action_id = None
        self.recovery = RecoveryPolicy(repair_limit, recovery_total_limit) if recovery else None
        self.extra_queries, self.turn = 0, 0
        self.context_audit = []
        self.structured_audit = []
        self.binding_audit = []
        self.start_semantics_audit = []
        self.time_plan = None
        self.task_spec = None
        self.semantic_verifier = semantic_verifier
        self.reflection_verifier = reflection_verifier
        self.semantic_blocking = semantic_blocking
        self.contract_registry = contract_registry
        self.contract_blocking = contract_blocking

    def save_audit(self):
        if self.audit_fn:
            self.audit_fn({'proposals': self.audit, 'actual_observations': self.observations,
                'public_semantics_source_sha256': self.sources, 'extra_queries': self.extra_queries,
                'context': self.context_audit, 'structured': self.structured_audit,
                'binding': self.binding_audit,
                'start_semantics': self.start_semantics_audit,
                'action_lifecycle': self.actions.snapshot(),
                'recovery_policy': self.recovery.snapshot() if self.recovery else None,
                'task_spec': self.task_spec.snapshot() if self.task_spec else None,
                'time_plan': self.time_plan.snapshot() if self.time_plan is not None else None})

    def record_structured(self, records):
        self.structured_audit = records
        self.save_audit()

    def call(self, tool, arguments, *, extra=False, action_id=None):
        if extra:
            if self.extra_queries >= self.query_limit:
                return None
            self.extra_queries += 1
        action_id = action_id or self.actions.propose(tool, arguments, turn=self.turn,
            observation_version=len(self.observations), extra_query=extra,
            parent_action_id=self.active_action_id)
        if extra and self.active_action_id:
            parent = self.actions.records[self.active_action_id]
            self.actions.records[action_id].goal_ids = list(parent.goal_ids)
            self.actions.records[action_id].goal_id = parent.goal_id
        started = time.monotonic()
        self.actions.dispatched(action_id)
        try:
            response = self.dispatch(tool, copy.deepcopy(arguments))
        except Exception as exc:
            self.actions.interrupted(action_id, exc)
            raise
        invocation = ToolInvocation(tool=tool, params=copy.deepcopy(arguments), observation=copy.deepcopy(response))
        self.actual.append(invocation)
        self.observations.append({'turn': self.turn, 'tool': tool, 'arguments': copy.deepcopy(arguments),
            'response': copy.deepcopy(response), 'extra_query': extra,
            'duration_seconds': time.monotonic() - started})
        self.actions.observed(action_id, response, len(self.observations))
        if extra and isinstance(response, dict) and response.get('status', {}).get('code', 200) >= 500:
            raise RuntimeError(f'Guard query infrastructure failure: {response}')
        return response

    def execute(self, tool, arguments):
        self.turn = self.structured_audit[-1]['turn'] if self.structured_audit else self.turn + 1
        action_id = self.actions.propose(tool, arguments, turn=self.turn,
                                         observation_version=len(self.observations))
        self.active_action_id = action_id
        if self.task_spec is not None and self.structured_audit:
            refs = self.structured_audit[-1].get('goal_refs', [])
            self.actions.records[action_id].goal_ids = list(refs)
            self.actions.records[action_id].goal_id = refs[0] if len(refs) == 1 else None
        record = {'turn': self.turn, 'tool': tool, 'arguments': copy.deepcopy(arguments),
            'action_id': action_id,
            'blocked': False, 'uncovered': [], 'extra_queries_before': self.extra_queries,
            'actual_calls_before': len(self.actual)}
        self.audit.append(record)
        key = json.dumps([tool, arguments.get('device_id'), arguments.get('cluster_id'),
                          arguments.get('command_id', arguments.get('attribute_id'))], sort_keys=True)
        try:
            self.guard.schema(tool, arguments)
            if self.time_plan is not None:
                record['time_refs'] = copy.deepcopy(self.time_plan.metadata.get('refs', []))
                self.time_plan.check(tool, arguments)
            if self.verify and self.failures.get(key, 0) > self.repair_limit:
                raise GuardError('recovery', 'Two repair attempts exhausted; change plan or finish honestly')
            if self.verify and self.failures.get(key, 0):
                record['repair_attempt'] = self.failures[key]
            before = {}
            if tool in ('execute_command', 'write_attribute', 'get_attribute'):
                query = self.call('get_device_structure', {'device_id': arguments['device_id']}, extra=True)
                if query is None:
                    record['uncovered'].append('prequery_budget_exhausted')
                elif not ok(query):
                    raise GuardError('capability', 'Device structure query failed', response=query)
                else:
                    before = query['data']
                    record['uncovered'].extend(self.guard.capability(tool, arguments, before))
                    if self.contract_registry is not None:
                        contract_action = {'function': arguments.get('command_id', arguments.get('attribute_id')),
                                           'args': arguments.get('args', {'value': arguments.get('value')})}
                        contract_result = self.contract_registry.validate(contract_action, before)
                        record['contract'] = contract_result.response()
                        record['uncovered'].extend(contract_result.uncovered)
                        if self.contract_blocking and not contract_result.accepted:
                            raise GuardError('deterministic', 'Executable device contract rejected action',
                                             reason_code=contract_result.reason_code,
                                             **contract_result.evidence,
                                             repair_hint=contract_result.repair_hint)
            elif tool == 'schedule_workflow':
                # Inspect only static capabilities; today's state cannot predict future state.
                structures = {}
                attempted = set()
                queried = False
                for step in arguments['steps']:
                    device = step['args']['device_id']
                    if (self.workflow_all_devices or not queried) and device not in attempted:
                        queried = True
                        attempted.add(device)
                        query = self.call('get_device_structure', {'device_id': device}, extra=True)
                        if query is not None and ok(query):
                            structures[device] = query['data']
                        elif self.workflow_all_devices and query is not None:
                            raise GuardError('capability', 'Workflow device structure query failed', response=query)
                        elif self.workflow_all_devices:
                            record['uncovered'].append('prequery_budget_exhausted')
                    if device in structures:
                        record['uncovered'].extend(self.guard.capability(step['tool'], step['args'],
                            structures[device], state=False))
                    else:
                        record['uncovered'].append('workflow_device_not_queried:' + device)
                record['uncovered'].append('future_state_preconditions')
            verifier = self.reflection_verifier or self.semantic_verifier
            if verifier is not None and tool in MUTATIONS:
                semantic_context = VerificationContext(
                    user_goal=getattr(self, 'user_goal', ''), environment_state=before,
                    proposed_action={'tool': tool, **copy.deepcopy(arguments)},
                    recent_actions=[{'tool': row.tool, **copy.deepcopy(row.params)} for row in self.actual[-8:]],
                    contract=record.get('contract'))
                semantic_result = verifier.verify(semantic_context)
                record['semantic_verification'] = semantic_result.as_dict()
                if self.semantic_blocking and semantic_result.verdict == 'DENY':
                    raise GuardError('semantic', 'Semantic verifier rejected the proposed action',
                                     reason_code=semantic_result.reason_code or 'SEMANTIC_REJECT',
                                     verifier=semantic_result.as_dict(),
                                     repair_hint=semantic_feedback(semantic_result).get('repair_hint', {}))
            recovery_context = None
            if self.recovery is not None:
                recovery_context = public_context(tool, arguments, self.observations)
                self.recovery.check(tool, arguments, recovery_context)
            response = self.call(tool, arguments, action_id=action_id)
            if self.recovery is not None:
                self.recovery.observe(tool, arguments, response, recovery_context, action_id)
            if self.time_plan is not None:
                self.time_plan.observe(tool, arguments, response)
            simulator_failed = not ok(response)
            record['reached_executor'] = True
            failed = not ok(response)
            if self.verify and ok(response):
                verifier = verify_effect_v2 if self.verification_version == 2 else verify_effect
                verification = {'status': 'not_applicable', 'verified': None}
                if tool in ('execute_command', 'write_attribute'):
                    effects = (expected_effect_v2 if self.verification_version == 2 else expected_effect)(tool, arguments, before)
                    metadata = response.get('data') or {}
                    if self.verification_version == 2 and metadata.get('unchanged'):
                        verification = {'status': 'no_effect', 'verified': None}
                    elif self.verification_version == 2 and (not effects or metadata.get('duration', 0) > 0 or metadata.get('suppressed')):
                        verification = verifier(effects, before, response)
                    else:
                        query = self.call('get_device_structure', {'device_id': arguments['device_id']}, extra=True)
                        verification = verifier(effects, query['data'], response) if query and ok(query) else {
                            'status': 'query_unavailable', 'verified': None}
                elif tool in ('schedule_workflow', 'cancel_workflow'):
                    workflow_id = response.get('data', {}).get('workflow_id', arguments.get('workflow_id'))
                    query = self.call('get_workflow_status', {'workflow_id': workflow_id}, extra=True) if workflow_id and (
                        self.verification_version == 1 or tool == 'cancel_workflow') else None
                    verification = {'status': 'registration_only' if tool == 'schedule_workflow' else 'cancellation',
                        'verified': None, 'public_status': query, 'future_success_verified': False}
                record['verification'] = verification
                failed |= verification.get('verified') is False
                response = {**response, 'harness_verification': verification}
                if self.verification_version == 2 and verification.get('verified') is False:
                    response = {**response, 'error': {'type': 'harness_postcondition',
                        'detail': verification, 'repair': 'Query public state and correct the action; do not claim success.'}}
            if failed:
                self.failures[key] = self.failures.get(key, 0) + 1
            else:
                record['recovered'] = bool(self.failures.pop(key, 0))
            record['simulator_error'] = simulator_failed
            return response
        except GuardError as exc:
            record.update({'blocked': True, 'layer': exc.layer, 'detail': exc.detail,
                           'reached_executor': False})
            self.failures[key] = self.failures.get(key, 0) + 1
            record['response'] = exc.response()
            self.actions.rejected(action_id, record['response'])
            return record['response']
        except Exception as exc:
            self.actions.interrupted(action_id, exc)
            if self.recovery is not None and self.actions.records[action_id].state == 'unknown':
                self.recovery.observe(tool, arguments, None,
                    public_context(tool, arguments, self.observations), action_id)
            raise
        finally:
            record['extra_queries'] = self.extra_queries - record['extra_queries_before']
            record['actual_calls'] = len(self.actual) - record['actual_calls_before']
            self.active_action_id = None
            self.save_audit()


class HarnessAgent:
    def __init__(self, llm, *, variant, max_steps, trace_fn=None, audit_fn=None,
                 repair_limit=2, query_limit=40, policy=None, token_count_fn=None):
        if variant not in ('B0', 'B1', 'B2', 'G', 'GV', 'GC', 'Full', 'GV2', 'GC2', 'Candidate', 'GD', 'GW', 'GDW', 'TimePlan', 'GThinking', 'Teacher', 'SFT9B', 'GB', 'SFT9B_B', 'GS', 'GR', 'GTS', 'GEC', 'RC', 'RCV', 'GTM', 'GTME', 'GTMEC'):
            raise ValueError(variant)
        policy = policy or {'verify': variant in ('GV', 'Full', 'GV2'),
            'verification_version': 2 if variant == 'GV2' else 1,
            'context_version': 2 if variant == 'GC2' else 1 if variant in ('GC', 'Full') else 0}
        if policy.get('execution_runtime'):
            incompatible = ('verify', 'context_version', 'time_plan', 'identifier_binding',
                            'start_semantics', 'recovery', 'task_spec', 'evidence_context',
                            'semantic_verifier', 'reflection_verifier', 'contract_registry',
                            'dead_front', 'workflow_all_devices')
            if variant in ('B0', 'GR', 'GTS', 'GEC') or any(policy.get(key) for key in incompatible):
                raise ValueError('Execution runtime comparison must isolate the tool execution boundary')
            from smarthome_agent_rl.execution.harness import RuntimeExecutor
            from smarthome_agent_rl.execution.simuhome_contract import SimuHomeContractAdapter
            contracts, sources = command_contracts(ROOT / 'deps/SimuHome/src/simulator/domain/clusters')
            power_rules, device_sources = public_power_rules(ROOT / 'deps/SimuHome/src/simulator/domain/devices')
            sources.update(device_sources)
            for filename in ('configs/device-contract-rules.json', 'deps/SimuHome/src/simulator/api/schemas.py'):
                sources[filename] = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
            self.executor = RuntimeExecutor(guard=ToolGuard(harness_schemas(tool_schemas()), contracts, power_rules),
                adapter=SimuHomeContractAdapter(contracts, power_rules), dispatch=run_tool,
                invocation_factory=ToolInvocation, sources=sources, query_limit=query_limit, audit_fn=audit_fn,
                verify_mutations=policy.get('runtime_verify', True))
        else:
            self.executor = GuardedExecutor(verify=policy['verify'], verification_version=policy['verification_version'], audit_fn=audit_fn,
                                       repair_limit=repair_limit, query_limit=query_limit,
                                       dead_front=policy.get('dead_front', False),
                                       workflow_all_devices=policy.get('workflow_all_devices', False),
                                       recovery=policy.get('recovery', variant == 'GR'),
                                       recovery_total_limit=policy.get('recovery_total_limit', 6),
                                       semantic_verifier=policy.get('semantic_verifier'),
                                       reflection_verifier=policy.get('reflection_verifier'),
                                       semantic_blocking=policy.get('semantic_blocking', False),
                                       contract_registry=policy.get('contract_registry'),
                                       contract_blocking=policy.get('contract_blocking', False))
        if policy.get('evidence_context', variant == 'GEC'):
            if variant in ('GTS', 'GR') or any(policy.get(key) for key in ('verify', 'context_version', 'time_plan',
                    'identifier_binding', 'start_semantics', 'recovery', 'task_spec',
                    'dead_front', 'workflow_all_devices')):
                raise ValueError('Evidence context must isolate prompt representation from other interventions')
            from smarthome_agent_rl.evidence_context import EvidenceContextProvider
            llm = EvidenceContextProvider(llm, self.executor, token_count_fn)
        provider = StructuredProvider(llm, finish_guard=False, recovery=False, guidance=False,
                                      audit_fn=self.executor.record_structured)
        if policy.get('time_plan'):
            from smarthome_agent_rl.time_plan import TimePlan
            self.executor.time_plan = TimePlan()
            provider.time_plan = self.executor.time_plan
        provider.schemas = self.executor.guard.schemas
        if policy.get('task_spec', variant == 'GTS'):
            if any(policy.get(key) for key in ('verify', 'context_version', 'time_plan', 'identifier_binding', 'start_semantics', 'recovery')):
                raise ValueError('TaskSpec experiment must isolate goal metadata from other interventions')
            from smarthome_agent_rl.task_spec import TaskSpec
            self.executor.task_spec = TaskSpec(self.executor)
            provider.task_spec = self.executor.task_spec
        if policy.get('start_semantics'):
            if policy['context_version'] or policy.get('time_plan') or policy.get('identifier_binding'):
                raise ValueError('Start semantics diagnosis requires complete unmodified task history')
            from smarthome_agent_rl.start_semantics import StartSemanticsProvider
            provider = StartSemanticsProvider(provider, self.executor)
        if policy.get('identifier_binding'):
            if policy['context_version'] or policy.get('time_plan'):
                raise ValueError('Identifier diagnosis requires complete unmodified task history')
            from smarthome_agent_rl.binding import BindingProvider
            provider = BindingProvider(provider, self.executor)
        if policy['context_version']:
            from smarthome_agent_rl.context import LedgerProvider, CompactLedgerProvider
            provider = CompactLedgerProvider(provider, self.executor, token_count_fn) if policy['context_version'] == 2 else LedgerProvider(provider, self.executor)
        if policy.get('task_runtime_context'):
            if variant != 'GTMEC' or not policy.get('task_runtime') or policy.get('task_runtime_clock') != 'public_events':
                raise ValueError('Resource context requires the dedicated public-event runtime arm')
            from smarthome_agent_rl.execution.resource_context import RuntimeContextProvider
            provider = RuntimeContextProvider(provider, lambda: getattr(self, 'task_runtime', None))
        self.agent = ReActAgent(provider, config=ReActConfig(max_steps=max_steps,
            show_assistant_raw=True, trace_fn=trace_fn))

    def run(self, query, *, user_location=None, current_time=None):
        self.executor.user_goal = query
        runtime = getattr(self, 'task_runtime', None)
        if runtime is not None:
            runtime.start(query, user_location=user_location, current_time=current_time)
        if self.executor.task_spec is not None:
            self.executor.task_spec.initialize(query)
        if self.executor.time_plan is not None:
            self.executor.time_plan.initialize(query, current_time)
        original = react_module.run_tool
        def execute_with_supervision(tool, arguments):
            response = self.executor.execute(tool, arguments)
            runtime.supervise(phase='agent_tool_return')
            return response
        react_module.run_tool = execute_with_supervision if runtime is not None else self.executor.execute
        try:
            result = self.agent.run(query, user_location=user_location, current_time=current_time)
            if runtime is not None:
                runtime.finish()
            # Evaluators see real calls, including extra public queries; blocked proposals aren't calls.
            result.tool_calls = list(self.executor.actual)
            return result
        finally:
            react_module.run_tool = original
