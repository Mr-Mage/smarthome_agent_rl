"""External provider harness; upstream ReAct and tools stay untouched."""
import copy
import inspect
import json
import re

from src.agents.providers.base import LLMProvider
from src.agents.tools import TOOL_REGISTRY
from src.agents.types import ChatMessage


TYPE_SCHEMAS = {
    "str": {"type": "string"}, "int": {"type": "integer"},
    "float": {"type": "number"}, "dict": {"type": "object"},
    "list": {"type": "array", "items": {}}, "any": {},
}
CONTRACT = """\n[STRUCTURED HARNESS OUTPUT CONTRACT]
This contract overrides the earlier JSON output example. Output exactly:
{"thought":"your reasoning", "call":{"tool":"one tool name", "arguments":{}}}
arguments is a JSON OBJECT, never a quoted JSON string. Follow the supplied schema.
Use the required parameters in the tool table. execute_command requires device_id,
endpoint_id, cluster_id, command_id AND args. No command parameters means args: {}.
Matter OnOff cluster uses command_id On or Off, endpoint_id from device structure.
Read actual tool observations before deciding that a device is absent. Discovery is
required before finish is available. Verify requested changes through query tools
before finishing; do not claim success from reasoning alone.
If a tool returns an error, fix the reported parameter or query its capabilities.
Do not repeat the same invalid call. A finish answer must describe observed results.
"""
BASIC_CONTRACT = """\n[STRUCTURED HARNESS OUTPUT CONTRACT]
This contract overrides the earlier JSON output example. Output exactly:
{"thought":"your reasoning", "call":{"tool":"one tool name", "arguments":{}}}
arguments is a JSON OBJECT, never a quoted JSON string. Follow the supplied schema.
"""
HARNESS_SETTINGS = {
    "structured-v1": {"finish_guard": True, "recovery": True, "guidance": True},
    "structured-no-gate": {"finish_guard": False, "recovery": True, "guidance": True},
    "structured-no-recovery": {"finish_guard": True, "recovery": False, "guidance": True},
    "structured-schema": {"finish_guard": False, "recovery": False, "guidance": False},
}


def tool_schemas():
    """Read argument names/types from the full pristine upstream registry."""
    schemas = {}
    for name, function in TOOL_REGISTRY.items():
        doc = inspect.getdoc(function) or ""
        args_doc = doc.split("Args:", 1)[1].split("Returns:", 1)[0]
        properties, required = {}, []
        for key, kind, description in re.findall(r"^\s*(\w+) \((\w+)\): ([^\n]+)", args_doc, re.M):
            if kind not in TYPE_SCHEMAS:
                raise ValueError(f"Unsupported upstream argument type: {name}.{key}: {kind}")
            properties[key] = {**copy.deepcopy(TYPE_SCHEMAS[kind]), "description": description}
            if "[required]" in description:
                required.append(key)
        if not properties and "(none)" not in args_doc:
            raise ValueError(f"No argument schema extracted for {name}")
        schemas[name] = {"type": "object", "properties": properties,
                         "required": required, "additionalProperties": False}
    return schemas


def canonical_action(body):
    call = body["call"]
    return {"thought": body["thought"], "action": call["tool"],
            "action_input": json.dumps(call["arguments"], ensure_ascii=False)}


class StructuredProvider(LLMProvider):
    def __init__(self, inner, audit_fn=None, *, finish_guard=True, recovery=True, guidance=True, time_plan=None,
                 task_spec=None):
        self.inner = inner
        self.schemas = tool_schemas()
        self.audit_fn = audit_fn
        self.audit = []
        self.finish_guard, self.recovery, self.guidance = finish_guard, recovery, guidance
        self.time_plan = time_plan
        self.task_spec = task_spec

    def response_format(self, finish_enabled=True):
        alternatives = []
        for name, schema in sorted(self.schemas.items()):
            if name == "finish" and not finish_enabled:
                continue
            alternatives.append({"type": "object", "properties": {
                "tool": {"type": "string", "const": name}, "arguments": schema},
                "required": ["tool", "arguments"], "additionalProperties": False})
        return {"type": "json_schema", "json_schema": {"name": "structured_smart_home_v1",
            "strict": True, "schema": {"type": "object", "properties": {
                "thought": {"type": "string"}, "call": {"anyOf": alternatives}},
                "required": ["thought", "call"], "additionalProperties": False}}}

    def validate(self, body, finish_enabled=True):
        if not isinstance(body, dict) or set(body) != {"thought", "call"} or not isinstance(body["thought"], str):
            raise ValueError("Expected thought and call")
        call = body["call"]
        if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
            raise ValueError("Expected tool and arguments")
        name, args = call["tool"], call["arguments"]
        if not isinstance(name, str) or name not in self.schemas or not isinstance(args, dict):
            raise ValueError("Unknown tool or invalid argument object")
        if name == "finish" and not finish_enabled:
            raise ValueError("Discovery required before finish")
        schema = self.schemas[name]
        if set(args) - set(schema["properties"]) or set(schema["required"]) - set(args):
            raise ValueError(f"Invalid keys for {name}")
        for key, value in args.items():
            kind = schema["properties"][key].get("type")
            valid = {"string": isinstance(value, str), "integer": type(value) is int,
                     "number": type(value) in (int, float), "object": isinstance(value, dict),
                     "array": isinstance(value, list), None: True}[kind]
            if not valid:
                raise ValueError(f"Invalid type for {name}.{key}")
        return canonical_action(body)

    def generate(self, messages, response_format=None):
        # Only the actual task's observations enable finish; few-shot examples do not.
        task_start = max(i for i, msg in enumerate(messages) if "This is your actual task." in msg.content)
        observations = [msg.content for msg in messages[task_start + 1:]
                        if msg.role == "user" and msg.content.startswith("observation:")]
        successful_discovery = False
        discovery_tools = {"get_rooms", "get_room_devices", "get_home_state", "get_device_structure",
                           "get_all_attributes", "get_attribute", "get_room_states"}
        previous_tool = None
        for msg in messages[task_start + 1:]:
            try:
                if msg.role == "assistant":
                    previous_tool = json.loads(msg.content).get("action")
                elif msg.content.startswith("observation:"):
                    observed = json.loads(msg.content.split(":", 1)[1])
                    successful_discovery |= previous_tool in discovery_tools and observed.get("status", {}).get("code") == 200
            except (ValueError, AttributeError):
                pass
        finish_enabled = not self.finish_guard or successful_discovery
        contract = CONTRACT if self.guidance else BASIC_CONTRACT
        if self.time_plan is not None and self.time_plan.rows:
            from smarthome_agent_rl.time_plan import OUTPUT_CONTRACT
            contract = OUTPUT_CONTRACT
        if not self.finish_guard:
            contract = contract.replace("Discovery is\nrequired before finish is available. ", "")
        converted = []
        for msg in messages:
            content = msg.content
            if msg.role == "system":
                content += contract
            elif msg.role == "assistant":
                try:
                    body = json.loads(content)
                    args = body["action_input"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    content = json.dumps({"thought": body["thought"],
                        "call": {"tool": body["action"], "arguments": args}}, ensure_ascii=False)
                except (ValueError, TypeError, KeyError):
                    pass
            converted.append(ChatMessage(role=msg.role, content=content))
        recovery_hint = None
        if observations and self.recovery:
            try:
                observation = json.loads(observations[-1].split(":", 1)[1])
                if observation.get("error") is not None:
                    recovery_hint = "Correct the last reported error before retrying: " + json.dumps(observation["error"])
                    converted.append(ChatMessage(role="user", content=recovery_hint))
            except (ValueError, AttributeError):
                pass
        record = {"turn": len(self.audit) + 1, "finish_enabled": finish_enabled,
                  "settings": {"finish_guard": self.finish_guard, "recovery": self.recovery, "guidance": self.guidance},
                  "recovery_hint": recovery_hint, "extra_model_calls": 0}
        self.audit.append(record)
        schema = self.response_format(finish_enabled)
        if self.time_plan is not None and self.time_plan.rows:
            converted.append(ChatMessage(role='user', content=self.time_plan.prompt()))
            schema = self.time_plan.augment_schema(schema)
        if self.task_spec is not None:
            converted.append(ChatMessage(role='user', content=self.task_spec.prompt()))
            schema = self.task_spec.augment_schema(schema)
        try:
            raw = self.inner.generate(converted, response_format=schema)
            body = json.loads(raw)
            if self.time_plan is not None and self.time_plan.rows:
                body = self.time_plan.consume(body)
            draft = None
            if self.task_spec is not None:
                body, draft = self.task_spec.prepare(body)
            action = self.validate(body, finish_enabled)
            if self.time_plan is not None and action['action'] == 'finish':
                self.time_plan.check('finish', json.loads(action['action_input']))
            if draft is not None:
                self.task_spec.commit(draft, action, record['turn'])
                record['goal_refs'] = list(draft['goal_refs'])
            record["normalized_action"] = action
            return json.dumps(action, ensure_ascii=False)
        except (ValueError, TypeError, KeyError) as exc:
            record["validation_error"] = str(exc)
            if self.task_spec is not None:
                self.task_spec.errors.append(str(exc))
            # Upstream parser supplies an observation; no action is silently repaired/executed.
            return "{}"
        finally:
            if self.audit_fn:
                self.audit_fn(self.audit)
