import json
import os
from pathlib import Path

import httpx
from openai import OpenAI

from .adapter import SimuHomeAdapter
from .reward import Weights

SYSTEM = '''You control a symbolic smart home using ReAct. Decide one action, then read its observation.
Return exactly one JSON object with: thought (brief string), action (tool name), action_input (object).
Tools:
get_home_state: {}
get_device_structure: {"device_id": string}
execute_command: {"device_id": string, "endpoint_id": integer, "cluster_id": string, "command_id": string, "args": object}
write_attribute: {"device_id": string, "endpoint_id": integer, "cluster_id": string, "attribute_id": string, "value": any}
finish: {}
Inspect structure if unsure of commands. For on_off_light: endpoint_id=1, cluster_id="OnOff", command_id="On" or "Off", args={}.
OnOff.OnOff is read-only; use commands to change it. All device attributes are observed after each action.
Never invent tools or device IDs. Do not claim completion before observing the requested state.'''


def parse_action(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {"thought", "action", "action_input"}:
        raise ValueError("Expected thought, action, action_input")
    if not isinstance(obj["thought"], str) or not isinstance(obj["action"], str):
        raise ValueError("thought and action must be strings")
    args = obj["action_input"]
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError("action_input must be an object")
    return {"tool": obj["action"], "args": args}


class ReActAgent:
    """Entry point imported and launched by real Agent Lightning v1 local controller."""

    def run(self):
        task = json.loads(os.environ["SMARTHOME_TASK"])
        config = json.loads(os.environ["SMARTHOME_CONFIG"])
        output = Path(os.environ["SMARTHOME_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        env = SimuHomeAdapter(config["simulator_url"], config["max_steps"], Weights(**config["reward"]))
        model = OpenAI(base_url=os.environ["AGL_OPENAI_BASE_URL"], api_key=os.environ["AGL_KEY"],
                       max_retries=0, timeout=120, http_client=httpx.Client(trust_env=False))
        events = httpx.Client(headers={"Authorization": f"Bearer {os.environ['AGL_KEY']}"},
                              timeout=15, trust_env=False)
        def emit(kind, data):
            events.post(os.environ["AGL_EVENT_URL"], json={"event_type": kind, "data": data}).raise_for_status()
        total, tokens_total, invalid_total = 0.0, 0, 0
        try:
            initial = env.reset(task)
            (output / "task.json").write_text(json.dumps(task, indent=2))
            (output / "initial_state.json").write_text(json.dumps(env.initial_state, indent=2))
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps({"task": task["instruction"], "observation": initial})}]
            emit("environment_reset", {"task_id": task["task_id"], "seed": task["seed"], "state": env.state})
            with (output / "trajectory.jsonl").open("w", encoding="utf-8") as trace:
                while not env.done:
                    response = model.chat.completions.create(model="auto", messages=messages,
                        temperature=0, max_tokens=config["max_tokens"], logprobs=True)
                    raw = response.choices[0].message.content or ""
                    used = response.usage.total_tokens if response.usage else None
                    if used is None:
                        raise RuntimeError("Model endpoint omitted token usage")
                    tokens_total += used
                    parse_error = None
                    try:
                        action = parse_action(raw)
                    except (ValueError, TypeError) as exc:
                        parse_error = str(exc)
                        action = {"tool": "__parse_error__", "args": {}}
                    step = env.step(action, tokens=used)
                    invalid_total += int(step["invalid"])
                    total += step["reward"]
                    record = {"step": env.steps, "model_output": raw, "usage": response.usage.model_dump(),
                              "parse_error": parse_error, "action": action, **step}
                    trace.write(json.dumps(record) + "\n")
                    trace.flush()
                    emit("environment_step", record)
                    messages.extend([{"role": "assistant", "content": raw},
                        {"role": "user", "content": json.dumps({"tool_result": step["tool_result"], "observation": step["observation"]})}])
            summary = {"task_id": task["task_id"], "seed": task["seed"], "success": step["success"],
                       "steps": env.steps, "invalid_actions": invalid_total, "total_tokens": tokens_total,
                       "reward": total, "terminated": step["terminated"], "truncated": step["truncated"],
                       "checks": step["checks"], "harness": "react", "training": False}
            (output / "final_state.json").write_text(json.dumps(env.state, indent=2))
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            emit("reward", {"value": total, "source": "environment_verifier", "reason": "v1-success-progress-invalid-cost"})
            print(json.dumps(summary), flush=True)
        except Exception as exc:
            (output / "error.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}))
            raise
        finally:
            env.close()
            model.close()
            events.close()
