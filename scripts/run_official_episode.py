"""Launch pristine ReAct with baseline or an explicitly selected external provider harness."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deps/SimuHome"))
sys.path.insert(0, str(ROOT))

from src.agents.providers import OpenAIChatProvider
from src.agents.strategies import ReActConfig, create_agent_strategy
from src.agents.tools import TOOL_REGISTRY, ToolConfig, set_tool_config
from smarthome_agent_rl.adapter import SimuHomeAdapter
from smarthome_agent_rl.reward import Weights, check_goals, reward_parts
from smarthome_agent_rl.accounting import pending_rejection
from smarthome_agent_rl.generation import generation_options, install_generation_options


def run(mode, *, task_failure_is_data=False):
    task = json.loads(os.environ["SMARTHOME_TASK"])
    config = json.loads(os.environ["SMARTHOME_CONFIG"])
    harness = config.get("harness", "upstream-react")
    if harness not in {"upstream-react", "structured-v1", "structured-no-gate", "structured-no-recovery", "structured-schema"}:
        raise ValueError(f"Unknown harness: {harness}")
    output = Path(os.environ["SMARTHOME_OUTPUT"])
    output.mkdir(parents=True, exist_ok=False)
    api_base = os.environ["AGL_OPENAI_BASE_URL"] if mode == "lightning" else config["model_endpoint"]
    api_key = os.environ["AGL_KEY"] if mode == "lightning" else "local-unused"
    env = SimuHomeAdapter(config["simulator_url"])
    weights = Weights(**config["reward"])
    event_client = httpx.Client(trust_env=False, timeout=15,
                               headers={"Authorization": f"Bearer {api_key}"})
    generation = generation_options(config)
    provider = OpenAIChatProvider(model=config["served_model"], temperature=generation["temperature"],
                                 seed=config["model_seed"], api_key=api_key,
                                 api_base=api_base, timeout=120)
    install_generation_options(provider, config)
    if mode == "direct" and config.get("diagnostic_return_token_ids"):
        original_create = provider._client.chat.completions.create
        def with_ids(**kwargs):
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "return_token_ids": True}
            return original_create(**kwargs)
        provider._client.chat.completions.create = with_ids
    model_calls, steps, events, observer_events = [], [], [], []
    total_reward = 0.0
    pending_call = None
    request_started = {}
    before_checks = []
    started = time.monotonic()

    def save(name, data):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def emit(kind, data):
        if mode == "lightning":
            response = event_client.post(os.environ["AGL_EVENT_URL"], json={"event_type": kind, "data": data})
            response.raise_for_status()

    def capture_response(response):
        # Observation-only HTTP hook: original provider builds and processes all requests.
        nonlocal pending_call
        response.read()
        body = response.json()
        request = json.loads(response.request.content)
        call = {"index": len(model_calls) + 1, "request": request,
                "http_status": response.status_code, "response": body,
                "duration_seconds": time.monotonic() - request_started.pop(id(response.request), time.monotonic())}
        model_calls.append(call)
        save("model_calls.json", model_calls)
        if response.is_success and body.get("choices"):
            pending_call = call

    provider._client._client.event_hooks["request"].append(lambda request: request_started.update({id(request): time.monotonic()}))
    provider._client._client.event_hooks["response"].append(capture_response)

    def trace(kind, payload, *, observer=False):
        nonlocal total_reward, before_checks, pending_call
        if observer:
            observer_events.append({"event": kind, "payload": payload, "source": "observer"})
            save("observer_events.json", observer_events)
        else:
            events.append({"event": kind, "payload": payload})
            save("upstream_events.json", events)
        if kind not in {"observation", "finish", "rejected_action"}:
            return
        if pending_call is None:
            raise RuntimeError("Upstream observation without a recorded model response")
        raw = pending_call["response"]["choices"][0]["message"]["content"]
        try:
            action = json.loads(raw)
            model_action = action
            if harness.startswith("structured-") and "call" in action:
                from smarthome_agent_rl.structured import canonical_action
                action = canonical_action(action)
        except (ValueError, TypeError, KeyError):
            action = {"unparsed": raw}
            model_action = action
        try:
            observation = json.loads(payload) if kind in {"observation", "rejected_action"} else None
        except ValueError:
            observation = {"unparsed": payload}
        invalid, infrastructure_error = False, None
        if isinstance(observation, dict):
            code = observation.get("status", {}).get("code")
            if code in {400, 404, 409, 422}:
                invalid = True
            elif isinstance(code, int) and code >= 400:
                infrastructure_error = observation.get("error")
            elif observation.get("error") is not None:
                message = str(observation["error"])
                invalid = message.startswith(("Unknown tool", "Invalid structured", "finish tool"))
                if not invalid:
                    infrastructure_error = message
        state = env.request("GET", "/home/state")
        after_checks = check_goals(state, task["goals"])
        usage = pending_call["response"].get("usage")
        if not usage or "total_tokens" not in usage:
            raise RuntimeError("Model endpoint did not provide token usage")
        parts = reward_parts(before_checks, after_checks, terminal=kind == "finish",
                             invalid=invalid, tokens=usage["total_tokens"], weights=weights)
        step = {"turn": len(steps) + 1, "event": kind, "action": action,
                "source": "observer" if observer else "upstream", "agent_feedback_emitted": not observer,
                "model_action": model_action,
                "observation": observation, "state": state, "checks": after_checks,
                "usage": usage, "invalid": invalid, "infrastructure_error": infrastructure_error,
                "reward_components": parts, "reward": sum(parts.values())}
        steps.append(step)
        total_reward += step["reward"]
        before_checks = after_checks
        pending_call = None
        with (output / "trajectory.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(step, ensure_ascii=False) + "\n")
        emit("environment_step", step)

    error = None
    task_failure = False
    result = None
    try:
        env.reset(task)
        before_checks = env.checks
        save("task.json", task)
        save("initial_state.json", env.state)
        decision_provider = provider
        if harness.startswith("structured-"):
            from smarthome_agent_rl.structured import HARNESS_SETTINGS, StructuredProvider
            decision_provider = StructuredProvider(provider, audit_fn=lambda data: save("harness_events.json", data),
                                                   **HARNESS_SETTINGS[harness])
            save("structured_schema.json", decision_provider.response_format())
        save("upstream_contract.json", {"agent_class": "src.agents.strategies.react_agent.ReActAgent",
             "provider_class": "src.agents.providers.openai_provider.OpenAIChatProvider",
             "tool_names": sorted(TOOL_REGISTRY), "max_steps": config["max_steps"],
             "model_seed": config["model_seed"], "temperature": generation["temperature"],
             "generation": generation, "baseline_id": config.get("baseline_id"),
             "harness": harness, "prompt_changes": harness != "upstream-react", "tool_changes": False,
             "decision_provider": type(decision_provider).__name__,
             "policy_changes": harness != "upstream-react", "loop_changes": False,
             "http_observer": "response hook on original provider client", "trace_hook": "ReActConfig.trace_fn"})
        database = None
        if config.get("retrieval"):
            from smarthome_agent_rl.retrieval import load_retrieval
            database = load_retrieval(config["retrieval"], output / "retrieval_events.json")
        set_tool_config(ToolConfig(base_url=config["simulator_url"], timeout=15, db=database))
        emit("environment_reset", {"task_id": task["task_id"], "state": env.state, "seed": task["seed"]})
        agent = create_agent_strategy("react", llm=decision_provider,
            config=ReActConfig(max_steps=config["max_steps"], show_assistant_raw=True, trace_fn=trace))
        result = agent.run(task["instruction"], user_location=task.get("user_location", next(iter(task["initial_home_config"]["rooms"]))),
                           current_time=task["initial_home_config"]["base_time"])
        save("upstream_result.json", asdict(result))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        task_failure = type(exc).__name__ == "AgentExecutionError" and (
            "explicit finish action is required" in str(exc)
            or "consecutive failures" in str(exc))
        save("error.json", error)
    finally:
        # Original strict ReAct throws on the third rejection before emitting an
        # observation. Record that already-completed model turn without feeding
        # extra observations to the agent or changing its control flow.
        rejection = pending_rejection(error, events) if pending_call is not None else None
        if rejection is not None:
            trace("rejected_action", json.dumps(rejection), observer=True)
        state = env.request("GET", "/home/state")
        checks = check_goals(state, task["goals"])
        tokens = sum(c["response"].get("usage", {}).get("total_tokens", 0) for c in model_calls)
        summary = {"mode": mode, "harness": harness, "task_id": task["task_id"],
                   "execution_completed": error is None, "goal_success": all(c["satisfied"] for c in checks),
                   "success": error is None and result is not None and all(c["satisfied"] for c in checks),
                   "finished": result is not None, "steps": len(steps), "model_calls": len(model_calls),
                   "tool_calls": sum(e["event"] == "action" for e in events),
                   "tool_feedbacks": sum(s["event"] == "observation" for s in steps),
                   "metrics_version": 4, "model_rejections_without_feedback": len(observer_events),
                   "invalid_actions": sum(s["invalid"] for s in steps),
                   "infrastructure_errors": sum(s["infrastructure_error"] is not None for s in steps),
                   "total_tokens": tokens, "reward": total_reward, "checks": checks,
                   "duration_seconds": time.monotonic() - started, "error": error, "training": False,
                   "baseline_id": config.get("baseline_id"), "run_purpose": config.get("run_purpose", "historical_dev"),
                   "task_failure_is_data": task_failure_is_data}
        summary["task_failure"] = task_failure
        save("final_state.json", state)
        save("summary.json", summary)
        if error is None or task_failure:
            emit("reward", {"value": total_reward, "source": "environment_verifier", "reason": f"{harness}-environment-v1"})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        env.close()
        provider._client.close()
        event_client.close()
    # Task failures are data; infrastructure/provider/agent execution failures fail the worker.
    if error is not None and not (task_failure and task_failure_is_data):
        raise RuntimeError(f"Upstream baseline execution failed: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "lightning"], required=True)
    parser.add_argument("--task-failure-is-data", action="store_true")
    args = parser.parse_args()
    run(args.mode, task_failure_is_data=args.task_failure_is_data)
