"""Checks against the actual SimuHome server, not a mock environment."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smarthome_agent_rl.adapter import SimuHomeAdapter
from smarthome_agent_rl.tasks import make_task
from smarthome_agent_rl.reward import check_goals


def check(base_url):
    task = make_task()
    first, second = [g["device_id"] for g in task["goals"]]
    env = SimuHomeAdapter(base_url, max_steps=8)
    def command(device, name):
        return {"tool": "execute_command", "args": {"device_id": device, "endpoint_id": 1,
                "cluster_id": "OnOff", "command_id": name, "args": {}}}
    try:
        env.reset(task)
        assert not any(c["satisfied"] for c in env.checks)
        missing = env.step(command("missing_device", "On"))
        assert missing["invalid"] and missing["reward_components"]["invalid"] == -0.1
        malformed = env.step({"tool": [], "args": {}})
        assert malformed["invalid"]
        readonly = env.step({"tool": "write_attribute", "args": {"device_id": first,
                            "endpoint_id": 1, "cluster_id": "OnOff", "attribute_id": "OnOff", "value": True}})
        assert readonly["invalid"]
        positive = env.step(command(first, "On"))
        assert positive["reward_components"]["progress"] == 0.1
        negative = env.step(command(first, "Off"))
        assert negative["reward_components"]["progress"] == -0.1
        env.step(command(first, "On"))
        final = env.step(command(second, "Off"))
        assert final["success"] and final["terminated"] and final["reward_components"]["success"] == 1
        env.reset(task)
        assert not any(c["satisfied"] for c in env.checks)
        stopped = env.step({"tool": "finish", "args": {}})
        assert stopped["terminated"] and not stopped["success"]
        env.max_steps = 1
        env.reset(task)
        cutoff = env.step({"tool": "get_home_state", "args": {}})
        assert cutoff["truncated"] and not cutoff["success"]
        bad = dict(task["goals"][0], value=1)
        assert not check_goals(final["state"], [bad])[0]["satisfied"]
        try:
            SimuHomeAdapter(base_url, max_steps=0)
        except ValueError:
            pass
        else:
            raise AssertionError("Zero step budget must be rejected")
        return {"passed": True, "checks": ["missing-device", "malformed-tool", "positive-step-budget", "read-only-attribute", "signed-progress",
                 "state-verified-success", "reset", "premature-finish", "truncation", "bool-type-check"]}
    finally:
        env.close()


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080/api"), indent=2))
