from urllib.parse import quote
import httpx

from .reward import Weights, check_goals, reward_parts


class InvalidAction(ValueError):
    pass


class SimuHomeAdapter:
    """One owned simulator per sequential runner; reset is global in upstream."""

    def __init__(self, base_url: str, max_steps: int = 12, weights: Weights | None = None):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=15, trust_env=False)
        self.max_steps = max_steps
        self.weights = weights or Weights()
        self.done = True

    def close(self):
        self.client.close()

    def request(self, method: str, path: str, body=None):
        response = self.client.request(method, path, json=body)
        if response.status_code in {400, 404, 409, 422}:
            raise InvalidAction(response.text)
        response.raise_for_status()
        wrapped = response.json()
        code = wrapped["status"]["code"]
        if code in {400, 404, 409, 422}:
            raise InvalidAction(str(wrapped.get("error")))
        if code != 200 or wrapped.get("error") is not None:
            raise RuntimeError(f"Simulator failure: {wrapped}")
        return wrapped["data"]

    def reset(self, task: dict):
        if task.get("schema") != "smarthome-task-v1" or not task.get("goals"):
            raise ValueError("Unsupported task schema or empty goals")
        self.task = task
        self.request("POST", "/simulation/reset", task["initial_home_config"])
        self.state = self.request("GET", "/home/state")
        self.checks = check_goals(self.state, task["goals"])
        self.steps, self.done = 0, False
        self.initial_state = self.state
        return self.observe()

    def observe(self):
        # V0 full device attributes, no privileged predicates or reward in the prompt.
        return {"rooms": self.state["rooms"]}

    def dispatch(self, action: dict):
        if not isinstance(action, dict) or set(action) != {"tool", "args"}:
            raise InvalidAction("Action must contain exactly tool and args")
        tool, args = action["tool"], action["args"]
        if not isinstance(tool, str):
            raise InvalidAction("tool must be a string")
        if not isinstance(args, dict):
            raise InvalidAction("args must be an object")
        specs = {
            "get_home_state": (set(), set()),
            "get_device_structure": ({"device_id"}, {"device_id"}),
            "execute_command": ({"device_id", "endpoint_id", "cluster_id", "command_id"},
                                {"device_id", "endpoint_id", "cluster_id", "command_id", "args"}),
            "write_attribute": ({"device_id", "endpoint_id", "cluster_id", "attribute_id", "value"},
                                {"device_id", "endpoint_id", "cluster_id", "attribute_id", "value"}),
            "finish": (set(), set()),
        }
        if tool not in specs:
            raise InvalidAction(f"Unsupported tool: {tool}")
        required, allowed = specs[tool]
        if not required <= set(args) or not set(args) <= allowed:
            raise InvalidAction(f"Required fields: {sorted(required)}; allowed: {sorted(allowed)}")
        for key in ["device_id", "cluster_id", "command_id", "attribute_id"]:
            if key in args and (not isinstance(args[key], str) or not args[key]):
                raise InvalidAction(f"{key} must be a nonempty string")
        if "endpoint_id" in args and (type(args["endpoint_id"]) is not int or args["endpoint_id"] < 0):
            raise InvalidAction("endpoint_id must be a nonnegative integer")
        if "args" in args and not isinstance(args["args"], dict):
            raise InvalidAction("command args must be an object")
        if tool == "finish":
            return {"finished": True}
        if tool == "get_home_state":
            return self.request("GET", "/home/state")
        device = quote(args["device_id"], safe="")
        if tool == "get_device_structure":
            return self.request("GET", f"/devices/{device}/structure")
        body = {k: v for k, v in args.items() if k != "device_id"}
        suffix = "commands" if tool == "execute_command" else "attributes/write"
        return self.request("POST", f"/devices/{device}/{suffix}", body)

    def step(self, action: dict, *, tokens: int = 0):
        if self.done:
            raise RuntimeError("Episode is finished; call reset")
        before = self.checks
        try:
            result, invalid = self.dispatch(action), False
        except InvalidAction as exc:
            result, invalid = {"error": str(exc)}, True
        # Transport/timeouts/5xx propagate: never label infrastructure errors as agent mistakes.
        self.state = self.request("GET", "/home/state")
        self.checks = check_goals(self.state, self.task["goals"])
        self.steps += 1
        success = all(c["satisfied"] for c in self.checks)
        finished = not invalid and action["tool"] == "finish"
        terminated = success or finished
        truncated = self.steps >= self.max_steps and not terminated
        self.done = terminated or truncated
        parts = reward_parts(before, self.checks, terminal=self.done, invalid=invalid,
                             tokens=tokens, weights=self.weights)
        return {"observation": self.observe(), "tool_result": result,
                "reward": sum(parts.values()), "reward_components": parts,
                "invalid": invalid, "success": success, "terminated": terminated,
                "truncated": truncated, "checks": self.checks, "state": self.state}
