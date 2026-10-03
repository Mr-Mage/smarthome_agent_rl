from dataclasses import dataclass
import math


def check_goals(state: dict, goals: list[dict]) -> list[dict]:
    if not goals:
        raise ValueError("At least one goal is required")
    devices = {d["device_id"]: d for r in state["rooms"].values() for d in r.get("devices", [])}
    checks = []
    for goal in goals:
        found = goal["device_id"] in devices and goal["attribute"] in devices[goal["device_id"]]["attributes"]
        actual = devices.get(goal["device_id"], {}).get("attributes", {}).get(goal["attribute"])
        target, op = goal["value"], goal["op"]
        if op == "eq":
            ok = found and type(actual) is type(target) and actual == target
        elif op in {"le", "ge"}:
            numeric = type(actual) in (float, int) and type(target) in (float, int)
            ok = found and numeric and math.isfinite(actual) and math.isfinite(target)
            ok = ok and (actual <= target if op == "le" else actual >= target)
        else:
            raise ValueError(f"Unsupported goal op: {op}")
        checks.append({**goal, "actual": actual, "satisfied": bool(ok)})
    return checks


@dataclass(frozen=True)
class Weights:
    success: float = 1.0
    progress: float = 0.2
    invalid: float = 0.1
    action_cost: float = 0.01
    token_cost_per_1k: float = 0.001


def reward_parts(before: list[dict], after: list[dict], *, terminal: bool,
                 invalid: bool, tokens: int, weights: Weights) -> dict:
    # Signed potential difference prevents farming reward by undoing and redoing goals.
    potential = lambda checks: sum(c["satisfied"] for c in checks) / len(checks)
    return {
        "success": weights.success * float(terminal and all(c["satisfied"] for c in after)),
        "progress": weights.progress * (potential(after) - potential(before)),
        "invalid": -weights.invalid * int(invalid),
        "cost": -weights.action_cost - weights.token_cost_per_1k * tokens / 1000,
    }
