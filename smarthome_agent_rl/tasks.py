import random


def make_task(seed: int = 20261003) -> dict:
    """Own development task, generated without reading official benchmark."""
    rng = random.Random(seed)
    room = rng.choice(["bedroom", "study"])
    first, second = f"{room}_light_1", f"{room}_light_2"
    return {
        "schema": "smarthome-task-v1", "task_id": f"dev-lights-{seed}",
        "split": "dev", "seed": seed,
        "instruction": f"Turn on {first} and turn off {second} in the {room}.",
        "initial_home_config": {
            "base_time": "2026-10-03 00:00:00", "tick_interval": 0.1,
            "fast_forward": False, "enable_aggregators": False, "max_ticks": None,
            "rooms": {room: {"devices": [
                {"device_id": first, "device_type": "on_off_light",
                 "attributes": {"1.OnOff.OnOff": False}},
                {"device_id": second, "device_type": "on_off_light",
                 "attributes": {"1.OnOff.OnOff": True}},
            ]}},
        },
        "goals": [
            {"device_id": first, "attribute": "1.OnOff.OnOff", "op": "eq", "value": True},
            {"device_id": second, "attribute": "1.OnOff.OnOff", "op": "eq", "value": False},
        ],
    }
