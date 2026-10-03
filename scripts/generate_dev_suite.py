"""Generate independent static dev fixtures and separate reachability plans.

Does not read benchmark. Reference plans are used only by the simulator preflight,
never loaded by the agent, provider, Lightning input, or evaluation suite runner.
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POWER = "1.OnOff.OnOff"


def goal(device, attribute, value):
    return {"device_id": device, "attribute": attribute, "op": "eq", "value": value}


def command(device, cluster, name, **args):
    return {"tool": "execute_command", "args": {"device_id": device, "endpoint_id": 1,
        "cluster_id": cluster, "command_id": name, "args": args}}


def generate():
    tasks, references = [], {}
    pairs = [("bedroom", "study"), ("kitchen", "living_room"),
             ("bathroom", "bedroom"), ("study", "utility_room")]
    for i, (room, other) in enumerate(pairs):
        for family in ["multiroom", "group", "dimmer", "fan"]:
            room_devices, other_devices, goals, actions = [], [], [], []
            protected = f"{other}_on_off_light_7"
            other_devices.append({"device_id": protected, "device_type": "on_off_light",
                                  "attributes": {POWER: bool(i % 2)}})
            goals.append(goal(protected, POWER, bool(i % 2)))
            if family == "multiroom":
                a, b = f"{room}_on_off_light_3", f"{other}_on_off_light_2"
                room_devices.append({"device_id": a, "device_type": "on_off_light", "attributes": {POWER: False}})
                other_devices.append({"device_id": b, "device_type": "on_off_light", "attributes": {POWER: True}})
                instruction = f"Turn on {a} in {room} and turn off {b} in {other}. Leave all other devices unchanged."
                goals += [goal(a, POWER, True), goal(b, POWER, False)]
                actions = [command(a, "OnOff", "On"), command(b, "OnOff", "Off")]
            elif family == "group":
                target = bool(i % 2)
                for n in [2, 4, 6]:
                    device = f"{room}_on_off_light_{n}"
                    room_devices.append({"device_id": device, "device_type": "on_off_light", "attributes": {POWER: not target}})
                    goals.append(goal(device, POWER, target))
                    actions.append(command(device, "OnOff", "On" if target else "Off"))
                instruction = f"Turn {'on' if target else 'off'} all three lights in {room}. Leave devices in {other} unchanged."
            elif family == "dimmer":
                device, level = f"{room}_dimmable_light_3", [64, 96, 128, 192][i]
                room_devices.append({"device_id": device, "device_type": "dimmable_light",
                    "attributes": {POWER: False, "1.LevelControl.CurrentLevel": 32}})
                goals += [goal(device, POWER, True), goal(device, "1.LevelControl.CurrentLevel", level)]
                instruction = f"Turn on {device} in {room} and set its brightness to level {level} (out of 254), immediately without a transition. Leave all other devices unchanged."
                actions = [command(device, "OnOff", "On"),
                           command(device, "LevelControl", "MoveToLevel", Level=level, TransitionTime=0)]
            else:
                device, mode = f"{room}_fan_2", [1, 2, 3, 1][i]
                room_devices.append({"device_id": device, "device_type": "fan", "attributes": {POWER: False}})
                goals += [goal(device, POWER, True), goal(device, "1.FanControl.FanMode", mode)]
                instruction = f"Turn on {device} in {room} and set its fan speed to {['off', 'low', 'medium', 'high'][mode]}. Leave all other devices unchanged."
                actions = [command(device, "OnOff", "On"), {"tool": "write_attribute", "args": {
                    "device_id": device, "endpoint_id": 1, "cluster_id": "FanControl", "attribute_id": "FanMode", "value": mode}}]
            task_id = f"dev-expanded-{family}-{i}"
            task = {"schema": "smarthome-task-v1", "task_id": task_id, "split": "dev",
                "seed": 2026100300 + len(tasks), "family": family, "user_location": other,
                "instruction": instruction,
                "initial_home_config": {"base_time": "2026-10-03 00:00:00", "tick_interval": 0.1,
                    "fast_forward": False, "enable_aggregators": True, "max_ticks": None,
                    "rooms": {room: {"devices": room_devices}, other: {"devices": other_devices}}}, "goals": goals}
            for r in task["initial_home_config"]["rooms"].values():
                r["state"] = {"temperature": 2400.0, "humidity": 5000.0, "illuminance": 100.0, "pm10": 30.0}
            tasks.append(task)
            references[task_id] = actions
    return tasks, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="configs/dev-expanded-v1.json")
    parser.add_argument("--references", default="configs/dev-expanded-reference-plans.json")
    args = parser.parse_args()
    tasks, references = generate()
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / args.references).write_text(json.dumps(references, indent=2) + "\n", encoding="utf-8")
    # Stratified partition: both independent simulators see all four task families.
    for shard in range(2):
        subset = [copy.deepcopy(t) for j, t in enumerate(tasks) if (j // 4) % 2 == shard]
        target.with_name(f"{target.stem}-shard{shard}{target.suffix}").write_text(
            json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "families": sorted({t["family"] for t in tasks}), "source": "own generator; no benchmark input"}))


if __name__ == "__main__":
    main()
