"""Compare unchanged upstream ReAct directly and through Lightning, without training."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.tasks import make_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/official-baseline.json")
    parser.add_argument("--run-dir")
    parser.add_argument("--task-file", help="Own dev fixture; official benchmark is not accepted")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    run = Path(args.run_dir).resolve() if args.run_dir else ROOT / "runs" / ("official-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    agent_root = ROOT.parent
    model_path = agent_root / "models/Qwen2.5-1.5B-Instruct"
    sim = ROOT / "deps/SimuHome"
    baseline_python = ROOT / ".venv-baseline/bin/python"
    owned, handles, commands = [], [], []
    client = httpx.Client(trust_env=False, timeout=20)
    key = "smarthome-local-rollout"
    auth = {"Authorization": f"Bearer {key}"}
    def save(name, value):
        (run / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    def git(*gitargs):
        return subprocess.check_output(["git", "-C", str(sim), *gitargs], text=True).strip()
    def launch(name, cmd, cwd, extra=None):
        logfile = (run / f"{name}.log").open("w")
        handles.append(logfile)
        extra = extra or {}
        commands.append({"service": name, "argv": list(map(str, cmd)), "cwd": str(cwd), "env": extra})
        save("commands.json", commands)
        process = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env={**os.environ, **extra},
                                   stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        owned.append(process)
        return process
    def wait_ready(url, process, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Service exited: {url}")
            try:
                if client.get(url).is_success:
                    return
            except httpx.TransportError:
                pass
            time.sleep(1)
        raise TimeoutError(url)
    def api(method, path, body=None):
        response = client.request(method, config["gateway_url"] + path, json=body, headers=auth)
        response.raise_for_status()
        return response.json()
    try:
        if git("status", "--porcelain"):
            raise RuntimeError("Baseline checkout must be pristine")
        for url in [config["simulator_url"], config["gateway_url"], config["model_endpoint"]]:
            parsed = urlparse(url)
            if parsed.hostname != "127.0.0.1":
                raise ValueError("Only local endpoints allowed in this stack")
            with socket.socket() as probe:
                probe.bind((parsed.hostname, parsed.port))
        source_files = ["src/agents/strategies/react_agent.py", "src/agents/providers/openai_provider.py",
                        "src/agents/tools.py", "prompts/agents/react.py"]
        save("versions.json", {"simuhome_commit": git("rev-parse", "HEAD"),
             "simuhome_dirty": False, "agentlightning_commit": subprocess.check_output(
                 ["git", "-C", str(agent_root / "agent-lightning"), "rev-parse", "HEAD"], text=True).strip(),
             "baseline_source_sha256": {p: hashlib.sha256((sim / p).read_bytes()).hexdigest() for p in source_files},
             "model_path": str(model_path), "gpu": args.gpu, "training": False})
        save("config.json", config)
        for label, interpreter in [("baseline", baseline_python), ("lightning", sys.executable)]:
            (run / f"{label}-dependencies.txt").write_text(subprocess.check_output(
                [str(interpreter), "-m", "pip", "freeze"], text=True))
        shutil.copytree(ROOT / "smarthome_agent_rl", run / "code/smarthome_agent_rl", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "scripts", run / "code/scripts", ignore=shutil.ignore_patterns("__pycache__"))
        simproc = launch("simulator", [ROOT / ".venv-simuhome/bin/python", "-m", "uvicorn", "src.simulator.api.app:app",
            "--host", "127.0.0.1", "--port", urlparse(config["simulator_url"]).port], sim)
        wait_ready(config["simulator_url"] + "/__health__", simproc, 60)
        backend = launch("model", [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path, "--served-model-name", config["served_model"],
            "--host", "127.0.0.1", "--port", urlparse(config["model_endpoint"]).port,
            "--max-model-len", config["model_context"], "--max-num-seqs", 1,
            "--gpu-memory-utilization", "0.20", "--enforce-eager", "--disable-log-requests"], ROOT,
            {"CUDA_VISIBLE_DEVICES": args.gpu})
        wait_ready(config["model_endpoint"] + "/models", backend, 300)
        print("H100 model ready; running unchanged upstream baseline directly", flush=True)
        task = json.loads(Path(args.task_file).read_text()) if args.task_file else make_task(config["seed"])
        if task.get("split") != "dev" or task.get("schema") != "smarthome-task-v1":
            raise ValueError("Expected an independent smarthome-task-v1 dev fixture")
        # Official baseline queries room state. Provide a standard room with aggregators.
        task["initial_home_config"]["enable_aggregators"] = True
        for room in task["initial_home_config"]["rooms"].values():
            room["state"] = {"temperature": 2400.0, "humidity": 5000.0, "illuminance": 100.0, "pm10": 30.0}
        save("task.json", task)
        worker_env = {"SMARTHOME_TASK": json.dumps(task), "SMARTHOME_CONFIG": json.dumps(config),
            "SMARTHOME_OUTPUT": str(run / "direct"), "PYTHONPATH": str(ROOT) + os.pathsep + str(sim),
            "OPENAI_API_KEY": "local-unused", "OPENAI_BASE_URL": config["model_endpoint"],
            "OPENAI_API_BASE": config["model_endpoint"]}
        directproc = launch("direct", [baseline_python, ROOT / "scripts/run_official_episode.py", "--mode", "direct"], ROOT, worker_env)
        try:
            direct_exit = directproc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(directproc.pid, signal.SIGKILL)
            directproc.wait(timeout=5)
            direct_exit = 124
        save("direct_exit.json", {"exit_code": direct_exit})
        print(f"Direct baseline finished (exit={direct_exit}); migrating same agent to Lightning", flush=True)
        server = launch("gateway", [Path(sys.executable).parent / "agl-server", "host=127.0.0.1",
            f"port={urlparse(config['gateway_url']).port}", f"key={key}",
            f"default_proxy.model_name={config['served_model']}", "default_proxy.val.temperature=0"], ROOT)
        wait_ready(config["gateway_url"] + "/healthz", server, 60)
        api("POST", "/api/models", [{"model": config["served_model"], "endpoint": config["model_endpoint"], "version": 0}])
        controller = launch("controller", [Path(sys.executable).parent / "agl-controller", "runner_type=local",
            "local_runner.maximum_size=1", "local_runner.poll_interval=1",
            f"agl_server.url={config['gateway_url']}", f"agl_server.key={key}"], ROOT,
            {"OPENAI_API_KEY": "local-unused", "OPENAI_BASE_URL": config["model_endpoint"],
             "OPENAI_API_BASE": config["model_endpoint"]})
        created = api("POST", "/api/rollouts", [{"input": {"task": task, "config": config, "output": str(run / "lightning")},
            "is_train": False, "metadata": {"harness": "upstream-react", "split": "dev"},
            "config": {"timeout_seconds": 600, "local": {"agent_class": "smarthome_agent_rl.official:OfficialBaselineAgent",
                "env_map": {"SMARTHOME_TASK": "input.task", "SMARTHOME_CONFIG": "input.config", "SMARTHOME_OUTPUT": "input.output"}}}}])
        rid = created[0]["rollout_id"]
        save("rollout_created.json", created)
        deadline = time.monotonic() + 630
        while time.monotonic() < deadline:
            if controller.poll() is not None:
                raise RuntimeError("Controller exited")
            detail = api("GET", f"/api/rollouts/{rid}")
            if detail["rollout"]["status"]["state"] in {"succeeded", "failed"}:
                break
            time.sleep(1)
        else:
            raise TimeoutError("Lightning rollout deadline exceeded")
        save("rollout.json", detail)
        event_list = api("GET", f"/api/rollouts/{rid}/events")
        save("lightning_events.json", event_list)
        triplets = api("GET", f"/api/rollouts/{rid}/events?format=triplet")
        save("lightning_triplets.json", triplets)
        summaries = {mode: json.loads((run / mode / "summary.json").read_text())
                     for mode in ["direct", "lightning"] if (run / mode / "summary.json").exists()}
        comparison = {"rollout_id": rid, "direct_exit": direct_exit,
                      "lightning_state": detail["rollout"]["status"]["state"],
                      "event_counts": dict(Counter(e["event_type"] for e in event_list)),
                      "summaries": summaries, "upstream_dirty_after": bool(git("status", "--porcelain"))}
        if len(summaries) == 2:
            calls = {mode: json.loads((run / mode / "model_calls.json").read_text()) for mode in summaries}
            comparison["first_request_identical"] = calls["direct"][0]["request"] == calls["lightning"][0]["request"]
            comparison["final_goal_checks_identical"] = summaries["direct"]["checks"] == summaries["lightning"]["checks"]
            comparison["action_sequences"] = {mode: [json.loads(line)["action"].get("action", "unparsed")
                for line in (run / mode / "trajectory.jsonl").read_text().splitlines()] for mode in summaries}
        if detail["rollout"]["status"]["state"] == "succeeded":
            migrated = summaries["lightning"]
            counts = comparison["event_counts"]
            assert counts.get("model_request") == migrated["model_calls"]
            assert counts.get("environment_step") == migrated["steps"] and counts.get("reward") == 1
            for event in triplets:
                if event["event_type"] == "model_request":
                    assert event["data"]["prompt_token_ids"] and event["data"]["response_token_ids"]
            reward = [e["data"]["value"] for e in event_list if e["event_type"] == "reward"][0]
            assert abs(reward - migrated["reward"]) < 1e-9
        save("comparison.json", comparison)
        print(json.dumps(comparison, indent=2), flush=True)
        if direct_exit != 0 or detail["rollout"]["status"]["state"] != "succeeded":
            raise RuntimeError("Baseline execution failed; captured both phases for diagnosis")
    except Exception as exc:
        save("failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        for proc in reversed(owned):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
        for handle in handles:
            handle.close()
        client.close()
        print(f"Artifacts: {run}", flush=True)


if __name__ == "__main__":
    main()
