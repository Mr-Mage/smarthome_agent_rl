"""Run actual simulator, vLLM, Lightning gateway and controller, with no trainer."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smarthome_agent_rl.tasks import make_task
from check_environment import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rollout.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model-path")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / args.config).read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    agent_root = Path(os.environ.get("AGENT_ROOT", str(root.parent)))
    model_path = args.model_path or str(agent_root / "models/Qwen2.5-1.5B-Instruct")
    sim = root / "deps/SimuHome"
    sim_python = root / ".venv-simuhome/bin/python"
    if not sim_python.exists() or not Path(model_path).exists():
        raise RuntimeError("Missing isolated simulator environment or local model")
    run = Path(args.run_dir).resolve() if args.run_dir else root / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    config_path = run / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    owned, handles, commands = [], [], []
    client = httpx.Client(timeout=20, trust_env=False)
    key = "smarthome-local-rollout"
    auth = {"Authorization": f"Bearer {key}"}
    gateway = config["gateway_url"]
    def launch(name, command, cwd, extra=None):
        logfile = (run / f"{name}.log").open("w")
        handles.append(logfile)
        commands.append({"service": name, "argv": command, "cwd": str(cwd), "env": extra or {}})
        (run / "commands.json").write_text(json.dumps(commands, indent=2))
        process = subprocess.Popen(command, cwd=cwd, env={**os.environ, **(extra or {})},
                                   stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        owned.append(process)
        return process
    def wait_ready(url, process, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Service exited {process.returncode}: {url}")
            try:
                if client.get(url).is_success:
                    return
            except httpx.TransportError:
                pass
            time.sleep(1)
        raise TimeoutError(f"Service startup timeout: {url}")
    def api(method, path, payload=None):
        resp = client.request(method, gateway + path, json=payload, headers=auth)
        resp.raise_for_status()
        return resp.json()
    def commit(path):
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    try:
        for url in [config["simulator_url"], config["model_endpoint"], gateway]:
            parsed = urlparse(url)
            if parsed.hostname != "127.0.0.1":
                raise ValueError("V0 managed stack requires loopback service URLs")
            with socket.socket() as probe:
                probe.bind((parsed.hostname, parsed.port))
        versions = {"agentlightning_commit": commit(agent_root / "agent-lightning"),
                    "simuhome_commit": commit(sim), "python": sys.version,
                    "model_path": model_path, "gpu": args.gpu, "training": False}
        versions["model_config_sha256"] = hashlib.sha256((Path(model_path) / "config.json").read_bytes()).hexdigest()
        (run / "versions.json").write_text(json.dumps(versions, indent=2))
        (run / "gpu_before.csv").write_text(subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used", "--format=csv"], text=True))
        (run / "dependencies.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        (run / "simulator-dependencies.txt").write_text(subprocess.check_output([str(sim_python), "-m", "pip", "freeze"], text=True))
        shutil.copytree(root / "smarthome_agent_rl", run / "code/smarthome_agent_rl", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(root / "scripts", run / "code/scripts", ignore=shutil.ignore_patterns("__pycache__"))
        sim_proc = launch("simulator", [str(sim_python), "-m", "uvicorn", "src.simulator.api.app:app",
            "--host", "127.0.0.1", "--port", str(urlparse(config["simulator_url"]).port)], sim)
        wait_ready(config["simulator_url"] + "/__health__", sim_proc, 60)
        (run / "environment_checks.json").write_text(json.dumps(check(config["simulator_url"]), indent=2))
        print("Simulator checks passed", flush=True)
        backend = launch("model", [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path, "--served-model-name", config["served_model"],
            "--host", "127.0.0.1", "--port", str(urlparse(config["model_endpoint"]).port),
            "--tensor-parallel-size", "1", "--max-model-len", "8192", "--max-num-seqs", "1",
            "--gpu-memory-utilization", "0.20", "--enforce-eager", "--disable-log-requests"], root,
            {"CUDA_VISIBLE_DEVICES": args.gpu})
        wait_ready(config["model_endpoint"] + "/models", backend, 300)
        print("H100 model ready", flush=True)
        server = launch("gateway", [str(Path(sys.executable).parent / "agl-server"), "host=127.0.0.1",
            f"port={urlparse(gateway).port}", f"key={key}",
            f"default_proxy.model_name={config['served_model']}", "default_proxy.val.temperature=0"], root)
        wait_ready(gateway + "/healthz", server, 60)
        api("POST", "/api/models", [{"model": config["served_model"], "endpoint": config["model_endpoint"], "version": 0}])
        controller = launch("controller", [str(Path(sys.executable).parent / "agl-controller"),
            "runner_type=local", "local_runner.maximum_size=1", "local_runner.poll_interval=1",
            f"agl_server.url={gateway}", f"agl_server.key={key}"], root)
        task = make_task(config["seed"])
        created = api("POST", "/api/rollouts", [{"input": {"task": task, "config": config, "output": str(run / "episode")},
            "is_train": False, "metadata": {"harness": "react", "split": "dev"},
            "config": {"timeout_seconds": 300, "local": {"agent_class": "smarthome_agent_rl.react:ReActAgent",
                       "env_map": {"SMARTHOME_TASK": "input.task", "SMARTHOME_CONFIG": "input.config", "SMARTHOME_OUTPUT": "input.output"}}}}])
        rollout_id = created[0]["rollout_id"]
        (run / "rollout_created.json").write_text(json.dumps(created, indent=2))
        deadline = time.monotonic() + 330
        while time.monotonic() < deadline:
            if controller.poll() is not None:
                raise RuntimeError("Lightning controller exited")
            detail = api("GET", f"/api/rollouts/{rollout_id}")
            if detail["rollout"]["status"]["state"] in {"succeeded", "failed"}:
                break
            time.sleep(1)
        else:
            raise TimeoutError("Rollout deadline exceeded")
        (run / "rollout.json").write_text(json.dumps(detail, indent=2))
        event_list = api("GET", f"/api/rollouts/{rollout_id}/events")
        (run / "lightning_events.json").write_text(json.dumps(event_list, indent=2))
        triplets = api("GET", f"/api/rollouts/{rollout_id}/events?format=triplet")
        (run / "lightning_triplets.json").write_text(json.dumps(triplets, indent=2))
        (run / "gpu_after.csv").write_text(subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used", "--format=csv"], text=True))
        if detail["rollout"]["status"]["state"] != "succeeded":
            raise RuntimeError(f"Lightning execution failed: {detail}")
        counts = dict(Counter(e["event_type"] for e in event_list))
        summary = json.loads((run / "episode/summary.json").read_text())
        assert counts.get("model_request", 0) >= 1 and counts.get("reward") == 1
        assert counts.get("environment_step") == summary["steps"]
        for e in triplets:
            if e["event_type"] == "model_request":
                assert e["data"]["prompt_token_ids"] and e["data"]["response_token_ids"]
        reward = [e["data"]["value"] for e in event_list if e["event_type"] == "reward"][0]
        assert abs(reward - summary["reward"]) < 1e-9
        if not summary["success"]:
            raise RuntimeError("Real rollout completed but task did not succeed; inspect saved trace")
        result = {"status": "passed", "rollout_id": rollout_id, "event_counts": counts, **summary}
        (run / "verification.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        (run / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2))
        raise
    finally:
        for process in reversed(owned):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        for handle in handles:
            handle.close()
        client.close()
        print(f"Run artifacts: {run}", flush=True)


if __name__ == "__main__":
    main()
