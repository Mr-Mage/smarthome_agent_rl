"""Paired dev tasks: pristine upstream and external structured harness, both Lightning."""
import argparse
from collections import Counter
import copy
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

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.reward import check_goals
from smarthome_agent_rl.generation import generation_options
from smarthome_agent_rl.p1 import schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/dev-suite-v2.json")
    parser.add_argument("--config", default="configs/official-baseline.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--harnesses", nargs="+", default=["upstream-react", "structured-v1"],
        choices=["upstream-react", "structured-v1", "structured-no-gate", "structured-no-recovery", "structured-schema"])
    parser.add_argument("--task-order", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--variant-order-offset", type=int, default=0)
    parser.add_argument("--diagnostic-mode", choices=["initial", "restart", "chain"])
    parser.add_argument("--baseline-run", default="runs/b0-qwen35-9b-20261003-03")
    args = parser.parse_args()
    assert len(set(args.harnesses)) == len(args.harnesses) and len(args.harnesses) >= 1
    config = json.loads((ROOT / args.config).read_text())
    generation = generation_options(config)
    tasks = json.loads((ROOT / args.suite).read_text())
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    for task in tasks:
        if task.get("schema") != "smarthome-task-v1" or task.get("split") != "dev":
            raise ValueError("Independent dev fixtures only")
        task["initial_home_config"]["enable_aggregators"] = True
        for room in task["initial_home_config"]["rooms"].values():
            room["state"] = {"temperature": 2400.0, "humidity": 5000.0, "illuminance": 100.0, "pm10": 30.0}
    actual_schedule = schedule(tasks, args.harnesses, args.task_order, args.variant_order_offset)
    if args.task_order == "reverse":
        tasks = list(reversed(tasks))
    run = Path(args.run_dir).resolve()
    agent_root, sim = ROOT.parent, ROOT / "deps/SimuHome"
    model_path = Path(config.get("model_path", agent_root / "models/Qwen2.5-1.5B-Instruct"))
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    tensor_parallel = int(config.get("tensor_parallel_size", 1))
    model_python = config.get("model_python", sys.executable)
    model_environment = config.get("model_environment", {})
    allowed_model_environment = {"CUDA_HOME", "CUDA_PATH", "FLASHINFER_WORKSPACE_BASE", "MAX_JOBS"}
    if not isinstance(model_environment, dict) or any(
        name not in allowed_model_environment or not isinstance(value, str) or not value
        for name, value in model_environment.items()
    ):
        raise ValueError("Invalid model process environment override")
    if len(args.gpu.split(",")) != tensor_parallel:
        raise ValueError("GPU allocation must match tensor_parallel_size")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if config.get("require_verified_model"):
        marker_path = model_path / "download-verification.json"
        marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
        if not (marker.get("verified") is True and marker.get("repository") == config["model_repo"]
                and marker.get("revision") == config["model_revision"]):
            raise RuntimeError("Pinned model download is not fully verified; use the engineering 1.5B config meanwhile")
    run.mkdir(parents=True, exist_ok=False)
    owned, handles, commands, results = [], [], [], []
    key = "smarthome-local-rollout"
    auth = {"Authorization": f"Bearer {key}"}
    client = httpx.Client(trust_env=False, timeout=20)
    def save(path, value):
        target = run / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    def launch(name, cmd, cwd, extra=None):
        logfile = (run / (name + ".log")).open("w")
        handles.append(logfile)
        extra = extra or {}
        commands.append({"name": name, "argv": list(map(str, cmd)), "cwd": str(cwd), "env": extra})
        save("commands.json", commands)
        proc = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env={**os.environ, **extra},
                                stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        owned.append(proc)
        return proc
    def wait_ready(url, proc, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
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
        dirty = subprocess.check_output(["git", "-C", str(sim), "status", "--porcelain"], text=True)
        assert not dirty
        for url in [config["simulator_url"], config["model_endpoint"], config["gateway_url"]]:
            parsed = urlparse(url)
            assert parsed.hostname == "127.0.0.1"
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((parsed.hostname, parsed.port))
        save("config.json", config)
        if config.get("require_verified_model"):
            save("model_download_verification.json", marker)
        save("harnesses.json", args.harnesses)
        save("tasks.json", tasks)
        save("schedule.json", actual_schedule)
        save("purpose.json", {"purpose": "diagnostic_only" if args.diagnostic_mode else "paired_formal",
            "task_order": args.task_order, "variant_order_offset": args.variant_order_offset})
        save("run_protocol.json", {"baseline_id": config.get("baseline_id"),
            "run_purpose": config.get("run_purpose", "historical_dev"), "generation": generation,
            "model_seed": config["model_seed"], "engine_seed": config.get("engine_seed", 0),
            "max_steps": config["max_steps"], "model_context": config["model_context"],
            "task_ids": [t["task_id"] for t in tasks], "harnesses": args.harnesses,
            "task_sha256": hashlib.sha256((run / "tasks.json").read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((run / "config.json").read_bytes()).hexdigest(),
            "training": False, "official_reproduction": False})
        source_files = ["src/agents/strategies/react_agent.py", "src/agents/providers/openai_provider.py",
                        "src/agents/tools.py", "prompts/agents/react.py"]
        save("versions.json", {"simuhome_commit": subprocess.check_output(["git", "-C", str(sim), "rev-parse", "HEAD"], text=True).strip(),
            "baseline_source_sha256": {p: hashlib.sha256((sim / p).read_bytes()).hexdigest() for p in source_files},
            "lightning_commit": subprocess.check_output(["git", "-C", str(agent_root / "agent-lightning"), "rev-parse", "HEAD"], text=True).strip(),
            "model": str(model_path), "gpu": args.gpu, "tensor_parallel_size": tensor_parallel, "training": False})
        for label, interpreter in [("baseline", ROOT / ".venv-baseline/bin/python"), ("lightning", sys.executable), ("model", model_python)]:
            (run / f"{label}-dependencies.txt").write_text(subprocess.check_output([str(interpreter), "-m", "pip", "freeze"], text=True))
        for directory in ["smarthome_agent_rl", "scripts", "tests", "configs"]:
            shutil.copytree(ROOT / directory, run / "code" / directory, ignore=shutil.ignore_patterns("__pycache__"))
        simulator = launch("simulator", [ROOT / ".venv-simuhome/bin/python", "-m", "uvicorn", "src.simulator.api.app:app",
            "--host", "127.0.0.1", "--port", urlparse(config["simulator_url"]).port], sim)
        wait_ready(config["simulator_url"] + "/__health__", simulator, 60)
        if config.get("retrieval"):
            retrieval = config["retrieval"]
            embedding_service = launch("doc-embeddings", [retrieval["python"], ROOT / "scripts/serve_doc_embeddings.py",
                "--model", retrieval["model_path"], "--port", urlparse(retrieval["endpoint"]).port], ROOT,
                {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
            wait_ready(retrieval["endpoint"] + "/health", embedding_service, 120)
            health = client.get(retrieval["endpoint"] + "/health").json()
            assert health["device"] == "cpu" and health["dimension"] == 384
            save("retrieval_environment_audit.json", {"health": health, "config": retrieval,
                "model_verification": json.loads((Path(retrieval["model_path"]) / "download-verification.json").read_text()),
                "index_verification": json.loads((Path(retrieval["index_path"]) / "verification.json").read_text()),
                "pid": embedding_service.pid, "verified": True})
            (run / "retrieval-dependencies.txt").write_text(subprocess.check_output([retrieval["python"], "-m", "pip", "freeze"], text=True))
        backend = launch("model", [model_python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path, "--served-model-name", config["served_model"],
            "--host", "127.0.0.1", "--port", urlparse(config["model_endpoint"]).port,
            "--max-model-len", config["model_context"], "--max-num-seqs", 1,
            "--gpu-memory-utilization", str(config.get("gpu_memory_utilization", 0.20)),
            "--tensor-parallel-size", tensor_parallel, "--seed", config.get("engine_seed", 0),
            "--enforce-eager", config.get("model_log_requests_flag", "--disable-log-requests"),
            *config.get("model_extra_args", [])], ROOT,
            {"CUDA_VISIBLE_DEVICES": args.gpu, **model_environment})
        wait_ready(config["model_endpoint"] + "/models", backend, config.get("model_startup_timeout", 300))
        gateway = launch("gateway", [Path(sys.executable).parent / "agl-server", "host=127.0.0.1",
            f"port={urlparse(config['gateway_url']).port}", f"key={key}",
            f"default_proxy.model_name={config['served_model']}",
            f"default_proxy.val.temperature={generation['temperature']}"], ROOT)
        wait_ready(config["gateway_url"] + "/healthz", gateway, 60)
        api("POST", "/api/models", [{"model": config["served_model"], "endpoint": config["model_endpoint"], "version": 0}])
        controller = launch("controller", [Path(sys.executable).parent / "agl-controller", "runner_type=local",
            "local_runner.maximum_size=1", "local_runner.poll_interval=1",
            f"agl_server.url={config['gateway_url']}", f"agl_server.key={key}"], ROOT,
            {"OPENAI_API_KEY": "local-unused", "OPENAI_BASE_URL": config["model_endpoint"], "OPENAI_API_BASE": config["model_endpoint"]})
        save("service_start.json", {"model_pid": backend.pid, "gateway_pid": gateway.pid,
            "started_at": time.time(), "run": str(run)})
        process_environment = dict(item.split("=", 1) for item in Path(f"/proc/{backend.pid}/environ").read_text().split("\0") if "=" in item)
        actual_environment = {k: process_environment.get(k) for k in ["CUDA_VISIBLE_DEVICES", *model_environment]}
        assert actual_environment == {"CUDA_VISIBLE_DEVICES": args.gpu, **model_environment}
        save("model_environment_audit.json", {"pid": backend.pid,
            "executable": str(Path(f"/proc/{backend.pid}/exe").resolve()),
            "configured_executable": str(Path(model_python).resolve()),
            "argv": Path(f"/proc/{backend.pid}/cmdline").read_text().split("\0"),
            "environment": actual_environment, "verified": True})
        if args.diagnostic_mode:
            from p1_diagnostics import perform
            perform(args.diagnostic_mode, Path(args.baseline_run).resolve(), run, config,
                    api, launch, controller, save)
            return
        print(f"H100 ready: {len(tasks)} fixed tasks x {len(args.harnesses)} harnesses, all through Lightning", flush=True)
        for index, task in enumerate(tasks):
            # Alternate order to reduce systematic cache/warm-up bias.
            shift = (index + args.variant_order_offset) % len(args.harnesses)
            variants = args.harnesses[shift:] + args.harnesses[:shift]
            for harness in variants:
                relative = f"episodes/{index:02d}-{task['task_id']}/{harness}"
                output = run / relative / "episode"
                variant_config = {**config, "harness": harness}
                created = api("POST", "/api/rollouts", [{"input": {"task": task, "config": variant_config, "output": str(output)},
                    "is_train": False, "metadata": {"harness": harness, "split": "dev", "task_id": task["task_id"]},
                    "config": {"timeout_seconds": 600, "local": {"agent_class": config.get("lightning_agent_class", "smarthome_agent_rl.official:OfficialBaselineAgent"),
                        "env_map": {"SMARTHOME_TASK": "input.task", "SMARTHOME_CONFIG": "input.config", "SMARTHOME_OUTPUT": "input.output"}}}}])
                rid = created[0]["rollout_id"]
                save(relative + "/rollout_created.json", created)
                deadline = time.monotonic() + 630
                while time.monotonic() < deadline:
                    if controller.poll() is not None:
                        raise RuntimeError("Controller exited")
                    detail = api("GET", f"/api/rollouts/{rid}")
                    if detail["rollout"]["status"]["state"] in {"succeeded", "failed"}:
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError(rid)
                save(relative + "/rollout.json", detail)
                events = api("GET", f"/api/rollouts/{rid}/events")
                save(relative + "/lightning_events.json", events)
                triplets = api("GET", f"/api/rollouts/{rid}/events?format=triplet")
                save(relative + "/lightning_triplets.json", triplets)
                summary = json.loads((output / "summary.json").read_text())
                record = {"task_id": task["task_id"], "harness": harness, "rollout_id": rid,
                          "lightning_state": detail["rollout"]["status"]["state"], "path": relative, "summary": summary}
                results.append(record)
                save("results.json", results)
                print(json.dumps({"task": task["task_id"], "harness": harness,
                    "success": summary["success"], "steps": summary["steps"], "invalid": summary["invalid_actions"],
                    "reward": summary["reward"], "error": summary["error"]}), flush=True)
                if (summary["error"] and not summary.get("task_failure")) or summary["infrastructure_errors"] or record["lightning_state"] != "succeeded":
                    trajectory = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
                    save("incomplete_round.json", {"path": relative, "summary_error": summary["error"],
                        "lightning_state": record["lightning_state"], "completed_episodes": len(results),
                        "infrastructure_evidence": [{"turn": s["turn"], "action": s["action"],
                            "observation": s["observation"]} for s in trajectory if s["infrastructure_error"]]})
                    raise RuntimeError(f"Incomplete round in {relative}: worker={record['lightning_state']}, "
                        f"infrastructure_errors={summary['infrastructure_errors']}, error={summary['error']}")
                counts = Counter(e["event_type"] for e in events)
                assert counts["environment_reset"] == counts["reward"] == 1
                assert counts["model_request"] == summary["model_calls"] == counts["environment_step"] == summary["steps"]
                calls = json.loads((output / "model_calls.json").read_text())
                for call, event in zip(calls, [e for e in events if e["event_type"] == "model_request"]):
                    assert event["data"]["request"] == {**call["request"], "return_token_ids": True}
                for event in triplets:
                    if event["event_type"] == "model_request":
                        assert event["data"]["prompt_token_ids"] and event["data"]["response_token_ids"]
                reward = next(e["data"]["value"] for e in events if e["event_type"] == "reward")
                assert abs(reward - summary["reward"]) < 1e-9
                checks = check_goals(json.loads((output / "final_state.json").read_text()), task["goals"])
                assert checks == summary["checks"]
                save(relative + "/verification.json", {"artifact_checks_passed": True, "task_success": summary["success"],
                    "event_counts": dict(counts), "reward": reward, "gateway_preserves_request": True})
        aggregation = {}
        for harness in args.harnesses:
            ss = [r["summary"] for r in results if r["harness"] == harness]
            aggregation[harness] = {"episodes": len(ss), "successes": sum(s["success"] for s in ss),
                "success_rate": sum(s["success"] for s in ss) / len(ss),
                "model_calls": sum(s["model_calls"] for s in ss), "invalid_actions": sum(s["invalid_actions"] for s in ss),
                "tokens": sum(s["total_tokens"] for s in ss), "reward_sum": sum(s["reward"] for s in ss)}
        assert not subprocess.check_output(["git", "-C", str(sim), "status", "--porcelain"], text=True)
        save("comparison.json", {"aggregation": aggregation, "results": results, "upstream_pristine": True,
            "same_model_and_budget": True, "official_benchmark": False, "training": False})
        print(json.dumps(aggregation, indent=2), flush=True)
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
