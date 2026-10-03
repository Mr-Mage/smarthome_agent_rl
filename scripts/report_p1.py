"""Report all repeated paired outcomes and costs; never select the best run."""
from collections import Counter, defaultdict
import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.p1 import B0, H1, paired_summary, digest, normalize, stats
from smarthome_agent_rl.structured import tool_schemas

def load(root, p):
    return json.loads((root / p).read_text(encoding="utf-8"))

def costs(rows):
    metrics = ("success", "goal_success", "model_calls", "tool_calls", "invalid_actions",
               "prompt_tokens", "completion_tokens", "total_tokens", "duration_seconds", "model_latency_seconds", "retrieval_calls", "retrieval_tokens", "retrieval_latency_seconds", "reward")
    return {"episodes": len(rows), **{m: sum(r[m] for r in rows) for m in metrics},
        "per_episode": {m: statistics.mean(r[m] for r in rows) if rows else None for m in metrics}}

def device_signatures():
    result = {}
    for source in (ROOT / "deps/SimuHome/src/simulator/domain/clusters").glob("*.py"):
        tree = ast.parse(source.read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            init = methods.get("__init__")
            if init is None:
                continue
            cluster, commands = None, None
            for n in ast.walk(init):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "__init__" and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                    cluster = n.args[0].value
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "commands" for t in n.targets) and isinstance(n.value, ast.Dict):
                    commands = n.value
            if not cluster or commands is None:
                continue
            for key, value in zip(commands.keys, commands.values):
                if not isinstance(key, ast.Constant) or not isinstance(value, ast.Attribute) or value.attr not in methods:
                    continue
                method = methods[value.attr]
                args = method.args.args[1:]
                result[(cluster, key.value)] = {"parameters": [a.arg for a in args],
                    "required": [a.arg for a in args[:len(args) - len(method.args.defaults)]],
                    "source": str(source.relative_to(ROOT)), "line": method.lineno,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "method": method.name,
                    "allows_extra": method.args.kwarg is not None}
    return result

def analyze_errors(steps, schemas, summary, signatures):
    evidence, seen, structures = [], Counter(), {}
    for step in steps:
        action = normalize(json.dumps(step["action"]))
        tool, args = action.get("tool"), action.get("arguments")
        obs = step["observation"]
        data = obs.get("data") if isinstance(obs, dict) else None
        if isinstance(data, dict) and "endpoints" in data:
            structures[data["device_id"]] = data
        if step["invalid"]:
            category, reason = "undetermined", "HTTP status alone does not identify device semantics"
            if tool is None:
                category, reason = "output_parse", "No parseable action representation"
            elif tool not in schemas or not isinstance(args, dict):
                category, reason = "outer_tool_arguments", "Unknown tool or argument object type"
            else:
                schema = schemas[tool]
                violations = []
                if set(schema["required"]) - set(args):
                    violations.append("missing " + str(sorted(set(schema["required"]) - set(args))))
                if set(args) - set(schema["properties"]):
                    violations.append("extra outer keys")
                for key, value in args.items():
                    kind = schema["properties"].get(key, {}).get("type")
                    if kind and not {"string": isinstance(value, str), "integer": type(value) is int,
                        "number": type(value) in (int, float), "object": isinstance(value, dict),
                        "array": isinstance(value, list)}[kind]:
                        violations.append(f"{key} must be {kind}")
                if violations:
                    category, reason = "outer_tool_arguments", "; ".join(violations)
                else:
                    structure = structures.get(args.get("device_id"), {})
                    cluster = structure.get("endpoints", {}).get(str(args.get("endpoint_id")), {}).get("clusters", {}).get(args.get("cluster_id"), {})
                    if tool == "write_attribute" and cluster.get("attributes", {}).get(args.get("attribute_id"), {}).get("readonly") is True:
                        category, reason = "readonly_attribute", "Earlier recorded get_device_structure marks this exact attribute readonly=true"
                    elif tool == "execute_command" and cluster and args.get("command_id") not in cluster.get("commands", []):
                        category, reason = "device_command_semantics", "Earlier device structure does not list this command"
                    elif tool == "execute_command" and cluster:
                        signature = signatures.get((args.get("cluster_id"), args.get("command_id")))
                        command_args = args.get("args")
                        if signature and isinstance(command_args, dict):
                            missing = set(signature["required"]) - set(command_args)
                            extra = set() if signature["allows_extra"] else set(command_args) - set(signature["parameters"])
                            if missing or extra:
                                category = "device_command_semantics"
                                reason = {"evidence": "Recorded device structure confirms this command; pinned simulator method signature rejects these keys",
                                          "missing": sorted(missing), "extra": sorted(extra), "signature": signature}
            signature = digest(action)
            seen[signature] += 1
            evidence.append({"turn": step["turn"], "category": category, "reason": reason,
                "action": action, "observation": obs, "repeated_same_invalid_action": seen[signature] > 1,
                "agent_feedback_emitted": step["agent_feedback_emitted"]})
            if seen[signature] > 1:
                evidence.append({"turn": step["turn"], "category": "error_recovery",
                    "reason": "Same invalid action repeated after an earlier tool rejection",
                    "action": action, "observation": obs, "recovered": False})
    if summary["finished"] and not summary["goal_success"]:
        evidence.append({"category": "early_finish", "reason": "Legal finish with unsatisfied final goals"})
    if summary["task_failure"]:
        evidence.append({"category": "error_recovery", "reason": "Task ended at agent limit/rejection", "error": summary["error"]})
    elif summary["success"] and any(step["invalid"] for step in steps):
        evidence.append({"category": "error_recovery", "reason": "Task eventually succeeded after an invalid action; this does not prove every rejected operation was individually recovered", "recovered": True})
    if any(s["infrastructure_error"] for s in steps):
        evidence.append({"category": "infrastructure", "reason": "Recorded infrastructure_error in trajectory"})
    return evidence

def main(destination, diagnostic, runs):
    destination.mkdir(parents=True, exist_ok=True)
    diag = load(diagnostic.parent, diagnostic.name)
    assert diag["artifact_checks_passed"] and diag["purpose"] == "diagnostic_only"
    rounds, per_round, errors = [], [], []
    schemas = tool_schemas()
    signatures = device_signatures()
    (destination / "device_semantics_evidence.json").write_text(json.dumps({f"{c}.{m}": v for (c, m), v in signatures.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    reference = load(runs[0], "config.json")
    reference_tasks = load(runs[0], "tasks.json")
    historical = ROOT / "runs/b0-qwen35-9b-20261003-03"
    accepted = load(historical, "config.json")
    assert {k: v for k, v in reference.items() if k not in {"run_purpose", "retrieval"}} == {k: v for k, v in accepted.items() if k != "run_purpose"}
    assert {t["task_id"]: t for t in reference_tasks} == {t["task_id"]: t for t in load(historical, "tasks.json")}
    for number, run in enumerate(runs, 1):
        verification = load(run, "verification.json")
        assert verification["artifact_checks_passed"] and verification["episodes"] == 32
        assert load(run, "purpose.json")["purpose"] == "paired_formal"
        assert load(run, "config.json") == reference
        retrieval_audit = load(run, "retrieval_environment_audit.json")
        assert retrieval_audit["verified"] and retrieval_audit["health"]["device"] == "cpu"
        assert retrieval_audit["model_verification"]["revision"] == reference["retrieval"]["model_revision"]
        assert retrieval_audit["index_verification"]["verified"]
        environment = load(run, "model_environment_audit.json")
        assert environment["verified"] and environment["executable"] == environment["configured_executable"]
        assert environment["environment"] == {"CUDA_VISIBLE_DEVICES": "0", **reference["model_environment"]}
        command = environment["argv"]
        assert command[command.index("--seed") + 1] == "0"
        assert command[command.index("--max-num-seqs") + 1] == "1"
        versions = load(run, "versions.json")
        historical_versions = load(historical, "versions.json")
        assert versions["baseline_source_sha256"] == historical_versions["baseline_source_sha256"]
        assert versions["simuhome_commit"] == historical_versions["simuhome_commit"]
        assert versions["lightning_commit"] == historical_versions["lightning_commit"]
        marker = load(run, "model_download_verification.json")
        assert marker["verified"] and marker["revision"] == reference["model_revision"] and marker["repository"] == reference["model_repo"]
        assert {t["task_id"]: t for t in load(run, "tasks.json")} == {t["task_id"]: t for t in reference_tasks}
        rows = []
        for record in load(run, "results.json"):
            prefix = record["path"] + "/episode/"
            summary, task, calls = load(run, prefix + "summary.json"), load(run, prefix + "task.json"), load(run, prefix + "model_calls.json")
            row = {"round": number, "task_id": task["task_id"], "family": task["family"],
                "harness": record["harness"], "source": str(run / prefix),
                **{m: summary[m] for m in ("success", "goal_success", "finished", "model_calls", "tool_calls",
                   "invalid_actions", "total_tokens", "duration_seconds", "reward")},
                "prompt_tokens": sum(c["response"]["usage"]["prompt_tokens"] for c in calls),
                "completion_tokens": sum(c["response"]["usage"]["completion_tokens"] for c in calls),
                "model_latency_seconds": sum(c["duration_seconds"] for c in calls)}
            retrieval_file = run / prefix / "retrieval_events.json"
            retrieval_events = json.loads(retrieval_file.read_text()) if retrieval_file.exists() else []
            row.update(retrieval_calls=len(retrieval_events), retrieval_tokens=sum(e["input_tokens"] for e in retrieval_events),
                       retrieval_latency_seconds=sum(e["duration_seconds"] for e in retrieval_events))
            rows.append(row)
            steps = [json.loads(line) for line in (run / (prefix + "trajectory.jsonl")).read_text().splitlines()]
            errors.append({"round": number, "task_id": task["task_id"], "harness": record["harness"],
                           "source": str(run / prefix), "evidence": analyze_errors(steps, schemas, summary, signatures)})
        rounds.append(rows)
        per_round.append({"round": number, "run": str(run), "harnesses": {h: {
            "all": costs([r for r in rows if r["harness"] == h]),
            "successful": costs([r for r in rows if r["harness"] == h and r["success"]]),
            "families": {f: costs([r for r in rows if r["harness"] == h and r["family"] == f])
                         for f in sorted({r["family"] for r in rows})}} for h in (B0, H1)}})
    assert load(runs[0], "schedule.json") == load(runs[2], "schedule.json")
    assert load(runs[1], "purpose.json")["task_order"] == "reverse"
    assert load(runs[1], "purpose.json")["variant_order_offset"] == 1
    report = {"artifact_checks_passed": True, "rounds": per_round, "task_pairs": paired_summary(rounds),
        "round_statistics": {h: {m: stats([r["harnesses"][h]["all"][m] for r in per_round])
            for m in ("success", "goal_success", "model_calls", "invalid_actions", "prompt_tokens",
                      "completion_tokens", "total_tokens", "duration_seconds", "reward")} for h in (B0, H1)},
        "records": [r for rs in rounds for r in rs], "diagnostic_cost": diag["cost"],
        "additional_pre_retrieval_chain_cost": diag.get("additional_pre_retrieval_chain_cost"),
        "retrieval_dependency_repair": {"same_for_both_harnesses": True,
            "model": reference["retrieval"], "original_index_overwritten": False,
            "indexing_cost": load(runs[0], "retrieval_environment_audit.json")["index_verification"]["indexing_cost"],
            "note": "P0 documentation retrieval was unavailable. User requested a real backend before resuming the original protocol; both groups use repaired local CPU BGE retrieval on identical upstream documents."},
        "history": {"P0_B0_success": "12/16", "included_in_statistics": False},
        "limits": ["16 development tasks repeated three times are 16 paired task units, not 48 independent samples",
            "H1 jointly changes representation, prompt contract, history serialization and schema; grammar effect is not isolated",
            "Order, warm state, restart and environmental time variation do not identify causal mechanisms",
            "Latency is measured at client/episode boundaries and excludes service startup"]}
    incomplete = ROOT / "runs/p1-paired-round1-20261003-01"
    overhead = []
    for r in load(incomplete, "results.json"):
        prefix = r["path"] + "/episode/"
        s = r["summary"]
        calls = load(incomplete, prefix + "model_calls.json")
        overhead.append({"task_id": r["task_id"], "harness": r["harness"], "source": str(incomplete / prefix),
            "model_calls": s["model_calls"], "prompt_tokens": sum(c["response"]["usage"]["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["response"]["usage"]["completion_tokens"] for c in calls),
            "total_tokens": s["total_tokens"], "duration_seconds": s["duration_seconds"],
            "infrastructure_errors": s["infrastructure_errors"], "included_in_formal_statistics": False})
    report["incomplete_round_overhead"] = {"episodes": len(overhead), "records": overhead,
        "totals": {m: sum(r[m] for r in overhead) for m in ("model_calls", "prompt_tokens", "completion_tokens", "total_tokens", "duration_seconds", "infrastructure_errors")},
        "reason": "Documentation retrieval unavailable; whole round stopped. New entire round was run after user-authorized retrieval repair."}
    for filename, value in (("paired_report.json", report), ("error_evidence.json", errors)):
        (destination / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    with (destination / "episode_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["records"][0]))
        writer.writeheader()
        writer.writerows(report["records"])
    lines = ["# P1：9B 最小 schema 三轮配对报告", "", "实际执行 16 项 dev × 两组 × 三轮 = 96 个 episode；三轮均独立核验通过。P0 的 12/16 仅作历史参照。", "",
        "H1 是对象输出表示与工具参数 schema 的组合改动；finish guard、恢复提示、额外指导均关闭。没有训练或扩展任务集。", "",
        "|轮|组|严格成功|goal_success|模型调用|工具调用|非法动作|prompt tokens|completion tokens|episode 秒|reward|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_round:
        for h in (B0, H1):
            c = r["harnesses"][h]["all"]
            lines.append(f"|{r['round']}|{'B0' if h == B0 else 'H1'}|{c['success']}/16|{c['goal_success']}|{c['model_calls']}|{c['tool_calls']}|{c['invalid_actions']}|{c['prompt_tokens']}|{c['completion_tokens']}|{c['duration_seconds']:.2f}|{c['reward']:.6f}|")
    b = report["round_statistics"][B0]["success"]
    h = report["round_statistics"][H1]["success"]
    lines += ["", f"B0 三轮平均 {b['mean']:.2f}/16，范围 {b['min']}–{b['max']}；H1 平均 {h['mean']:.2f}/16，范围 {h['min']}–{h['max']}。平均差 {(h['mean'] - b['mean']):+.2f} 项。该差值仅适用于这 16 项开发任务，不能据最佳一轮宣布普遍提升。", "",
        "|轮|组|成功任务数|成功任务调用|成功任务非法动作|成功 prompt|成功 completion|成功 episode 秒|成功 reward|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_round:
        for h in (B0, H1):
            c = r["harnesses"][h]["successful"]
            lines.append(f"|{r['round']}|{'B0' if h == B0 else 'H1'}|{c['episodes']}|{c['model_calls']}|{c['invalid_actions']}|{c['prompt_tokens']}|{c['completion_tokens']}|{c['duration_seconds']:.2f}|{c['reward']:.6f}|")
    lines += ["", "|轮|任务族|组|成功 / 4|goal_success|模型调用|非法动作|prompt|completion|秒|reward|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_round:
        for h in (B0, H1):
            for family, c in r["harnesses"][h]["families"].items():
                lines.append(f"|{r['round']}|{family}|{'B0' if h == B0 else 'H1'}|{c['success']}/4|{c['goal_success']}|{c['model_calls']}|{c['invalid_actions']}|{c['prompt_tokens']}|{c['completion_tokens']}|{c['duration_seconds']:.2f}|{c['reward']:.6f}|")
    lines += ["", "|任务|B0 三轮成功|H1 三轮成功|成功差均值|调用差均值 [范围]|token 差均值 [范围]|非法动作差均值 [范围]|秒差均值 [范围]|结果翻转 B0/H1|",
        "|---|---|---|---:|---|---|---|---|---|"]
    for task, item in report["task_pairs"].items():
        d = item["H1_minus_B0"]
        fmt = lambda m: f"{d[m]['mean']:+.2f} [{d[m]['min']:+.2f}, {d[m]['max']:+.2f}]"
        lines.append(f"|{task}|{item[B0]['success']['values']}|{item[H1]['success']['values']}|{d['success']['mean']:+.2f}|{fmt('model_calls')}|{fmt('total_tokens')}|{fmt('invalid_actions')}|{fmt('duration_seconds')}|{item['success_flips'][B0]}/{item['success_flips'][H1]}|")
    counts = Counter(e["category"] for row in errors for e in row["evidence"])
    combined = {variant: costs([row for row in report["records"] if row["harness"] == variant]) for variant in (B0, H1)}
    base, harness = combined[B0], combined[H1]
    lines += ["", "三轮累计成本（同一批 16 项任务的重复执行）：",
        f"B0/H1 严格成功 {base['success']}/{harness['success']}（各 48 次执行）；模型调用 {base['model_calls']}/{harness['model_calls']}；非法动作 {base['invalid_actions']}/{harness['invalid_actions']}；生成 tokens {base['total_tokens']}/{harness['total_tokens']}。",
        f"H1 相对 B0 的生成 token 成本变化 {(harness['total_tokens'] / base['total_tokens'] - 1) * 100:+.2f}%，episode 延迟变化 {(harness['duration_seconds'] / base['duration_seconds'] - 1) * 100:+.2f}%。延迟差包含运行顺序和环境时间变化。",
        "", "这三轮是否支持提升，应同时看逐任务成功变化、非法动作与成本；不能把累计 48 次执行视为 48 项独立任务。"]
    lines += ["", "错误证据分类（事件计数，类别不是互斥任务数）：", "", json.dumps(dict(counts), ensure_ascii=False), "",
        "输出解析、工具外层参数、设备命令语义、只读属性、错误恢复、提前结束与基础设施错误见 error_evidence.json。内层命令参数若没有原始服务错误或能力证据，标记 undetermined；不会只凭 HTTP 400 认定原因。重复非法动作单列，代表未避免同样的错误，不能等同于完整恢复率。", "",
        "全部任务与成功任务成本均含真实执行中的查询、非法动作和 finish；prompt/completion、逐任务三轮均值/范围、reward 差与结果翻转的完整值见 paired_report.json；每个 episode 的独立数据见 episode_metrics.csv。", "",
        "诊断成本单列：" + json.dumps(diag["cost"], ensure_ascii=False), "",
        "重复任务不是新的独立样本，不报告以 48 个独立任务为前提的显著性。排程第 1/3 轮相同，第 2 轮反序并交换起始组；环境逐项重置，服务每轮重启。延迟不含加载/服务启动时间。"]
    lines += ["", "检索依赖修复：用户要求先完成真实检索再按原协议实验。两组均使用 CPU BGE-small 和同一份原上游文档的新匹配索引；保留原 Ada 索引和 SimuHome 源码。该环境修复使 P0 不能作为同环境统计样本。检索 embedding tokens 不并入生成 token reward，完整检索次数、tokens、延迟在逐 episode CSV/JSON 单列。索引构建一次性开销：" + json.dumps(report["retrieval_dependency_repair"]["indexing_cost"], ensure_ascii=False), "",
        "|轮|组|检索调用|embedding tokens|检索秒|", "|---|---|---:|---:|---:|"]
    for r in per_round:
        for h in (B0, H1):
            c = r["harnesses"][h]["all"]
            lines.append(f"|{r['round']}|{'B0' if h == B0 else 'H1'}|{c['retrieval_calls']}|{c['retrieval_tokens']}|{c['retrieval_latency_seconds']:.3f}|")
    lines += ["", "修复前诊断链路额外开销（不计正式统计）：" + json.dumps(report["additional_pre_retrieval_chain_cost"], ensure_ascii=False),
        "", "未完成第一轮额外开销（13 个 episode，全部不计正式统计）：" + json.dumps(report["incomplete_round_overhead"]["totals"], ensure_ascii=False),
        "", "下载续传、索引构建和服务启动日志保留；服务启动/下载墙钟时间不算 episode 推理延迟。"]
    (destination / "P1-三轮配对报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stability = ["# P1：9B 稳定性与 Lightning 接入报告", "", "72 次固定请求重放与 8 个完整 episode 均独立核验通过。诊断不计入正式任务成功率。", "",
        "四任务族各第一个任务的首请求来自已核验 P0；完整消息、输出 schema、采样参数冻结。直接路径同样添加 return_token_ids=true。72 次实际后端请求哈希均与冻结请求一致。", "",
        "|任务|路径|请求数|不同全文|不同规范化动作|不同 prompt IDs|不同 reply IDs|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for g in diag["groups"]:
        stability.append(f"|{g['task_id']}|{g['path']}|{g['requests']}|{g['unique_texts']}|{g['unique_actions']}|{g['unique_prompt_ids']}|{g['unique_response_ids']}|")
    same = {m: sum(p[m] for p in diag["paired_requests"]) for m in ("same_text", "same_action", "same_prompt_ids", "same_response_ids")}
    stability += ["", "36 对相邻 direct/Gateway 请求一致次数：" + json.dumps(same, ensure_ascii=False), "",
        "|任务|direct 成功|Lightning 成功|direct reward|Lightning reward|首次动作分叉|首次反馈分叉|",
        "|---|---|---|---:|---:|---|---|"]
    for c in diag["chains"]:
        a, z = c["summaries"]["direct"], c["summaries"]["lightning"]
        stability.append(f"|{c['task_id']}|{a['success']}|{z['success']}|{a['reward']:.6f}|{z['reward']:.6f}|{c['first_divergence_turn']['action']}|{c['first_divergence_turn']['observation']}|")
    stability += ["", "每次重置后的目标相关初始状态与 P0 相同，首个后端请求完全一致；每条轨迹的真实反馈、最终目标检查、usage 和 reward 均复算，Gateway 的模型/环境/reward 事件及 token IDs 齐全。", "",
        "文字与动作变化分开统计；thought 改变不视为动作改变。完整环境中的时钟、聚合状态可能随时间变化，所以初始状态核对目标属性，其他状态差异保留在轨迹。顺序、热状态和重启共同变化，不能单独解释为已确定因果。请求相同不能保证全程逐字相同。", "",
        "开销：" + json.dumps(diag["cost"], ensure_ascii=False), "", "详细逐对、首次分叉、最终状态和 reward 见 diagnostics_verification.json、两次固定请求诊断目录及检索修复后的链路诊断目录。"]
    (destination / "P1-稳定性报告.md").write_text("\n".join(stability) + "\n", encoding="utf-8")
    print(json.dumps({"report_verified": True, "episodes": 96, "successes": {h: report["round_statistics"][h]["success"] for h in (B0, H1)}}, ensure_ascii=False))

if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), [Path(p).resolve() for p in sys.argv[3:]])
