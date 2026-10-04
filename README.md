# SmartHome Agent Harness

原 SimuHome ReAct + Agent Lightning；Qwen3.5-9B actor，本地 Qwen3.6-35B-A3B judge。
四卡分工：GPU 0/1 两路独立 actor；GPU 2/3 共享 TP2 judge；CPU BGE 检索。

已完成 N0–N7：600 dev + 768 final。final 成功数：B0 66/192, G 76/192, GC 69/192, Full 73/192。
开发记录：`docs/nodes/`；约定：`AGENTS.md`；历史资料：`docs/archive/pre-mvp/`。
结果与证据：`runs/harness-mvp/primary-v1/`、`outputs/harness-mvp/`；原始失败保留，主实验不混入额外诊断。
结果解释见 `docs/nodes/N6.md`，独立验收见 `docs/nodes/N7.md`；judge 三票来自同一模型，保留原实时动力学。

服务器：`ssh h100`，激活已有 `../activate-agent-lightning.sh`；模型用 qwen36-vllm，episode/模拟器用各自 venv，环境与用户上游补丁保留。

运行入口：`scripts/harness_services.py`、`scripts/run_benchmark_suite.py`、`scripts/report_benchmark.py`。
独立复核：`python scripts/verify_benchmark.py <run-dir>`；冻结：`work/harness-mvp/final-freeze.json`。
