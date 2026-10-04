# SmartHome Agent Harness

原 SimuHome ReAct + Agent Lightning；Qwen3.5-9B actor，本地 Qwen3.6-35B-A3B judge。
四卡分工：GPU 0/1 两路独立 actor；GPU 2/3 共享 TP2 judge；CPU BGE 检索。

已完成 N0–N7：600 dev + 768 final。final 成功数：B0 66/192, G 76/192, GC 69/192, Full 73/192。
开发记录：`docs/nodes/`；约定：`AGENTS.md`；历史资料：`docs/archive/pre-mvp/`。
持续实验报表：[论文参照、9B 基线与节点结果](docs/实验报表.md)。
结果与证据：`runs/harness-mvp/primary-v1/`、`outputs/harness-mvp/`；原始失败保留，主实验不混入额外诊断。
结果解释见 `docs/nodes/N6.md`，独立验收见 `docs/nodes/N7.md`；judge 三票来自同一模型，保留原实时动力学。

服务器：`ssh h100`，激活已有 `../activate-agent-lightning.sh`；模型用 qwen36-vllm，episode/模拟器用各自 venv，环境与用户上游补丁保留。

运行入口：`scripts/harness_services.py`、`scripts/run_benchmark_suite.py`、`scripts/report_benchmark.py`。
独立复核：`python scripts/verify_benchmark.py <run-dir>`；冻结：`work/harness-mvp/final-freeze.json`。

并发配置：`configs/harness-concurrent-fixed.json`，32 个独立模拟器槽、CUDA Graph 与前缀缓存、工作流优先调度。已修复畸形参数触发 simulator 崩溃的边界检查，完整计时复跑见 [N8](docs/nodes/N8.md)；旧正式结果与失败轮保留。
前置测量：`scripts/run_concurrency_preflight.py`；完整验收：`python scripts/run_timed_benchmark.py --config configs/harness-concurrent-fixed.json --run-dir runs/concurrency-capacity/<新目录>`。

N8–N13已完成：全量并发验收43分50秒；新final seed42 B0/G/旧Full=60/69/71（各192），Holm p=0.216，未显著。Verify v2、Context v2未过dev门槛，最终保留G；旧结果、失败和开发诊断均保留。记录：[N10](docs/nodes/N10.md)、[N11](docs/nodes/N11.md)、[N12](docs/nodes/N12.md)、[N13](docs/nodes/N13.md)。

v2 清单与门槛：`configs/benchmark-v2/`、`configs/harness-v2-protocol.json`。运行 `python scripts/run_harness_v2.py --stage modules --run-dir runs/harness-v2/<新目录>`；完成模块对比后保留模型服务，检查结果并提交节点记录，再用相同目录 `--stage finalize` 进行集成选择、冻结与新final。各阶段共用模型，episodes并行；seed42为正式检验，43/44为可靠性诊断。现有证据：`runs/harness-v2/primary-v2/`，独立复核与收据打包：`python scripts/package_harness_v2.py`。原始episodes保留在`runs/`，审阅包在`outputs/harness-v2/`。
