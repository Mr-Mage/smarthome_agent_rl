# SmartHome Agent Harness

复现官方 SimuHome 单轮 ReAct benchmark，接入 Agent Lightning；固定 Qwen3.5-9B actor，不训练权重。
4×H100各一个actor，每actor16个隔离模拟器槽，总并发64；Qwen3.6-35B-A3B judge常驻2×A800，CPU BGE检索。

当前：N16完成864次有效配对，TimePlan未过门槛，保留G。N17离线审计发现时间拦截主要为缺step映射；N18未进入。N19剩余留出95个，且无合格增强，正式验收未准入；按[开发计划](docs/后续开发计划.md)停止新增模块，进入N20归档。
N13正式seed42：B0/G/旧Full成功60/69/71（各192）；G非法执行96→16，SR差的Holm p=0.216，提升未证实。Verify v2、Context v2、N15 D/W及TimePlan均不纳入候选。旧final不再用于调参。

结果：[实验报表](docs/实验报表.md)；节点：[docs/nodes](docs/nodes/)；开发约定：[AGENTS.md](AGENTS.md)。
源码、配置、测试分别在`smarthome_agent_rl/`与`scripts/`、`configs/`、`tests/`；原始运行在忽略的`runs/`，审阅包在`outputs/`，临时文件在`work/`。

服务器：`ssh h100`。沿用`../activate-agent-lightning.sh`，激活后回项目目录；模型用`qwen36-vllm`，episode/模拟器用既有venv。不改依赖或Lightning用户补丁，运行前保持部署工作树干净。

复现入口：`python scripts/run_node_experiment.py --config configs/time-plan.json --run-dir runs/time-plan/<新目录>`；这是负结果复现，不是推荐增强策略，当前计划不再新增运行。
冻结配置：[Guard v3](configs/guard-v3.json)、[TimePlan](configs/time-plan.json)。每轮四卡配对，预声明种子42/43/44；先smoke再dev，普通任务失败留在分母，基础设施故障整阶段无效。
阶段核验：`python scripts/verify_benchmark.py <阶段目录>`；统计：`scripts/report_benchmark.py`；收据：`scripts/summarize_node_experiments.py`。
保留证据：`runs/harness-v2/primary-v2/`、`runs/guard-v3/n15-v1/`、`runs/time-plan/`。历史结果与论文参照见实验报表；judge三票来自同一模型，不是三个独立judge。
