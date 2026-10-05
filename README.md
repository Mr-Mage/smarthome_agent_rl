# SmartHome Agent Harness

复现官方 SimuHome 单轮 ReAct benchmark，接入 Agent Lightning；harness 基线为 Qwen3.5-9B，保留冻结权重结果。
4×H100各一个actor，每actor16个隔离模拟器槽，总并发64；Qwen3.6-35B-A3B judge常驻2×A800，CPU BGE检索。

当前：harness阶段N20已归档，保留G。N22原生thinking未过成本门槛；N23冻结35个训练任务、361个动作标签。N24单轮四卡LoRA训练与独立合并已核验，全流程7分17秒，尚无SR收益结论；下一节点为配对评测。后续见[SFT执行计划](docs/SFT执行计划.md)，剩余95个未使用任务仍封存。
N13正式seed42：B0/G/旧Full成功60/69/71（各192）；G非法执行96→16，SR差的Holm p=0.216，提升未证实。Verify v2、Context v2、N15 D/W及TimePlan均不纳入候选。旧final不再用于调参。

结果：[实验报表](docs/实验报表.md)；节点：[docs/nodes](docs/nodes/)；开发约定：[AGENTS.md](AGENTS.md)。
源码、配置、测试分别在`smarthome_agent_rl/`与`scripts/`、`configs/`、`tests/`；原始运行在忽略的`runs/`，审阅包在`outputs/`，临时文件在`work/`。

服务器：`ssh h100`。沿用`../activate-agent-lightning.sh`，激活后回项目目录；模型用`qwen36-vllm`，episode/模拟器用既有venv。不改依赖或Lightning用户补丁，运行前保持部署工作树干净。

复现入口：`python scripts/run_node_experiment.py --config configs/time-plan.json --run-dir runs/time-plan/<新目录>`；这是负结果复现，不是推荐增强策略，当前计划不再新增运行。
冻结配置：[Guard v3](configs/guard-v3.json)、[TimePlan](configs/time-plan.json)。每轮四卡配对，预声明种子42/43/44；先smoke再dev，普通任务失败留在分母，基础设施故障整阶段无效。
阶段核验：`python scripts/verify_benchmark.py <阶段目录>`；统计：`scripts/report_benchmark.py`；收据：`scripts/summarize_node_experiments.py`。
保留证据：`runs/harness-v2/primary-v2/`、`runs/guard-v3/n15-v1/`、`runs/time-plan/`。历史结果与论文参照见实验报表；judge三票来自同一模型，不是三个独立judge。
最新归档：`outputs/harness-post-n15/review-v2/`，28,310个有效阶段文件与109份审阅收据核验通过；失败成本有缺usage记录，详见[N20](docs/nodes/N20.md)。机器索引：[TimePlan及准入](docs/data/time-plan.json)。
