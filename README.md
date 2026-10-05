# SmartHome Agent Harness

复现官方 SimuHome 单轮 ReAct benchmark，接入 Agent Lightning；harness 基线为 Qwen3.5-9B，保留冻结权重结果。
4×H100各一个actor，每actor16个隔离模拟器槽，总并发64；Qwen3.6-35B-A3B judge常驻2×A800，CPU BGE检索。

当前：N21–N25已执行。35个训练任务、361个动作标签完成单轮四卡LoRA训练及独立合并；eval48三种子配对原9B/SFT成功51/24（各144），非法执行31/125，否决本次SFT配置，继续保留原模型+G。原生thinking也未过成本门槛。结果仅为历史暴露任务上的开发诊断，剩余95个未使用任务仍封存。
N13正式seed42：B0/G/旧Full成功60/69/71（各192）；G非法执行96→16，SR差的Holm p=0.216，提升未证实。Verify v2、Context v2、N15 D/W及TimePlan均不纳入候选。旧final不再用于调参。

结果：[实验报表](docs/实验报表.md)；节点：[docs/nodes](docs/nodes/)；开发约定：[AGENTS.md](AGENTS.md)。

N26观测ID提示诊断完成：原9B成功26/72→26/72，tokens+3.01%；SFT成功15/72→10/72，tokens+64.03%。完整目录下错误ID样本不足，主机制证据不足，不采用提示、不扩大该消融；发现与终止覆盖仍需研究，详见[N26](docs/nodes/N26.md)。审阅包`outputs/observation-binding/n26-review-v1/`。
N27只读复核全部31个原G seed42失败与361个训练目标：5个独立任务混淆通电与运行启动，准入N28最小公开语义提示配对；原G错误房间查询0。评测疑点保留官方失败，不改分母；无新增推理/训练，详见[N27](docs/nodes/N27.md)。
源码、配置、测试分别在`smarthome_agent_rl/`与`scripts/`、`configs/`、`tests/`；原始运行在忽略的`runs/`，审阅包在`outputs/`，临时文件在`work/`。

服务器：`ssh h100`。沿用`../activate-agent-lightning.sh`，激活后回项目目录；模型用`qwen36-vllm`，episode/模拟器用既有venv。不改依赖或Lightning用户补丁，运行前保持部署工作树干净。

历史harness复现：`python scripts/run_node_experiment.py --config configs/time-plan.json --run-dir runs/time-plan/<新目录>`；TimePlan为负结果，当前不推荐采用。
SFT流程与冻结配置见[SFT执行计划](docs/SFT执行计划.md)、[训练](configs/sft-pilot-training.json)、[配对评测](configs/sft-pilot-evaluation.json)。一次训练已完成；不覆盖现有产物、不追加epochs，不用eval选择训练checkpoint。原始权重、adapter、独立merged权重及全部失败证据保留。
冻结配置：[Guard v3](configs/guard-v3.json)、[TimePlan](configs/time-plan.json)。每轮四卡配对，预声明种子42/43/44；先smoke再dev，普通任务失败留在分母，基础设施故障整阶段无效。
阶段核验：`python scripts/verify_benchmark.py <阶段目录>`；统计：`scripts/report_benchmark.py`；收据：`scripts/summarize_node_experiments.py`。
保留证据：`runs/harness-v2/primary-v2/`、`runs/guard-v3/n15-v1/`、`runs/time-plan/`。历史结果与论文参照见实验报表；judge三票来自同一模型，不是三个独立judge。
最新SFT归档：`outputs/sft-pilot/final-review-v1/`，13,845个有效阶段文件与143份本机收据核验通过，保留失败成本、数据和权重身份；[N25](docs/nodes/N25.md)记录否决结论。历史归档：`outputs/harness-post-n15/review-v2/`，28,310个有效阶段文件与109份收据核验通过；失败成本有缺usage记录，详见[N20](docs/nodes/N20.md)。机器索引：[TimePlan及准入](docs/data/time-plan.json)。
