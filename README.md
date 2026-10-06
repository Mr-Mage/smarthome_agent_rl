# SmartHome Agent Harness

官方 SimuHome 单轮 ReAct benchmark 的可审计 harness，接入 Agent Lightning。当前采用冻结的 **Qwen3.5-9B + Guard（G）**：模型规划，执行器检查公开契约，模拟器维护权威状态，原官方 evaluator 评测。

## 结果与边界

N13正式seed42、192任务：B0/G成功60/69（31.25%/35.94%），非法执行96→16（减少83.33%）；SR差+4.69pp，95% CI[-1.04,+10.94]pp，Holm p=0.216，未证明显著提升。论文采用600任务及不同模型/judge，不能直接横向宣称超越论文。

N31动作生命周期已接入G。N32恢复预算、N33目标关联、N34证据上下文为可选实现，未替换G；Context/Verify、TimePlan及一次历史LoRA SFT也未准入。N35/N36交付验收完成，当前计划收尾；不追加训练，95个未使用任务封存。状态见[计划](PLAN.md)，机器索引见[delivery.json](docs/data/delivery.json)。

可展示：强类型调用、公开契约、状态与回执边界、错误归类、四卡隔离、配对实验及失败成本。持久恢复、事务/幂等、多用户权限和生产SLO未验收。

## 工程入口

- [项目架构](docs/项目架构.md)：意图、记忆、上下文、状态、恢复及实现边界。
- [实验报表](docs/实验报表.md)：论文参照、9B基线、正式结果和全部节点负结果。
- [节点记录](docs/nodes/)：[N31](docs/nodes/N31.md)、[N32](docs/nodes/N32.md)、[N33](docs/nodes/N33.md)、[N34](docs/nodes/N34.md)、[N35](docs/nodes/N35.md)、[N36](docs/nodes/N36.md)。
- [依赖身份](dependencies.lock.json)、[开发约定](AGENTS.md)：沿用环境和用户补丁。

源码在`smarthome_agent_rl/`，运行脚本在`scripts/`，配置在`configs/`，测试在`tests/`。原始运行放`runs/`，审阅包放`outputs/`，临时工具放`work/`，均不入Git；不删除唯一证据。

## 检查与轨迹演示

本机或CI仅需Python标准库，不安装项目依赖、不请求模型：

```bash
python scripts/check_project.py
python scripts/review_episode.py <episode>/G/lightning --manifest <冻结SHA索引> --output outputs/delivery/case
```

轨迹审阅输出`review.json`和`review.md`：动作状态、完整参数、尝试、父提案、观测引用和原官方结果。未给冻结索引时仅计算自身SHA，不能证明来源真实；动作`completed`或工作流`registered`均不等于用户目标成功。已核对96条轨迹/705个动作，全量150测试149通过/1环境跳过；三个历史案例与十二份当前G审阅见`outputs/delivery/n36-review-v1/extracted/runs/delivery/`及[N36](docs/nodes/N36.md)。CI配置已加入仓库，GitHub端执行状态未核验。

服务器全量检查沿用现有venv：

```bash
PYTHONPATH=.:deps/SimuHome PYTHONNOUSERSITE=1 .venv-baseline/bin/python scripts/check_project.py --full
```

## 四卡运行

`ssh h100`后进入`/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl`。模型环境`qwen36-vllm`；driver使用现有激活脚本，episode/模拟器使用既有venv，不改依赖。

```bash
source ../activate-agent-lightning.sh
cd /HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl
python scripts/run_node_experiment.py --config configs/harness-release.json --run-dir runs/delivery/<新目录>
python scripts/verify_benchmark.py runs/delivery/<新目录>/smoke
```

[交付配置](configs/harness-release.json)：4×H100各一个9B actor，每actor16个独立模拟器槽，共64槽；Qwen3.6-35B-A3B judge常驻2×A800，CPU BGE检索。仅已暴露calibration12的B0/G×seed42共24次链路验收，不重新选优或替代正式实验。judge三票来自同一模型，不是三个独立judge。

旧实验方案、SFT数据/权重和失败证据均保留；历史入口见[后续开发计划](docs/后续开发计划.md)、[SFT执行计划](docs/SFT执行计划.md)。正式结论以[N13](docs/nodes/N13.md)与实验报表为准。
