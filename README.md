# SmartHome Agent Harness

基于官方 SimuHome 单轮 ReAct benchmark 与 Agent Lightning 的可审计智能家居 Agent harness。项目实现工具契约校验、动作生命周期记录、四卡隔离调度与配对评测，保留完整失败及成本证据。

**当前默认：Qwen3.5-9B + Guard（G）。已完成至 N36，新增 N37 全量验收待运行。** 项目概览、设计取舍与案例见[项目总结书](docs/项目总结书.md)。

## 结果与边界

N13正式实验：冻结的192个官方任务、十二类各16个，预声明seed42。

| 指标 | B0：原9B ReAct | G：原9B + Guard |
|---|---:|---:|
| 成功任务 / SR | 60/192，31.25% | 69/192，35.94% |
| 非法执行总次数 | 96 | 16 |
| Actor tokens / 任务 | 41,015.4 | 43,730.1 |
| Harness额外查询总次数 | 0 | 295 |

非法执行减少 **83.33%**，actor tokens增加 **6.62%**。SR差+4.69个百分点，95% CI[-1.04,+10.94]，Holm p=0.216，**未证明显著提升成功率**。非法执行只计实际工具返回的非法动作错误，不含schema拒绝；tokens包含成功及失败任务，不含judge。论文采用600任务及不同模型/judge，不能直接横向宣称超越论文。

N31动作生命周期已接入G。N32恢复预算、N33目标关联、N34证据上下文为可选实现，未替换G；Context/Verify、TimePlan及一次历史LoRA SFT也未准入。N35/N36交付验收完成。N37按用户授权增加当前官方600任务的B0/G配对验收，包括此前封存95任务；已有192任务结果仍单独保留，全量不作独立留出收益。状态见[计划](PLAN.md)，N36交付索引见[delivery.json](docs/data/delivery.json)。

## 核心工程设计

模型提出计划 → schema约束调用 → Guard校验 → SimuHome执行 → 公开观测反馈；官方隐藏目标与评测结果不进入模型上下文。

| 责任 | 设计与边界 |
|---|---|
| 意图与规划 | 默认沿用上游ReAct；显式TaskSpec已实现但未采用 |
| 调用契约 | 校验参数、能力、只读属性和部分公开前置条件；未知覆盖保留`uncovered` |
| 状态与回执 | N31记录动作状态、完整参数、尝试及证据；接受、注册、完成、未知分别呈现 |
| 记忆与上下文 | 公开观测账本与静态知识检索分开；Context/证据引用增强未采用 |
| 异常与重试 | 区分领域失败、未知执行结果和基础设施故障；GR默认关闭，网络重试沿用上游 |
| 隔离与评测 | 4 actor × 16独立模拟器槽，配对固定actor；独立A800 judge，按冻结规则验收效果与成本 |

持久恢复、可靠幂等/事务、多用户权限和生产SLO未验收。动作完成不自动等于用户目标成功。

## 工程入口

- [项目总结书](docs/项目总结书.md)：项目目标、核心设计、正式结果、失败案例和交付边界。
- [项目架构](docs/项目架构.md)：意图、记忆、上下文、状态、恢复及实现边界。
- [实验报表](docs/实验报表.md)：论文参照、9B基线、正式结果和全部节点负结果。
- [节点记录](docs/nodes/)：[N31](docs/nodes/N31.md)、[N32](docs/nodes/N32.md)、[N33](docs/nodes/N33.md)、[N34](docs/nodes/N34.md)、[N35](docs/nodes/N35.md)、[N36](docs/nodes/N36.md)。
- [依赖身份](dependencies.lock.json)、[开发约定](AGENTS.md)：沿用环境和用户补丁。

源码在`smarthome_agent_rl/`，运行脚本在`scripts/`，配置在`configs/`，测试在`tests/`。原始运行放`runs/`，审阅包放`outputs/`，临时工具放`work/`，均不入Git；不删除唯一证据。克隆仓库不包含这些运行产物，已有证据包位置由[交付索引](docs/data/delivery.json)记录。

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

以下入口用于已有服务器环境。`ssh h100`后进入`/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl`。模型环境`qwen36-vllm`；driver使用现有激活脚本，episode/模拟器使用既有venv，不改依赖。

```bash
cd /HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl
source ../activate-agent-lightning.sh
cd /HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl
python scripts/run_node_experiment.py --config configs/harness-release.json --run-dir runs/delivery/<新目录>
python scripts/verify_benchmark.py runs/delivery/<新目录>/smoke
```

[交付配置](configs/harness-release.json)：4×H100各一个9B actor，每actor16个独立模拟器槽，共64槽；Qwen3.6-35B-A3B judge常驻2×A800，CPU BGE检索。仅已暴露calibration12的B0/G×seed42共24次链路验收，不重新选优或替代正式实验。judge三票来自同一模型，不是三个独立judge。

旧实验方案、SFT数据/权重和失败证据均保留；历史入口见[后续开发计划](docs/后续开发计划.md)、[SFT执行计划](docs/SFT执行计划.md)。正式结论以[N13](docs/nodes/N13.md)与实验报表为准。
