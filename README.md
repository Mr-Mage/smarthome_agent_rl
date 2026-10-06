# SmartHome Agent Harness

基于官方 SimuHome 单轮 ReAct benchmark 与 Agent Lightning 的可审计智能家居 Agent harness。项目实现工具契约校验、动作生命周期记录、四卡隔离调度与配对评测，保留完整失败及成本证据。

**当前默认：Qwen3.5-9B + Guard（G）。官方600任务全量结果已完成；新执行 runtime 尚无在线收益结论。** 项目概览、设计取舍与案例见[项目总结书](docs/项目总结书.md)。

N41–N46已实现执行分层、SQLite任务与调度器；N61统一入口完成原生验收。N63监督门槛失败，N64修复错误回执与重复写盘；独立GTME在[N66](docs/nodes/N66.md)的720回合消融中通过工程验收，191个定时Job中64个有公开读回证据。G/GTME成功率34.17%/35.00%，区间跨0且成本增加，未证明稳定收益。目标未核实的Task保留WAITING。默认G不变；本轮不追加训练、多轮或新benchmark。

N47已完成48次四组校准：G/新契约/新契约加读回/原ReAct成功5/5/5/6（各12）。新契约覆盖不足，按预声明门槛停止扩大实验，保留G；未证实读回收益。当前进度见[状态索引](docs/data/1007-plan-status.json)，失败及全部成本见[N47](docs/nodes/N47.md)。

[N67](docs/nodes/N67.md)独立目标解析诊断未准入：结构有效207/360，本代理开发者语义审阅6/36通过；漏控制条件及跨句引用限制仍未解决，停止原生目标集成。默认G不变。

[N70](docs/nodes/N70.md)仅换35B解析器后，结构有效317/360（88.06%）、模型全YES 119/360、开发者语义审阅11/36通过，仍未达到冻结门槛。控制语义与目标条件遗漏未解决；开发者审阅和同模型审阅均非独立语义真值。

[N71](docs/nodes/N71.md)补齐可选语义验证的公开回执来源与缺失标记，原生工程验收通过；[N72](docs/nodes/N72.md)完成38个真实动作的四组配对审阅。9B公开上下文格式全部有效，却漏6个开发者明确冲突；35B公开上下文只拒绝1个并有4次截断。未准入阻断或目标完成判断。

[N68–N69](docs/nodes/N69.md)已接通公开资源声明与Agent Job摘要；72回合确认34声明、19次实际HTTP上下文送达。成功13/36→14/36，但全部结果翻转均无候选上下文，未证明收益；本轮无真实冲突，冲突检测仅有原生工程测试证据。

HomeBench公开全量17,366任务已完成：N54原提示B0 EM27.34%、F+Guard 54.92%；N57独立W格式+Guard三种子EM为54.93%/55.80%/54.96%。主要收益来自输出表示与非法调用过滤；这是静态指令匹配，不能等同设备执行成功率。含94条暴露校准任务，三种子为描述性复核；[全量结果与成本](docs/nodes/N57.md)。

[对称只读诊断](docs/nodes/N60.md)：同样加W格式与Guard后，原提示58.02%、契约提示54.96%。契约没有正收益且增加tokens；不能把不对称的原B0→FG差值归因于契约或规划改进。

## 结果与边界

N37全量验收：当前官方快照600任务、十二类各50个，B0/G×seed42共1,200次。

| 指标 | B0：原9B ReAct | G：原9B + Guard |
|---|---:|---:|
| 成功 / SR | 197/600，32.83% | 231/600，38.50% |
| 非法执行总次数 | 325 | 70 |
| Actor tokens / 任务 | 43,469.2 | 45,853.9 |
| 额外查询总次数 | 0 | 903 |
| 未完成（计失败） | 5 | 8 |

G的SR增加 **5.67个百分点**，非法执行减少 **78.46%**，actor tokens/任务增加 **5.49%**。本快照配对95% CI[+2.17,+9.00]、p=0.00245；包含历史开发暴露任务，不能当作600任务独立留出收益。全流程约22分钟，12,391阶段文件核验通过；[完整报表](docs/实验报表.md)、[全量证据索引](docs/data/full-benchmark.json)。

下面保留N13独立留出主检验，与全量成绩分开解释。

N13正式实验：冻结的192个官方任务、十二类各16个，预声明seed42。

| 指标 | B0：原9B ReAct | G：原9B + Guard |
|---|---:|---:|
| 成功任务 / SR | 60/192，31.25% | 69/192，35.94% |
| 非法执行总次数 | 96 | 16 |
| Actor tokens / 任务 | 41,015.4 | 43,730.1 |
| Harness额外查询总次数 | 0 | 295 |

非法执行减少 **83.33%**，actor tokens增加 **6.62%**。SR差+4.69个百分点，95% CI[-1.04,+10.94]，Holm p=0.216，**未证明显著提升成功率**。非法执行只计实际工具返回的非法动作错误，不含schema拒绝；tokens包含成功及失败任务，不含judge。论文采用600任务及不同模型/judge，不能直接横向宣称超越论文。

N31动作生命周期已接入G。N32恢复预算、N33目标关联、N34证据上下文为可选实现，未替换G；Context/Verify、TimePlan及一次历史LoRA SFT也未准入。N35/N36交付验收完成。N37已完成用户授权的官方600任务B0/G配对验收，包括此前封存95任务；已有192任务结果仍单独保留，全量不作独立留出收益。状态见[计划](PLAN.md)，N36交付索引见[delivery.json](docs/data/delivery.json)。

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
- [1006改进计划状态](docs/data/1006-plan-status.json)：N38 Contract、N39 Semantic Verification/Self-Reflection、N40 Process Reward 接口；本轮只做 Harness 与消融，保留 RL 接口但不训练、不生成新 benchmark。
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

N37完整600任务复现沿用同一环境，在干净Git工作区运行：

```bash
python scripts/run_node_experiment.py --config configs/harness-full-benchmark.json --run-dir runs/full-benchmark/<新目录>
python scripts/verify_full_benchmark.py --run runs/full-benchmark/<新目录>
```

已有N37结果及原始证据保留，无需为查看报告重跑。[全量配置](configs/harness-full-benchmark.json)保持B0/G、seed42及所有失败，不追加调参。

## HomeBench公开单轮对照

[公开资源适配](docs/nodes/N50.md)固定17,366任务/100家庭及上游SHA。公开字段、zero-shot提示内容与Counter指标保持一致；Qwen3.5模板要求user消息，因此各臂统一追加空user消息。这是指令生成EM/F1，无设备执行或judge。B0适配后原提示、B1加公开契约、B2复用B1输出加Guard，成本按实际请求计一次。

在当前隔离工作区、既有venv运行（四张H100需空闲）：

```bash
bench_source=work/public-benchmarks/homebench/6d650caa19ba061e8790fc0f78f506169d029e4e
.venv-baseline/bin/python scripts/fetch_public_benchmark.py --benchmark HomeBench --output "$bench_source" --proxy http://127.0.0.1:7897
.venv-baseline/bin/python scripts/run_public_benchmark.py --source "$bench_source" --output runs/homebench/calibration --launch-actors
.venv-baseline/bin/python scripts/verify_public_benchmark.py --source "$bench_source" --run-dir runs/homebench/calibration
```

校准接入门槛通过后，同配置增加`--stage full --calibration runs/homebench/calibration`并使用新输出目录。冻结配置见[homebench-ablation.json](configs/homebench-ablation.json)；原始预测、失败、usage、GPU采样和共享请求均保留，当前实验状态见[实验报表](docs/实验报表.md)。

旧实验方案、SFT数据/权重和失败证据均保留；历史入口见[后续开发计划](docs/后续开发计划.md)、[SFT执行计划](docs/SFT执行计划.md)。正式结论以[N13](docs/nodes/N13.md)与实验报表为准。
