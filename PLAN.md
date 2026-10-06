# Harness 节点与最终交付

当前默认Qwen3.5-9B + G。只做官方单轮harness复现、诊断与消融；不追加SFT/RL、语音、多轮、新benchmark或生产接入。用户新增授权N37官方600任务全量验收，包括此前封存95任务；不重新选优。

| 节点 | 当前状态 | 结论与入口 |
|---|---|---|
| N0–N7 基础复现 | 完成 | 官方任务、Lightning接入、Guard/Verify/Context及配对实验；历史证据保留 |
| N8–N13 并发与正式验收 | 完成 | G正式非法执行96→16；SR显著收益未证实；[N13](docs/nodes/N13.md) |
| N14–N20 约束消融与归档 | 完成；条件节点N18未进入 | 四actor64槽；D/W、TimePlan未准入，N19不启封新final；[N20](docs/nodes/N20.md) |
| N21–N25 历史训练试验 | 完成，配置否决 | 一次LoRA已验收，不追加训练；[N25](docs/nodes/N25.md) |
| N26–N29 绑定与启动诊断 | 完成；N29未准入 | 不采用ID/启动提示，不扩大验证；[N28](docs/nodes/N28.md) |
| N30 架构核对 | 完成 | [项目架构](docs/项目架构.md)区分已使用、可选未采用与生产缺口 |
| N31 动作生命周期 | 完成，G已接入 | 状态/回执/未知结果分开，192条行为一致；[N31](docs/nodes/N31.md) |
| N32 独立恢复预算 | 工程完成，GR默认关闭 | 离线审计不代表在线收益；[N32](docs/nodes/N32.md) |
| N33 显式目标关联 | 完成，GTS未准入 | SR/成本/安全及语义门槛失败；[N33](docs/nodes/N33.md) |
| N34 证据上下文 | 完成，GEC未准入 | 输入节省0%，不开展在线配对；[N34](docs/nodes/N34.md) |
| N35 工程审阅与检查 | 完成 | 96轨迹/705动作核验，150测试149通过/1跳过；[N35](docs/nodes/N35.md) |
| N36 最终交付 | 完成 | 24次链路/453阶段文件核验，7包复核、GPU释放；[N36](docs/nodes/N36.md) |
| N37 全量benchmark | 完成 | 官方600任务、B0/G成功197/600、231/600；1,200次核验；全量快照结果，不作独立留出收益；[N37](docs/nodes/N37.md) |
| N38 Executable Contract | 离线完成 | 通用设备/函数/参数/断言契约，结构化拒绝；默认未启用blocking；[N38](docs/nodes/N38.md) |
| N39 Semantic Verification | 离线完成 | public-only四维决策、置信度门控、Self-Reflection解析器；不生成新benchmark、不做在线对照 |
| N40 Process Reward / RL接口 | 接口完成 | RL-A/B/C、冻结verifier reward、交替调度；按本轮范围不启动训练；[N40](docs/nodes/N40.md) |

| N41 基线冻结 | 完成 | 永久 tag `baseline/1007-9b-g`、八份源码 SHA 核验；[N41](docs/nodes/N41.md) |
| N42 执行分层 | 工程完成 | Tool Trace / Mutation / Workflow 分开；[N42](docs/nodes/N42.md) |
| N43 Task 持久化 | 工程完成 | SQLite、版本修订、所有权和整个目标验证；[N43](docs/nodes/N43.md) |
| N44 声明式模拟器契约 | 工程完成 | 公开参数/范围/部分前置条件与读回规则；[N44](docs/nodes/N44.md) |
| N45 Scheduler | 工程完成 | 原生监督/定时动作/wake-up、原子认领与未知结果处理；[N45](docs/nodes/N45.md) |
| N46 真实执行链接入 | 工程完成 | 服务器44专项通过、全量211项210通过/1跳过；真实设备及可选ReAct接入；[N46](docs/nodes/N46.md) |
| N47 契约/读回消融 | 已冻结，待执行 | G/RC/RCV/B0；校准12、开发24×三种子；四卡64槽，未知前置条件覆盖不足则停止；`configs/runtime-ablation.json` |

N37全量验收已完成。N41–N46为工程验证，暂无新benchmark效果；下一步分别冻结契约与读回验证消融。按最新确认不追加训练、不生成新benchmark、不展开多轮或新的压力任务；持久模块保留工程证据。普通失败及全部成本保留，接口测试不作效果收益。

冻结协议保留在[后续实验方案](docs/后续实验方案.md)、[后续开发计划](docs/后续开发计划.md)、[SFT执行计划](docs/SFT执行计划.md)及各配置中；其“待执行”描述是历史冻结文本，当前状态以本表、节点记录和[实验报表](docs/实验报表.md)为准。
