# 当前状态

N0–N7 已验收。final：B0 66/192, G 76/192, GC 69/192, Full 73/192；统计与全部成本见 `runs/harness-mvp/primary-v1/final/report.json`。
每节点记录见 `docs/nodes/`；历史 P1 不与本轮官方 benchmark 混算。

N8：最小上游修复已提交并锁定；600 dev+768旧final完整核验，43分50秒，通过一小时门槛。
N9：历史dev审计完成，新final192冻结；原dev120保留。
N10/N11：验证与紧凑上下文v2已实现，67项测试通过；修正API枚举表示误报后整轮重跑，两模块均未过dev门槛，不纳入候选。
N12：完整模块1,800次+集成1,080次核验通过，Candidate=G，重复策略共享结果。
N13：新final1,728次核验通过，seed42 B0/G/Full=60/69/71（各192），两项Holm p=0.216，未显著。最终保留dev所选G，四卡已释放。
证据：`runs/harness-v2/primary-v2/`；旧开发诊断保留在`primary-v1/`；汇总与限制见`docs/实验报表.md`。

N14已完成：A800常驻judge+四个单卡9B；64/128总槽各144次完整核验，吞吐35.73/29.81次/分钟，无抢占，选择64槽（每actor16）。72项测试通过；全流程16分34秒，H100已清理、judge仍在线。产物`runs/concurrency-capacity/a800-four-actors-v1/`。下一主线为dead-front依赖与工作流能力检查消融，见`docs/后续实验方案.md`。
