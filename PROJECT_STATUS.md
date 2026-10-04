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

N15已完成：smoke288+dev1,440次核验；G/GD/GW/GDW成功126/124/133/124（各360），非法执行57/16/49/36。三项候选均未过冻结组合门槛，保留G；全流程35分21秒，分离Agent与评测延迟。历史分支`feat/guard-v3`，原始证据`runs/guard-v3/n15-v1/`。

N16已完成：有效smoke144+dev720核验；G/TimePlan成功130/117（各360），QT4-1可行8/0，tokens+11.71%，增强未过门槛，保留G。v1–v4无效及启动/预检失败保留。
N17已完成离线诊断：300次T拦截中291次缺step映射，9次时间冲突；不能推断T/R的独立因果贡献。按停止条件不开展在线消融，N18未进入。
N19准入已完成：官方600个中保守暴露505，剩95；冻结检验规划需最多663个。没有合格增强且留出不足，不跑新final，转N20归档。分支`feat/time-plan-attribution`；机器汇总`docs/data/time-plan.json`，节点与报表保留负结果。
N20已完成：28,310个有效阶段文件、109份收据核验一致，有效阶段token汇总一致；全部失败尝试保留，缺usage的调用成本标下界。86项测试通过，上游与用户补丁不变，H100已释放、A800 judge常驻。计划执行结束：保留G，SR显著提升仍未证实。归档`outputs/harness-post-n15/review-v2/`。
