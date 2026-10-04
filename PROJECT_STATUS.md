# 当前状态

N0–N7 已验收。final：B0 66/192, G 76/192, GC 69/192, Full 73/192；统计与全部成本见 `runs/harness-mvp/primary-v1/final/report.json`。
每节点记录见 `docs/nodes/`；历史 P1 不与本轮官方 benchmark 混算。

N8：最小上游修复已提交并锁定；复跑600 dev通过核验，旧final768性能验收中。
N9：历史dev审计完成，新final192冻结；原dev120保留。
N10/N11：验证与紧凑上下文v2已实现，65项测试通过，等待模块实验。
N12/N13：模块→集成→新final入口已实现；32槽并发、三个actor种子、dev门槛选择、重复策略去重。v2已排队，N8通过并释放四卡后启动。
