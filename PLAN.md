# Harness 执行节点

| 节点 | 交付和验收 |
|---|---|
| N0 工程基点 | Git 历史统一、开发约定、旧文档归档 |
| N1 四卡运行 | GPU 0/1 各一个 9B actor；GPU 2/3 的 TP2 judge；真实推理与隔离验证 |
| N2 官方评测 | 12 类、120 dev/192 final 隔离清单；原 evaluator、Lightning、24 项 smoke |
| N3 Guard | schema/能力/已知前置条件校验；B0/G 配对和误拦截诊断 |
| N4 Verify/Repair | 局部效果验证、最多两次恢复；G/GV 配对 |
| N5 Context | 确定性状态账本、旧状态失效；GC/Full 与信息保留检查 |
| N6 冻结实验 | dev 五组 600 episodes；final 四组 768 episodes；额外诊断单列 |
| N7 验收归档 | 配对统计、证据核验、临时文件整理和运行入口 |

协议：seed 20261004，官方每类 10 dev/16 final，已接触案例不进 final。
actor 沿用 P1 参数；judge 为本地 Qwen3.6-35B-A3B，非 thinking、温度 0，三票使用同一服务的种子 42/43/44。
保持官方任务、初始状态、时间语义和评价规则；harness 只使用公开 API/设备语义，不访问隐藏目标。
final 前冻结代码、任务摘要和参数；失败保留，不择优补跑。
