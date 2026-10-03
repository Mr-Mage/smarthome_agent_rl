# Harness 消融与稳定性报告

日期：2026-10-03（北京时间）。完成四项固定dev任务 × 五种harness配置的20次真实Lightning rollout，并对出现失败的单灯任务另做5次诊断复测。仍使用H100已有Qwen2.5-1.5B-Instruct，没有训练、换权重或修改SimuHome源码。

**当前可支持的结论：对象格式 + 参数schema是值得保留的简化基线；额外finish gate和动态恢复提示没有显示稳定的额外成功率收益。** 有一次no-gate失败，换顺序并重启服务复测后成功；原版在单灯任务上本身更省调用。不能据4项静态任务宣布某个组件普遍有效或无效。

## 主实验：固定4项任务

原始证据：`runs/structured-ablation-20261003-01`。输入与之前V1复测相同：两灯控制、先查询的两灯控制、反向两灯控制、单灯且保持另一灯状态。

| 配置 | 状态目标满足且合法finish | 总模型调用 | 非法动作/输出 | 总tokens |
|---|---:|---:|---:|---:|
| 原版 ReAct | 1/4 | 24 | 18 | 135,407 |
| 仅对象格式 + 参数 schema | 4/4 | 18 | 2 | 84,201 |
| 完整 V1 去掉 finish gate | 3/4 | 32 | 19 | 183,860 |
| 完整 V1 去掉动态恢复提示 | 4/4 | 16 | 1 | 75,995 |
| 完整 V1 | 4/4 | 17 | 2 | 80,746 |

所有指标包含失败episode，不只统计成功轨迹。单灯任务的部分目标初始已满足，reward进度仍按初始/最终差值计算，并验证未请求修改的另一灯保持原状态。

| 配置标识 | typed对象/schema | 结束gate | 动态错误提示 | V1参数/验证指导语 |
|---|---|---|---|---|
| upstream-react | 否（原版JSON字符串） | 原版 | 原版观察 | 原版 |
| structured-schema | 是 | 否 | 否 | 否，仅格式契约 |
| structured-no-gate | 是 | 否 | 是 | 是，去掉gate可用性说明 |
| structured-no-recovery | 是 | 是 | 否 | 是 |
| structured-v1 | 是 | 是 | 是 | 是 |

完整V1输出契约与前一阶段保持相同。代码只是增加可关闭的开关，不给变体写另一套agent循环。所有变体继续使用原版ReActAgent和20个工具函数。finish gate仅在本任务实际成功查询设备/房间之后开放finish，不限制全部写操作，也不读取verifier目标。

仅schema并不等于“只有解码器这一项变化”：对象格式需要额外格式契约及few-shot/历史动作的传输适配。该组比较不能将格式表示和grammar约束的作用分开。no-gate还去掉了gate可用性的一句说明；一般验证/错误处理指导语仍保留。no-recovery只去掉每次错误后的额外user提示，原版错误观察和V1通用处理指导语仍保留。

## 失败轨迹与复测

主实验no-gate在单灯任务上失败：先write_attribute(attribute_id=On)得到错误；成功查询get_device_structure后，仍多次重复无效属性写入。20步中19次invalid，未合法finish。schema保证参数形状，却不能保证属性名/写权限的语义合法性。

主实验完整V1也不是没有无效行为：单灯任务两次尝试add_device创建已存在的light，随后查询房间设备、调用On并finish，共5步、2次invalid。无动态恢复提示版本则先错误写属性、查询结构、调用On并finish，共4步、1次invalid。

针对这个失败单灯任务，另跑 `runs/structured-ablation-20261003-02`；用独立服务启动，并将五种配置按主列表的相反顺序执行。输入、模型、seed、温度、预算不变。

| 单灯诊断配置 | 成功 | 模型调用 | 非法动作 | tokens |
|---|---:|---:|---:|---:|
| 完整 V1 | 1/1 | 4 | 1 | 19,380 |
| 完整 V1 去掉动态恢复提示 | 1/1 | 4 | 1 | 19,369 |
| 完整 V1 去掉 finish gate | 1/1 | 4 | 1 | 19,341 |
| 仅对象格式 + 参数 schema | 1/1 | 4 | 0 | 18,387 |
| 原版 ReAct | 1/1 | 2 | 0 | 8,785 |

此前no-gate失败在这轮未复现。复测同时改变了顺序和服务热状态，不能只归因于顺序。诊断是失败案例重测，不是新独立任务；主实验的3/4结果保留，不把复测成功替换失败，也不将这些选择性重测混入主成功率。

## 实际观察到的可复现性问题

主实验单灯任务中，完整V1与no-recovery的**首个请求JSON完全相同**，包含messages、seed=42、temperature=0、模型与response_format。然而首个输出不同：完整V1选择add_device，no-recovery选择write_attribute。此时尚无工具错误，恢复提示还未被添加，因此不能把这个初始动作差异解释为恢复提示效果。

这个对照记录在 `stability_observation.json`，可直接与两份model_calls.json核对。固定seed和temperature在当前服务配置下没有获得逐字输出一致性；根因尚未查证，不断言具体来自某个后端机制。

因此，75,995 vs 80,746 tokens等小样本成本差异只能描述这次运行，不能当作“关闭恢复提示必然省token”的结论。下一轮正式实验应先审计后端重复请求一致性，并增加重复试验和独立任务覆盖。

## 验证、代码与复现

两轮25个episode均通过独立artifact检查：配对任务和初始目标状态一致，模型/seed/调用上限一致，原始模型动作、适配结果与原版assistant事件相同，实际schema允许的工具与gate开关一致。每步状态与最终设备谓词由真实模拟器验证，token/invalid/进度/成本/终止奖励重算与Lightning reward一致，prompt/response token IDs非空。成功记录均有原版合法finish且执行正常完成。

7项测试通过，包含原版parser兼容、完整注册表、历史缺参拒绝、类型/未知参数拒绝、few-shot不能开启gate、恢复不加额外模型调用，以及开关只改变声明的规则。原版源码git status为空；结束后4张H100显存均为0MiB。

```bash
cd /HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl
bash scripts/run-suite-h100.sh --suite configs/dev-suite-v2.json --run-dir runs/my-ablation \
  --harnesses upstream-react structured-schema structured-no-gate structured-no-recovery structured-v1
python scripts/verify_harness_suite.py runs/my-ablation

bash scripts/run-suite-h100.sh --suite configs/dev-single-diagnostic.json --run-dir runs/my-single-diagnostic \
  --harnesses structured-v1 structured-no-recovery structured-no-gate structured-schema upstream-react
python scripts/verify_harness_suite.py runs/my-single-diagnostic
```

run-dir必须不存在。`configs/ablation-v1.json`记录配置表/假设，执行参数以保存的commands.json/harnesses.json为准。`structured.py`提供开关；`run_harness_suite.py`支持多组harness并旋转顺序；验证器按代码快照核对实际开关。

采集时metrics_version=2的原始summary保持不变。这次最终源码把success进一步明确为“状态目标满足 + 合法finish + 正常执行”，额外goal_success独立表示环境目标是否满足，metrics_version=3；只是完善统计定义，不改变模型决策，也未重跑或改写历史结果。现有25次结果在更严格成功定义下也全部通过核验。

## 适用范围

这仍是4项静态开关灯dev任务，不是官方benchmark，3项是两灯变体。没有证明温湿度/定时目标、检索、复杂动作参数、安全或多模态能力。暂无SFT、GRPO或任何训练器消费这些轨迹。

项目保留原版、完整V1和简化schema变体，不根据这几个样本自动宣布替换所有配置。下一步优先解决评测可复现性，再扩展独立dev任务和重复试验；成功轨迹与失败负reward均已完整保存。
