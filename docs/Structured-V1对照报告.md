# Structured Harness V1：H100 配对对照报告

核对日期：2026-10-03（北京时间）。**基于原版 SimuHome ReAct 循环的外部 structured-v1 harness 已完成真实设备控制**。去重后的4项固定开发任务，原版成功1项，改进版成功4项。两者均由用户已有 Agent Lightning 执行，使用同一个本地 Qwen2.5-1.5B-Instruct 模型；没有训练或更换权重。

## 实测结果

主结果：`runs/structured-v1-20261003-02`，输入 `configs/dev-suite-v2.json`，共8次配对online rollout。

| 指标（4项任务总计） | 原版 upstream-react | structured-v1 |
|---|---:|---:|
| 真实状态验证成功 | 1/4 | 4/4 |
| 模型调用 | 24 | 16 |
| 非法动作/输出 | 18 | 1 |
| prompt + completion tokens | 135,407 | 76,023 |
| reward总和 | -1.075407 | 4.363977 |

逐任务数据，后两列为“成功情况 / 模型调用 / 非法动作”：

| 独立dev任务 | 原版 | 改进版 |
|---|---|---|
| dev-lights-20261003 | 失败 / 1 / 0 | 成功 / 4 / 0 |
| dev-lights-20261003-query-first | 失败 / 20 / 18 | 成功 / 4 / 0 |
| dev-lights-20261004-reverse | 失败 / 1 / 0 | 成功 / 4 / 0 |
| dev-lights-20261005-single | 成功 / 2 / 0 | 成功 / 4 / 1 |

四项分别是两灯开关、明确要求先查询的两灯开关、反向两灯开关、单灯开关且保持另一灯状态。普通任务没有额外给模型规定操作序列。所有目标、初始配置在运行前固定，并在每对实验中重新reset。单灯任务的未操作灯状态也由goal验证，避免误操作被忽略。

首轮 `structured-v1-20261003-01` 也保存了8次rollout；其中两个seed生成了完全相同的两灯任务。去重复测将第三项改为反向控制，未改harness决策代码。首轮仅作开发记录，不把重复样本作为额外任务覆盖或扩充主结果样本量。

## 实现了什么

`smarthome_agent_rl/structured.py` 实现原版LLMProvider接口，组合原版OpenAIChatProvider。`create_agent_strategy("react", llm=decision_provider)` 仍创建原版ReActAgent；原版20个工具函数、调度、观察反馈及显式finish解析都继续使用。

V1是以下几项变化的组合：

1. 从原版完整TOOL_REGISTRY的docstring读取参数名、类型、必填字段；构造按tool区分的JSON schema。把action_input的嵌套JSON字符串改成真正的arguments对象，必填args不能遗漏；禁用工具顶层未知参数。
2. 模型只生成一个 `thought + call(tool, arguments)`。调用原版前校验类型/必填参数，再作无语义修改的传输适配，转成原版需要的action/action_input字符串。工具名称或操作参数完全来自模型，未静默补参数、代模型选动作。
3. 给原版system prompt追加输出契约和错误处理提示，并将few-shot/历史assistant动作转换成相同对象格式，保持示例的工具与意图。没有去掉few-shot或缩减工具注册表。
4. 在实际任务出现成功的设备/房间查询前，从生成schema中暂时排除finish。演示中的查询不会打开finish；没有使用verifier的goal来决定模型是否可结束。这一规则只限制结束，不是所有写操作的安全检查。
5. 上一次工具返回错误时，在下一次正常模型调用中附加纠正提示。没有额外恢复模型请求或额外预算。

HTTP adapter/reset、独立状态verifier和reward仍在外层。模型不获得额外特权状态/goal；如果它选择原版公开状态查询工具，则正常得到该工具的观察。SimuHome源码没有修改，也未复制到交付包。

## 一条真实成功轨迹

普通两灯任务的改进版轨迹是：

```text
get_room_devices(bedroom)
→ execute_command(bedroom_light_1, endpoint=1, OnOff, On, args={})
→ execute_command(bedroom_light_2, endpoint=1, OnOff, Off, args={})
→ finish
```

两灯状态分别变为true/false，仿真器真实状态检查通过。反向任务则实际执行Off/On。

单灯任务保留了唯一一次非法动作：模型先尝试write_attribute，使用无效的attribute_id=On；工具返回错误。模型随后查询get_device_structure，改用execute_command(On,args={})，最后finish。它没有通过把失败动作改写为成功来清零invalid。负分已包含在该episode reward中，另一盏灯保持原状态。

## 控制变量与验证

两阶段都经 Lightning Gateway + Local Controller，is_train=false；温度0、model seed42、max_steps20、context32768、同模型服务、同工具集和同reward。每个task只改变harness配置。按任务交替运行顺序，降低固定先后顺序造成的warm-up/cache偏差。

6项测试已在H100独立baseline环境通过：完整工具覆盖、历史遗漏args拒绝、参数类型与未知字段拒绝、原版parser往返兼容、few-shot不能开启finish、错误恢复没有额外模型调用。

独立 `scripts/verify_harness_suite.py` 对两轮16次真实rollout通过检查：

- 配对task与初始目标状态相同，模型/seed/调用上限相同。
- raw模型结果、传输适配结果与原版ReAct实际assistant事件一致，没有替换动作。
- Gateway请求只增加return_token_ids，模型prompt/response token IDs均非空。
- 调用/步数/非法动作/token统计一致，最终状态谓词与success一致。
- reward由初始/最终状态进度、终止成功、非法动作和全部token成本重算，与本地sum及Lightning reward event一致。
- 所有结果均包含reset/step/model/reward记录，失败任务保持失败状态。

上游git status在运行后为空，4个关键上游文件hash记录于versions.json。所有测试服务已结束，4张H100显存占用均为0MiB。

## 复现与产物

```bash
cd /HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl
bash scripts/run-suite-h100.sh --suite configs/dev-suite-v2.json --run-dir runs/my-v1-paired-run
python scripts/verify_harness_suite.py runs/my-v1-paired-run
# 参数/上游兼容测试，不启动模型服务
PYTHONPATH=.:deps/SimuHome .venv-baseline/bin/python -m unittest discover -s tests -v
```

run-dir必须不存在。原版direct→Lightning复现入口 `scripts/run-h100.sh` 保留，V1通过suite入口显式选择，避免把改进版冒充原版。

每轮保存config/tasks/versions、依赖清单、服务命令、日志及代码快照。每个episode保留原始模型请求/响应、原版事件、harness配置与校验记录、初始/最终状态、逐步reward、原版AgentResult、Lightning events/token IDs、最终summary与独立verification。聚合comparison.json含全部配对结果，没有只选成功轨迹。

## 结论范围

这是小型、静态开关灯dev测试，3项属于两灯变体、1项为单灯；不是官方benchmark，也不能据4项任务推断广泛成功率。两灯普通指令的原版提前结束，query-first原版在非法参数中循环，单灯原版本身可以成功。V1在这组任务上修复了前两类可见失败。

schema、prompt契约、finish gating、恢复提示同时变化，因此当前只能评价这个组合harness，不能断言收益全部来自JSON schema。单任务成本不总是更低；总token下降主要因为减少了原版20步失败循环。温度0仍不保证整个服务配置下逐字复现，仿真时间也随wall-clock推进；这里比较静态设备目标。

检索embedding、环境/定时目标、安全stress、多模态未覆盖。下一阶段应扩展独立dev任务并对schema/finish gating/recovery分别消融，再进入更大规模评测；当前继续不做SFT/GRPO。
