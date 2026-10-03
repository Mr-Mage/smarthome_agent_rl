# SimuHome 接口与边界（含历史 V0）

核对日期：2026-10-03（北京时间）。上游：holi-lab/SimuHome，固定提交 `83d28837b69f0cbf1bc02ed5334cb8b561a9d54d`。
Agent Lightning 使用用户已跑通的 v1.0.2，提交 `d381995396274039f2bb1cbe5ff42ac8067f4e47`。

## Episode / task

上游 episode JSON 包含 `meta`（seed/query_type/case 等）、`query`、`user_location`、`eval.required_actions`、`eval.goals`、`initial_home_config`。
`initial_home_config` 提供 base_time、tick_interval、rooms，以及每个设备的 device_id/device_type/attributes。
属性采用扁平路径，例如 `1.OnOff.OnOff`；重置接口接受完整配置，而不是 `reset(seed)`。
上游生成器先生成家庭、目标与初始快照，再通过模型生成用户 query。

本项目 task 使用独立的 `smarthome-task-v1`：seed 只控制自己的 task generator，然后把显式初始配置 POST 给模拟器。
验收任务属于 dev，不读取、复制或筛选 benchmark 任务。当前任务空间只有两种房间名称与两个开关灯，不代表大规模任务生成能力。
Goal 存在 verifier 中，不发送给模型；模型仅接收用户指令、工具说明和设备状态。

## HTTP / tool schema

所有模拟器 API 以 `/api` 为前缀。正常响应：`{status: {code: 200, message: ...}, data: ..., error: null}`。
reset 可能在 HTTP 200 响应体中返回非 200 的 status.code，adapter 同时检查 HTTP 和响应体。

| 功能 | 路由 | 参数 |
|---|---|---|
| 健康 | GET `/api/__health__` | 无 |
| 重置 | POST `/api/simulation/reset` | 完整 SimulationConfig |
| 状态 | GET `/api/home/state` | 无 |
| 设备结构 | GET `/api/devices/{id}/structure` | device_id |
| 命令 | POST `/api/devices/{id}/commands` | endpoint_id, cluster_id, command_id, args |
| 属性写入 | POST `/api/devices/{id}/attributes/write` | endpoint_id, cluster_id, attribute_id, value |
| 房间设备 / 环境 | GET `/api/rooms/{id}/devices`、`/states` | room_id |
| 时间推进 | POST `/api/simulation/fast_forward_to` | to_tick, 可选 room_ids |
| workflow | POST `/api/schedule/workflow` | start_time, steps |

V0 暴露前五种操作中的状态、结构、命令、属性写入，并添加本地 `finish`。reset 由 runner 控制。
`finish` 不调用上游，不改变状态，不自动成功。达到全部目标会结束 episode；达到 max_steps 则 truncated。
上游 OnOff 状态是只读属性，正确操作是 endpoint=1、cluster=OnOff、command=On/Off。
设备和命令是否合法最终由上游判断；adapter 先检查工具名称、字段与类型。
400/404/409/422 作为 invalid；超时、断线和 5xx 为基础设施错误，终止执行并保存 error。

## State transition / virtual time

上游 Home 内有仿真线程，每 tick 依次处理 time-aware devices、aggregators、scheduled workflows、API queue，再推进 current_tick。
房间环境支持温度、湿度、照度、PM10 等规则，设备控制与环境反馈通过聚合器关联。
`fast_forward=true` 表示仿真线程不按 tick_interval 睡眠，不等于“每个 agent action 恰好推进一步”。
`fast_forward_to` 只能推进，不能回退；还会处理途中设备、环境与 workflow。
reset 是进程级全局 Home 重置，因此同一服务不可并发多个 episode。V0 Controller maximum_size=1；未来并行需要每个 worker 独立模拟器。

V0 打开实际 API 仿真线程，但为保证开关灯测试可重置，禁用 aggregators，仅验证设备状态变化。
current_tick/current_time 仍随实际等待推进；同 seed 不保证逐 tick 相同。
当前观察只含 rooms（全部设备属性），不暴露 wall-clock tick 给 agent。未来温湿度/定时任务必须单独规定虚拟时间协议，并验证公平性。

## Benchmark / evaluator

官方 benchmark 保存在 `data/benchmark/`。README 提醒：当前快照经过增强模拟器重新生成，不等同论文 Table 1 数据。

| 类别 | 当前 evaluator 主要依据 |
|---|---|
| qt1 feasible | required_actions、finish 最终答复、LLM judge panel |
| qt2 feasible | required_actions，并 reset 到原始配置推进到相同 tick 得到无操作 baseline，比环境变化方向 |
| qt3 feasible | required_actions、最终设备属性 asserts |
| qt4 feasible | required_actions、when.at_tick ± tolerance_ticks 时间窗口内的目标设备 asserts；按时间顺序推进 |
| infeasible | 各类别独立 evaluator，涉及不可执行情况与答复评价；V0 不接入 |

上游 required_actions helper 检查工具名和参数是否出现，虽然接收 outcome 字段，当前匹配时没有强制 outcome.ok=true。
本项目 reward 不直接复用该 helper，而检查真实状态。它是独立的开发 verifier，不是官方 benchmark 分数。
官方 evaluator 会 reset/fast-forward 或调用 judge，不能不加区分地当作每步 reward。

## Reward

每一步：`r = terminal_success + 0.2*(phi_after-phi_before) - 0.1*invalid - 0.01 - 0.001*tokens/1000`。
`phi` 为满足目标谓词的比例，success=终止且全部满足。episode reward 为逐步之和，只有一个 scalar reward event 发送给 Lightning。
tokens 是所有模型请求的 prompt_tokens + completion_tokens，总量包含重复上下文；每步成本覆盖查询、错误解析和 finish。
进度用带符号差值，撤销条件会产生负分；不会通过往返操作净增进度 reward。
支持设备属性 eq/le/ge，eq 区分 bool 与 int。当前不支持环境/时序 predicate。

## 许可证与独立集成

[SimuHome README](https://github.com/holi-lab/SimuHome/tree/83d28837b69f0cbf1bc02ed5334cb8b561a9d54d) 声明 CC BY-NC-ND 4.0。
保留原版为项目外部依赖 `deps/SimuHome`，通过 HTTP 集成，不复制、修改或重新发布其源码，不把依赖打包入本项目交付压缩包。
作者：Gyuhyeon Seo、Jungwoo Yang、Junseong Pyo、Nalim Kim、Jonggeun Lee、Yohan Jo，ICLR 2026。
SimuHome 相关使用应遵守原许可；本实现没有推定商业使用或改作授权。

源码核查路径（均属于外部依赖）：`src/simulator/api/{routes,schemas,responses}.py`、`src/simulator/application/home_initializer.py`、`src/simulator/domain/home.py`、`src/agents/strategies/react_agent.py`、`src/pipelines/episode_evaluation/`。

## 2026-10-03 原版 baseline 接入补充

上述 V0 工具子集、完整状态观察、目标满足即结束、禁用 aggregators 描述只适用于历史 `react.py`。当前默认入口已改为原版 ReAct：完整 TOOL_REGISTRY，原版 prompt/few-shot/parser，要求显式 finish，不向模型注入完整状态。reset、特权状态查询及奖励由外部 wrapper 处理，官方工具仍直接通过上游 HTTP helper 执行。

为了支持原版 get_room_states，当前两灯任务开启 aggregators，并显式初始化房间温湿度/照度/PM10。与历史 V0 的设置不同，不能把两者的步数或token差异当作受控 harness 消融。direct 与 Lightning 则在每次运行内重置同一份配置。

原版 agent 已在独立 Python3.13 环境安装完整依赖。docs retrieval工具保留；没有本地 embedding 模型，当前不验证检索能力。源码保持不变并保存哈希。结果见 `原版baseline与Lightning接入报告.md`。
