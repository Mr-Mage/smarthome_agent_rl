# 原版 SimuHome baseline 与 Lightning 接入报告

核对日期：2026-10-03（北京时间）。结论：已在用户 H100 上直接运行原版 ReAct，再将同一个原版 agent 接入已有 Lightning runner。真实模型→工具→环境反馈→reward 记录通路可用；**原版 agent 在当前 1.5B 模型上的两项开发任务都未成功**。本报告不把执行完成或日志校验通过记为任务成功。

## 为什么调整实现顺序

先前自写 ReAct 的 V0 也使用 Lightning 执行，但缩减工具、提供完整状态，并在目标满足时自动结束。它只能证明最小通路，不能代替 SimuHome 官方 baseline。当前按“原版运行→迁移 Lightning→再做可比较的改进”组织项目，默认入口已切换。历史 V0 与结果保留。

SimuHome 是 git clone 的外部原版，固定提交 `83d28837b69f0cbf1bc02ed5334cb8b561a9d54d`；运行前后 git status 都为空。agent/provider/tools/prompt 源码 SHA256 记录在每次 runs 的 versions.json。我们自己编写启动、HTTP adapter、状态 verifier、奖励、采集器和 Lightning 入口，没有重写官方决策循环。

原版 ReActAgent / OpenAIChatProvider、20个完整工具、system/user prompt、few-shot、JSON解析、显式 finish 和最大20步保持原版。外部观察通过公开 trace_fn 与 provider HTTP response hook；额外查询得到的完整状态仅供 verifier，不送入原版模型上下文。

## 数据、模型与依赖

- SimuHome 本体及其自带 benchmark/文档资源已经拉取。两项任务是独立 dev 任务，没有使用官方 benchmark 生成训练数据，也未新下载 SFT/RL轨迹。
- 复用 H100 已有 `Qwen2.5-1.5B-Instruct` 权重，没有下载新模型。它是接入检查用的模型；并非上游 README 中示例的 gpt-5-mini/gpt-4.1/qwen3-30b-instruct 模型配置。因此当前不是论文/官方 benchmark 复现。
- 复用用户 Agent Lightning v1.0.2、Python3.12.12、vLLM0.12.0。独立 Python3.13.14 baseline 环境安装了66个上游锁定依赖，wheel均按官方 SHA256 校验。pip check 和原版 imports 通过。
- 4张 H100 80GB 中仅GPU0用于此次推理；结束后4张卡显存占用都为0 MiB。未启动 Trainer、SFT、GRPO或参数更新。

## 实测对照

| 任务与运行 | 原版 direct | 同原版 Lightning |
|---|---:|---:|
| 自然语言两灯控制（01）：任务成功 | false | false |
| 01：模型调用 / 步数 | 1 / 1 | 1 / 1 |
| 01：设备工具调用 | 0 | 0 |
| 01：tokens / reward | 4368 / -0.014368 | 4368 / -0.014368 |
| 明确先查询设备（03）：任务成功 | false | false |
| 03：模型调用 / 工具反馈步数 | 20 / 20 | 20 / 20 |
| 03：实际工具尝试（不含格式解析失败） | 17 | 13 |
| 03：非法动作 / 基础设施错误 | 18 / 0 | 18 / 0 |
| 03：tokens | 116170 | 117119 |
| 03：reward | -2.116170 | -2.117119 |

01里模型没有查询设备，直接声称灯不存在并 finish；真实状态目标均未满足。Lightning状态 succeeded 仅表示原版程序正常返回。

03是额外的接入开发任务，用户 query 明确要求先 get_room_devices，再控制两灯并验证。两边都完成了设备列表和结构查询，但控制动作多次遗漏必填 args、使用非法参数，部分 finish 输出也未通过原版格式解析；最终达到20步而没有合法显式 finish，上游抛出 AgentExecutionError。Lightning状态为failed，保留原版失败状态并采集负reward，不伪造成功。命令退出码1是原版 episode失败的结果，详见 failure.json/error.json。

02是同一查询任务的早期采集版本，原始日志保留；其中 success响应的 error:null 被采集器误标为2次基础设施错误，且上游抛错时未发送负reward。03只修正采集分类和失败reward，并重跑；不修改原版 agent，也不通过修改历史记录隐藏问题。02不作为最终指标。

另外，01～03原始summary里的tool_calls计的是observation反馈数量，包含格式解析失败；这些原始文件保持不变。独立验证器按原版action事件重新计数，03实际工具尝试分别为17/13，并写入verification.json。交付源码已将tool_calls改为实际工具尝试，另外保留tool_feedbacks，标记metrics_version=2；这项计数修正未改变任何决策，未另跑模型。

## Lightning 接入验证

使用已有 v1 Local Controller 的 `run()` 入口。OfficialBaselineAgent 在独立Python3.13子进程运行原版agent，继承Controller注入的Gateway模型proxy和event地址；is_train=false。跨Python版本隔离由入口处理，决策逻辑仍在上游。

03 rollout id：`669d606e6e70456b8527890489ef0742`；01：`cb68f5131d57418bab4cb86cf5e0c76f`。
03有1个environment_reset、20个model_request、20个environment_step、1个reward。模型prompt/response token IDs均非空，事件数量、token usage、状态谓词、逐步reward之和与最终负reward一致。原版AgentExecutionError不会被吞掉。

逐项检查20个Lightning后端请求：Gateway保留原版messages、model、seed=42、temperature=0和response_format，仅增加return_token_ids=true用于记录。两阶段首请求完全一致，最终目标检查一致。

**没有宣称整条生成轨迹完全一致**：03前4个模型输入一致，第4次返回的thought文字开始不同，之后历史与动作序列分叉，token总量也不同。根因尚未证实；不能将这个差异视为迁移带来的性能变化。采样设置一致并不在当前服务配置下保证逐字复现。记录保留于model_calls.json与lightning_events.json。

## Reward

`r_t = terminal_success + 0.2*(phi_after-phi_before) - 0.1*invalid - 0.01 - 0.001*tokens/1000`。
phi是满足真实设备属性目标的比例；带符号progress防止反复撤销/重做刷分。查询、非法动作和finish都计成本。success须合法finish且真实目标全部满足。

本次03 progress/success为0，18次invalid扣1.8，20步扣0.2，加token成本，得到上述负reward。达到步数上限是可验证的任务失败，仍发送已采集轨迹的负reward；真正基础设施错误独立记录。当前没有训练器消费这些失败状态。

## 复现

H100项目：`/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl`。

```bash
cd /HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl
# 原始自然语言任务，重建两阶段baseline
bash scripts/run-h100.sh --run-dir runs/my-baseline
# 明确查询的开发链路任务；当前模型预期可能任务失败，日志照常保留
bash scripts/run-h100.sh --task-file configs/dev-query-first.json --run-dir runs/my-query-test
# 独立核验真实事件与reward；核验通过不代表任务成功
python scripts/verify_official_artifacts.py runs/my-query-test
```

run-dir必须不存在。setup脚本、全部参数、服务命令、环境依赖和代码快照均已保存；无需重装已有Lightning环境。新环境安装方法见README。

## 范围与下一步

这两项只是静态设备属性开发检查，不是官方benchmark分数，没有成功任务的新baseline结论，也不能跟历史自写V0做公平性能比较。官方aggregators开启，当前仿真tick仍随wall-clock推进；额外观测增加延迟，时序/环境控制任务还需制定时间协议。

原版get_cluster_doc保留，但需要embedding服务；当前仅提供聊天模型服务，两项任务都未调用检索。定时workflow、多轮用户交互、安全、多模态均未验证。

后续先固定独立dev任务集与预算，建立可重复的原版结果。第一项harness改进应针对当前实际失败：工具参数的结构约束/执行前校验，以及对非法动作的恢复。它应作为明确的新变体，在Lightning里与原版使用同模型、同任务、同预算比较。如更换更大模型，则单独报告模型变化；有稳定基线之后再考虑训练。

SimuHome原版依赖保持外部且pristine；交付包不包含其源码、模型权重或第三方wheel。历史及新失败记录均保留，用户Lightning仓库已有改动未被更改。
