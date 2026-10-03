# SmartHome-Agent-RL：原版 ReAct 与结构化 Harness

代码已纳入 Git：默认分支 `main`，私有远程仓库 [Mr-Mage/smarthome_agent_rl](https://github.com/Mr-Mage/smarthome_agent_rl)。日常提交、同步、SSH 与依赖恢复见 [Git 工作流](docs/Git工作流.md)。模型、环境和原始实验材料不进入仓库。

> P1 已完成：72 次固定请求、8 个链路诊断及 96 个三轮配对 episode 均独立核验。B0 三轮严格成功 [12, 12, 12] /16；H1 [12, 12, 12] /16。平均差 +0.00 项。 不训练；详见 [P1 三轮报告](docs/p1-20261003/P1-三轮配对报告.md) 和 [稳定性报告](docs/p1-20261003/P1-稳定性报告.md)。后文旧进度保留为历史记录。

> P1 修复前历史状态：72 次请求 + 8 个链路诊断已独立核验；正式第一轮因 get_cluster_doc 的检索依赖故障中止，完整轮数 0/3，无三轮成绩。需要 FAISS-compatible embedding 后端配置；本轮服务已释放，不训练。见 [P1 执行报告](docs/p1-20261003/P1-执行与阻塞报告.md)。

## 当前自建 B0 入口

学习入口：[Agent 模块设计与 Lightning 基线学习文档](docs/agent-learning-20261003/Agent模块设计与Lightning基线学习文档.md)。按真实代码解释 ReAct、工具、模型服务、Lightning、reward 和独立评测，包含伪代码、源码快照及成功/失败轨迹；9B 重复稳定性与三轮 harness 对照已在 P1 完成，详见顶部报告。

2026-10-03：**Qwen3.5-9B 的 P0 已完成，固定 16 项 dev 严格成功 12/16（75%），独立产物核验通过。** 四项失败均为调光目标未满足；不训练。完整协议、成本和失败证据见 [P0 9B 基线验收报告](docs/P0-9B-baseline-20261003.md)。

15 个模型文件的官方哈希全部通过；加载及纯文本推理已验证。下载沿用本机 relay / SSH 反向转发，权重直接写入 H100。Conda、激活脚本和依赖保持原样；仅模型子进程使用既有 CUDA 12.8 和独立 FlashInfer 缓存。

```bash
# 默认 9B 配置包含模型子进程的 CUDA 12.8 路径覆盖
bash scripts/run-single-h100.sh --run-dir runs/my-qwen35-b0
python scripts/verify_harness_suite.py runs/my-qwen35-b0
python scripts/summarize_baseline.py runs/my-qwen35-b0
# 历史工程回退模型
bash scripts/run-single-h100.sh --config configs/b0-engineering-1.5b.json --run-dir runs/my-engineering-b0
```

当前运行：`runs/b0-qwen35-9b-20261003-03`；配置：`configs/b0-qwen35-9b-cu128-runtime.json`；队列已结束并核验，见 `logs/b0-queued-state.json`。原冻结配置和失败运行完整保留。P1 稳定性与最小 schema 对照已完成，见本文顶部的新报告。

当前方案：Qwen3.5-9B → 自建 ReAct 基线 → harness 对照 → SFT → RL。原仓库结果复现已取消，32B 下载与评测已停止。以 [项目方案与进展简报](docs/项目方案与进展简报.md) 和当前 P0 报告为准；后文为历史工程记录。

当前默认入口先直接运行 **SimuHome 原版 ReAct agent**，再把同一个 agent 接入用户已有 Agent Lightning v1.0.2。只做 symbolic online rollout，不启动训练。

## 历史 1.5B 单卡 H100 结果（2026-10-03）

新节点已恢复在线推理。复用现有 `qwen36-vllm` 的 torch 2.10.0+cu128 / vLLM 0.19.0 作为模型后端；Lightning 仍用原 v1.0.2 环境。原 CUDA13 后端在新节点初始化失败，未修改驱动或用户环境。历史后端不同，历史数字不能直接与本轮比较。

- 原仓库 CLI：qt3 feasible seeds 1–3、Qwen2.5-1.5B-Instruct，成功 0/3，原始失败记录与原聚合已保存；属于流程试跑，非论文结果复现。
- 独立 dev：16 项任务 × 原版/schema-only，32 次 Lightning online rollout 已完成。成功 0/16 → 4/16，调用 233 → 115，非法动作 182 → 54；调光和风扇两版均未成功。
- H100 与本地独立核验全部通过；原 SimuHome 保持 pristine，不训练。第三次格式拒绝的成本漏记已在外部观察器修复，metrics_version=4；13 项检查通过。
- 当时排队的 Qwen3-32B 下载与原 CLI 评测已按新方向停止，不恢复此历史队列。

完整证据与失败分析见 [单卡 H100 报告](docs/单卡H100运行与对齐报告.md)。本轮结果：`runs/single-h100-dev-20261003-04/verification.json`；原 CLI：`runs/single-h100-20261003-03/official/verification.json`。下载状态快照：`logs/single-h100-status.json`，实时状态以 H100 日志为准。

```bash
bash scripts/run-suite-h100.sh --config configs/single-h100-dev-model-cu128.json \
  --suite configs/dev-expanded-v1.json --harnesses upstream-react structured-schema \
  --gpu 0 --run-dir runs/my-single-h100-dev
python scripts/verify_harness_suite.py runs/my-single-h100-dev
```

四卡并行入口和资源隔离检查仍保留在 `scripts/run-parallel-h100.sh`，单卡工作流入口为 `scripts/run-single-h100.sh`。单卡上 GPU 工作顺序执行，CPU 预检和模型下载可以并行。

## 历史四任务消融结果与复测

固定4项dev任务、5种配置、20次Lightning rollout：原版成功1/4；仅对象格式/schema、完整V1、无动态恢复提示均4/4；无finish gate为3/4。失败单灯另做5次相反顺序诊断，五种配置均成功，原版单灯只需2次调用。出现了首请求相同而输出不同的实测情况，因此额外组件收益和微小成本差异尚不能稳定归因。完整记录见 `docs/Harness消融与稳定性报告.md`，不要将这些小样本当作官方benchmark。

```bash
bash scripts/run-suite-h100.sh --suite configs/dev-suite-v2.json --run-dir runs/my-ablation \
  --harnesses upstream-react structured-schema structured-no-gate structured-no-recovery structured-v1
python scripts/verify_harness_suite.py runs/my-ablation
```

保留所有失败、不训练、不自动把某个配置宣布为通用最优。summary metrics_version=3明确success同时要求目标满足、合法finish及正常执行，goal_success单独记录环境目标；历史原始文件与指标定义见各报告。

## 历史 Structured Harness V1 对照

在H100同一Qwen2.5-1.5B-Instruct上，两版都通过Lightning完成配对rollout。去重dev集4项任务：原版成功1/4，structured-v1成功4/4；总模型调用24→16，非法动作18→1，总tokens135407→76023。不是官方benchmark结果；schema/提示词/finish gating/恢复提示是组合变体。完整证据见 `docs/Structured-V1对照报告.md`。

```bash
cd /HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl
bash scripts/run-suite-h100.sh --suite configs/dev-suite-v2.json --run-dir runs/my-v1-paired-run
python scripts/verify_harness_suite.py runs/my-v1-paired-run
```

新代码：`smarthome_agent_rl/structured.py`（原版provider的组合harness）、`scripts/run_harness_suite.py`（两版都在Lightning的配对runner）、`configs/dev-suite-v2.json`（固定去重dev任务）、`tests/test_structured.py`（6项上游兼容/历史参数错误检查）。每轮保留代码快照与全部失败。原版入口、历史V0和报告保留。

## 来源与职责

- `deps/SimuHome`：git clone 的外部原版仓库，固定提交 `83d28837b69f0cbf1bc02ed5334cb8b561a9d54d`；不修改、不随交付包分发。
- 原版配置的 `ReActAgent`、`OpenAIChatProvider`、提示词、few-shot、完整工具注册表、解析与显式 `finish` 规则均保持原样。structured-v1显式改变provider输入/输出契约和finish可用条件，但仍复用原版ReAct循环与工具函数，变化单独记录。
- 本项目自己编写外部启动、状态验证、日志采集和 Lightning 入口。完整状态只给 verifier，不会额外注入原版 agent 的模型上下文。
- 原先自己写的 `react.py` 保留为历史 V0，入口改为 `scripts/run-custom-h100.sh`。它提前给完整状态、缩减工具并在目标满足时自动结束，不能作为官方 baseline，也不能据其较少步数断言更强。

## 已准备的 H100 环境

项目：`/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl`。
Lightning 复用已有 Conda `agent-lightning`（Python 3.12.12）、Agent Lightning v1.0.2。新节点模型后端使用已有 `qwen36-vllm` 的 vLLM 0.19.0，当前复跑请使用带 `cu128` 的配置。
模型已存在：父目录 `models/Qwen3.5-9B`（完整校验通过），以及历史 `models/Qwen2.5-1.5B-Instruct`。推理只使用 GPU 0。
模拟器 `.venv-simuhome` 与官方 agent `.venv-baseline` 单独使用 Python 3.13.14，不改变原训练环境。
原版 agent 的 66 个依赖按上游 lock 固定版本，并经过 SHA256 校验离线安装；依赖清单见 `requirements-baseline.lock.txt`。
SimuHome 仓库自带 benchmark/文档等资源已经拉取；本次 rollout 使用我们独立生成的 dev 任务，没有下载训练轨迹或使用 benchmark 训练。
用户 Lightning 仓库原有的 5 个文件改动未被改变；保存于 `logs/agentlightning-existing.patch`。

## 历史单 episode 入口

以下是历史入口，使用原 CUDA13 模型后端；新单卡节点请使用顶部已验证的 cu128 对照入口。历史运行方式：

```bash
cd /HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl
bash scripts/run-h100.sh --run-dir runs/my-unique-run
```

默认 `configs/official-baseline.json`：本地端口 18080/18000/18181，temperature=0，model seed=42，max_steps=20，模型上下文32768。
支持 `--config`、`--run-dir`、`--gpu`、`--task-file`；run-dir 必须尚不存在。默认任务为自然语言两灯控制。
明确要求先查询的开发链路任务：

```bash
bash scripts/run-h100.sh --task-file configs/dev-query-first.json --run-dir runs/my-query-first-run
```

两种入口都依次执行 direct 与 Lightning，重新 reset 同一份任务。原版任务成功单独看 summary.success；达到上限等原版异常保留为 failed/退出码1，并记录已采集的失败 reward。只启动并清理自己持有的服务进程；端口被占用则停止。

新环境先运行 `bash scripts/setup-simulator.sh`，再运行 `bash scripts/setup-baseline.sh`。
后者从 pristine 上游 lock 导出完整依赖；可用 `BASELINE_PROXY_URL`/`BASELINE_INDEX_URL` 配置下载，或者 `BASELINE_WHEELHOUSE=/absolute/wheels bash scripts/setup-baseline.sh` 离线安装。H100 的校验过的 wheelhouse 位于 `work/official-wheels`，不含在交付包中。

## Lightning 接入

本版本 Lightning 使用 Gateway + Local Controller 的 `run()` 入口。`OfficialBaselineAgent` 在独立 Python 3.13 进程运行原版 agent，继承 Controller 注入的模型 proxy/event 地址和 key。模型请求实际通过 Gateway 发给 H100 vLLM；reset/step/reward 发送为 Lightning events。`is_train=false`。

可使用 `python scripts/verify_official_artifacts.py runs/<run>` 独立核验事件与奖励（核验通过不代表任务成功）。

Python 3.12 与3.13隔离是依赖兼容处理；决策逻辑没有重写。Gateway 自动记录模型 token IDs，外部 wrapper 使用原版公开 trace 回调和 HTTP response hook 记录原始模型请求、响应和 usage。

## 代码结构

```text
smarthome_agent_rl/official.py      Lightning local-controller run() 入口
smarthome_agent_rl/adapter.py       HTTP reset、状态查询及历史 V0 动作适配
smarthome_agent_rl/tasks.py         独立 dev 任务
smarthome_agent_rl/reward.py        状态谓词、progress/invalid/cost
smarthome_agent_rl/react.py         历史自写 V0，不是默认 baseline
scripts/run_official_episode.py    启动原版 agent、只观察工具执行并记录奖励
scripts/run_official_stack.py      direct → Lightning 的服务与验证流程
scripts/run-h100.sh                默认原版 baseline 对照入口
scripts/run-custom-h100.sh         历史 V0 入口
configs/official-baseline.json     模型、端口、预算、奖励权重
configs/dev-query-first.json       独立的明确查询开发任务
runs/official-*/                   原始轨迹、模型请求和 Lightning events
```

当前summary的tool_calls按原版action事件计实际工具尝试，tool_feedbacks计所有observation；历史01～03文件的旧字段解释及独立审计结果见报告。

每次保存服务日志、命令/配置/版本/依赖、源文件哈希与代码快照；每阶段保存 task/initial_state/final_state、upstream_result/events、model_calls、trajectory、summary。`comparison.json` 分开报告执行状态、任务 success、首请求/目标检查一致性、动作序列和事件计数。

## Reward 与验证边界

每步：`terminal_success + 0.2*Δphi - 0.1*invalid - 0.01 - 0.001*tokens/1000`；phi 是已满足目标的比例。查询和显式 finish 也计成本。progress 是有符号差值。success 由真实设备属性验证，只有显式 finish 且全部目标满足才发成功分。

Lightning `succeeded` 表示进程完成；必须另看 `summary.success` 判断任务完成。当前仅支持设备属性 eq/le/ge；不是官方 evaluator/benchmark 分数。
官方 `get_cluster_doc` 工具保留，但它需要 embedding 服务，本次仅有聊天模型服务；如果调用会得到检索服务错误。没有冒用外部 API key。当前两灯任务未覆盖该能力。
房间 aggregators 已打开以支持官方查询；真实仿真 tick 随 wall-clock 推进，两个运行的时间相关观察不保证完全一致。额外状态查询也增加延迟，当前只验证静态设备属性。
完整结果和限制见 `docs/原版baseline与Lightning接入报告.md`；旧 `docs/验收报告.md` 仅记录历史自写 V0。

## 上游与许可

[SimuHome](https://github.com/holi-lab/SimuHome/tree/83d28837b69f0cbf1bc02ed5334cb8b561a9d54d)：CC BY-NC-ND 4.0；作者与接口分析见 `docs/SimuHome接口与边界.md`。外部依赖保持 pristine，通过原版 API 和原版 Python 类集成，不分发其源码。
[Agent Lightning](https://github.com/microsoft/agent-lightning/tree/d381995396274039f2bb1cbe5ff42ac8067f4e47)：MIT。
