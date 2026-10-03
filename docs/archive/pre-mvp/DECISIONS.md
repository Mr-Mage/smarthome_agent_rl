# 设计决策记录

更新：2026-10-03（Asia/Shanghai）。本轮根据当前代码、历史证据和明确的用户指令整理，**没有新增实现**。

“用户约束”表示原计划或用户明确选择；“实现选择”表示 agent 在完成授权任务时自行决定的具体方式，不代表用户逐项认可。项目根没有 Git 历史，因此无法仅凭当前文件精确归属每一行；以下区分的是可核实的约束与实现方案。

当前事实见 [PROJECT_STATUS](PROJECT_STATUS.md)，剩余计划与停止点见 [PLAN](PLAN.md)。

## D-01：复用上游 ReAct、工具与模拟器

**来源：用户约束；外部包装方式为实现选择。** 保留 SimuHome 源码和既有 Lightning 用户修改，不重写决策循环。通过项目 runner、provider decorator、trace/HTTP hook 和 ToolConfig 扩展外部集成。

原因：使基线与上游行为、失败处理可比较，避免同时改变循环、工具与模型。代价：依赖固定版本的类、私有 SDK 结构和错误文本，升级成本高。当前 SimuHome pristine；Lightning 五个已有修改文件与未跟踪测试保留。

相关文件：`baseline.py`、`run_official_episode.py`、`generation.py`、`structured.py`。

## D-02：合法任务失败作为 rollout 数据

**来源：实现选择，服务于保留全部失败的用户要求。** 当前使用 `BaselineAgent`，让预算/连续拒绝的合法失败保留完整轨迹与 reward，并允许 worker 正常完成；网络、模型服务或 reward 交付错误仍失败。

原因：Lightning worker failure 和任务目标失败是两层状态，若混在一起会丢失负例或造成选择性重跑。代价：上游 `AgentExecutionError` 目前通过类型名与消息识别，较脆弱。错误分类和原始 error 都需保存。

旧 `OfficialBaselineAgent` 仍保留其抛错行为，不能混作当前 worker。相关文件：`baseline.py`、`official.py`、`run_official_episode.py`、对应 failure-data tests。

## D-03：只补记第三次拒绝，不新增反馈

**来源：实现选择。** 上游第三次连续拒绝在发出 observation 前抛异常。观察器根据已有 trace 记录已经发生的模型 turn，注明 `agent_feedback_emitted=false`，计入 usage、非法动作与成本；不把补记内容送回模型。

原因：避免低估失败成本，同时保留原行为。代价：需要同时理解 upstream_events 与 observer_events，不能把保存的 observer 记录当成模型实际读到的反馈。

相关文件：`accounting.py`、`run_official_episode.py`、`test_abort_accounting.py`。

## D-04：采样、预算与环境只在模型子进程覆盖

**来源：保持既有模型/参数/Conda/venv 为用户约束；受限覆盖与 SDK 包装为实现选择。** 9B、seed42/engine0、.7/.8/top_k20、20/2048 预算保持既有协议。CUDA_HOME/CUDA_PATH 只对生成模型子进程设为 CUDA12.8，FlashInfer 使用独立缓存，未改 Lightning 激活脚本或原 Conda。

原因：实测继承 CUDA13 会导致 FlashInfer libcudart 链接失败；只覆盖子进程能保留用户环境。代价：绝对路径与既有包版本使跨机器迁移复杂。当前实际 `/proc` 环境与 argv 已核验。

相关文件：`generation.py`、`run_harness_suite.py`、9B/cu128 配置、model_environment_audit。

## D-05：H1 使用对象表示和外层工具 schema

**来源：用户 P1 计划；当前 StructuredProvider 实现来自已有代码。** 使用已有 `structured-schema`，关闭 finish_guard、recovery、guidance。工具 docstring 提取外层参数；`call.arguments` 是对象，再转换回上游 action_input；历史 assistant 消息同步转换，仅加 BASIC_CONTRACT。

原因：运行最小组合对照，避免新增恢复策略混入成绩。代价：这是表示、契约、历史序列化与 schema 的组合，不能独立归因 grammar；外层 `execute_command.args` 为 object，但没有约束设备命令内部 `Level`/`TransitionTime` 等语义。

实测：外层参数错误证据 B0/H1 为 12/0（跨三轮），但严格成功均 12/16，H1 非法动作及总体成本增加。保留 H1 结果，不将它替代 B0 或宣布提升。

相关文件：`structured.py`、`run_official_episode.py`、`report_p1.py`。

## D-06：首请求冻结与 token-ID 对齐

**来源：用户 P1 计划；SHA256、规范化方法与 worker 为实现选择。** 从验收 P0 原始首请求冻结完整 messages/schema/采样；direct 同样添加 `return_token_ids=true`。同时记录原文本与仅含工具/参数的规范化动作。

原因：Gateway val 模式会修改 model/temperature 和 token 标记，必须比较实际后端请求体；thought 变化不能等同于动作变化。代价：冻结材料包含服务器绝对 source 路径，本机诊断重放依赖路径处理，不能随意改原证据。

相关文件：`p1.py`、`p1_replay.py`、`p1_diagnostics.py`、`verify_p1_diagnostics.py`。

## D-07：使用三轮、顺序配对与整轮 fail-stop

**来源：用户 P1 协议；runner 参数和目录实现为实现选择。** 每轮重启服务，每个任务相邻执行两组并交替先后；第1/3轮正序 offset0，第2轮反序 offset1。保存实际排程。基础设施故障整轮中止，新建完整轮，不以补跑单项替代成绩。

原因：减少固定先后偏差，保留真实失败和可重复证据。代价：三轮相同16项不是48项独立样本；顺序、热状态和重启未因果隔离；当前 run_p1 硬编码当次日期/目录，不能通用恢复。

相关文件：`p1.py`、`run_harness_suite.py`、`run_p1.py`、`verify_harness_suite.py`。

## D-08：沿用实际任务初始化

**来源：既有 P0 行为；在 P1 保持相同实际任务为用户约束。** 原 runner 为房间设置标准 state 并启用 aggregators；P1 冻结验收运行的实际 tasks，而非只看早期任务模板。

原因：上游房间状态查询需要这些环境信息；任务输入必须与已运行的环境一致。代价：配置模板和实际 tasks 不能混为一谈。未来改变环境初始化必须形成新协议，不能悄悄沿用旧成绩。

相关文件：`generate_dev_suite.py`、`run_harness_suite.py`、`p1-dev-frozen.json`、各轮 tasks/run_protocol。

## D-09：新建 CPU BGE 检索服务及匹配索引

**来源：用户明确要求“保持原协议，先完成真实检索后端再实验”；具体模型/CPU/服务接口为 agent 自行选择。** 原工具和 1536 维 Ada 索引来自 upstream，但没有可用对应 embedding 后端；生成 vLLM 不提供所需 embedding。选择固定 revision 的 BAAI/bge-small-en-v1.5，在 CPU 运行，重建 384 维独立索引。

原因：不依赖未配置的外部 embedding 凭据，保留一张 H100 给9B；不能把384维BGE查询直接用于1536维Ada索引。两组共享相同修复，原索引保留。

代价：改变了检索模型与索引，因此 P0 只作历史参照，不能宣称恢复成原 Ada 服务或做同环境 P0/P1 因果比较。关键词 smoke 和正式调用证明功能可用，不证明检索质量等价或最优。

相关文件：`serve_doc_embeddings.py`、`retrieval.py`、下载/索引脚本、检索共享配置。

## D-10：复用原文档加载、切分与 ToolConfig 注入

**来源：保留原工具协议为用户约束；注入方式与清单为实现选择。** 继续使用原 load_docs、1000/200 chunking、FAISS 和 get_cluster_doc；70份文档生成4130 chunks。通过已有 `ToolConfig(db=...)` 注入匹配数据库，不修改 upstream 工具。

原因：保持文档与工具格式，修复真实依赖而不伪造工具返回。query 使用 BGE 检索 prefix，文档不加 prefix；CLS pooling/归一化都在本地服务实现。

代价：模型 tokenizer 的512长度截断与1000字符切分不完全等价；尚无独立检索准确率/覆盖率评测。索引加载依赖可信本地 pickle 与 hash 清单，后续维护需要约束其来源。

## D-11：隔离兼容 hub 包，保留原 Conda

**来源：保留原环境为用户约束；具体兼容包复用为实现选择。** 新 embedding venv 继承既有环境包；Transformers4.57.6 与该环境 hub1.23.0 不匹配，复制既有 Lightning 环境的 hub0.36.2 及 dist-info 到新 venv，并记录模块哈希。

原因：当时网络安装失败且不得改用户 Conda，复用现成兼容包使 CPU 服务真实运行。代价：这是当次环境修复，不是完整锁定的标准安装流程；需要未来单独设计依赖重建方案。现有两个 Conda 未修改，venv 没有打进交付包。

相关文件：`isolate_embedding_dependency.py`、embedding-venv-isolation、各轮依赖 freeze。

## D-12：失败证据按可核实信息分类

**来源：用户要求不凭 HTTP400 猜原因；源码签名解析与恢复事件为实现选择。** 使用先前轨迹的设备结构判断命令存在/readonly；结合固定模拟器源码 AST 方法签名判断内部参数名缺失或错误大小写。只有状态码但缺具体证据时保留 undetermined。

原因：需要区分外层 JSON、设备语义和恢复问题。代价：AST/docstring 解析只支持目前源码结构，恢复事件可以与非法动作事件重叠；事件数不是任务数或完整恢复率。

相关文件：`report_p1.py`、error_evidence、device_semantics_evidence。

## D-13：检索、诊断、中止轮成本分开记账

**来源：用户原生成 reward/预算与排除规则；具体成本字段为实现选择。** 保留 prompt/completion 与 generation reward；单列检索调用、embedding tokens、CPU延迟及索引构建成本。诊断、修复前八条链路和13个中止 episode 不计正式成功率。

原因：不能让修复改变原 reward，也不能隐去真实额外开销。代价：报告里的生成/episode成本不是完整部署总成本；模型加载、下载和服务启动另有日志，不在 episode 延迟里。

相关文件：`retrieval.py`、`report_p1.py`、各轮 retrieval_events。

## D-14：单模拟器顺序运行与资源所有权

**来源：一张 H100 和顺序实验为用户约束；进程组与独立分片设计为实现选择。** 原 reset 为全局操作，当前一套服务顺序运行；runner 只终止自己创建的进程组。多资源并行需要独立 GPU、端口、simulator、Gateway、Controller，不共享 reset。

原因：避免任务之间污染环境、保留用户已有连接与进程。代价：吞吐低，多9B并行尚未实测；CPU检索端口也需在未来多分片协议中明确隔离。当前 H100 已释放。

相关文件：`run_harness_suite.py`、`run_parallel_dev.py`、`test_parallel.py`。

## D-15：保存历史材料并集中交付

**来源：完整可追溯交付为用户要求；多目录归档与 finalizer 为实现选择。** 保留中止轮、旧报告、原始轨迹；生成最新报告、学习文档、源快照、清单和归档，分别存服务器和本机。

原因：失败和环境修复需要可审计，不能只留最佳成绩。代价：多个副本、残留过期段落和固定日期脚本降低可维护性；本机核验会重写verification造成换行哈希差异。原归档完整，但展开副本不能笼统叫不可变证据。

相关文件：`finalize_p1.py`、本机 `deliver_final.py`；问题 I-02/I-03/I-04/I-05。

## D-16：不追加变体或训练来追逐提升

**来源：用户明确约束。** P1 即使 H1 无收益也交付真实结论；没有新设备能力反馈、错误反馈、任务扩展、SFT或RL。

原因：避免事后选变体与测试污染。下一步仅建议先做工程基点/维护审查，再设计设备能力反馈的预先冻结对照。建议尚未转成实现，也未获此次review后的继续授权。

## 本轮新增的控制决策

用户本轮要求暂停开发且允许创建/更新三份文档。当前只在本机根创建它们，将代码、配置、旧报告、原证据包和服务器保持原样。上述技术债只记录，不修复；文档完成后停止。未来决策需在用户恢复开发后再落地。
