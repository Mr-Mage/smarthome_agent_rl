# SmartHome Agent Harness

在原 SimuHome ReAct 与 Agent Lightning 上比较 Guard、Verify/Repair、Context 三类机制。
使用 Qwen3.5-9B actor、本地 Qwen3.6-35B-A3B judge 和官方 12 类 benchmark。

- 开发约定：`AGENTS.md`
- 功能节点：`PLAN.md`；结果：`docs/nodes/`
- 环境与上游版本：`dependencies.lock.json`，用户 Lightning 补丁：`patches/`
- 历史资料：`docs/archive/pre-mvp/`；大文件证据不入 Git。

服务器通过 `ssh h100` 访问；激活已有 `../activate-agent-lightning.sh`，模型使用 `qwen36-vllm`，模拟器与 episode 使用各自 venv。不得混用或改写环境。
