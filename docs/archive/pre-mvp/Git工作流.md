# Git 管理与服务器工作流

2026-10-03 建库；服务器工程是此次导入的源码来源。

- 仓库：`git@zrj_git:Mr-Mage/smarthome_agent_rl.git`，对应 GitHub 私有仓库。
- 默认分支：`main`。功能或修复使用单独分支，通过 PR 合并。
- Git 管理源码、测试、任务/模型配置、依赖版本、文档和汇总报告。
- 模型权重、Conda/venv、运行日志、原始轨迹、FAISS 二进制索引、压缩包、密钥和本机覆盖配置不进入 Git。
- `runs/`、`work/`、`outputs/` 中的既有实验材料保留。历史源码快照及其学习文档链接依赖服务器原材料或完整交付包；快照不重复纳入 Git。
- 禁止以 `git add -f` 强行提交上述材料。首次导入没有启动训练或重新运行 GPU 实验。

## SSH 与网络

`zrj_git` 是服务器的 SSH Host 别名，私钥为 `~/.ssh/zrj_git`。
它连接 `ssh.github.com:443`，使用 `scripts/github_http_connect.py` 建立 HTTP CONNECT 隧道。
其他 Host 配置和密钥保留，修改前的配置备份在 `~/.ssh/config.pre-smarthome-git-*`。

代理默认 `127.0.0.1:7897`；这依赖当前已有的代理/SSH 反向转发，服务器离线或代理停止时 Git 网络操作会失败。
替换代理时可设置 `SSH_HTTP_PROXY_HOST`、`SSH_HTTP_PROXY_PORT`；不要在仓库内保存代理密码。
此 SSH 别名和私钥仅在当前服务器可用，新机器需要配置自己的 GitHub 身份，不应复制私钥。

```bash
ssh -T zrj_git
git ls-remote origin
```

GitHub 的认证成功消息会同时说明不提供 shell，`ssh -T` 返回 1 是正常现象。

## 日常开发

```bash
cd /HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengrj/agent/smarthome_agent_rl
git status --short
git switch main
git pull --ff-only origin main
git switch -c feat/your-change
# 编辑源码，然后运行与改动相关的检查。
git diff --check
git diff
git add scripts/changed_file.py tests/related_test.py
git diff --cached
git commit -m "Describe the concrete change"
git push -u origin HEAD
```

在 GitHub 创建 PR、审阅并合并后：

```bash
git switch main
git pull --ff-only origin main
```

工作区有修改时先提交或明确使用 `git stash`；不同历史先检查，不使用强推、`reset --hard` 或覆盖方式处理同步。
若今后需要本机编辑，先 clone 此仓库，使用本机自己的 SSH 别名或 HTTPS 登录；本机 push 后服务器 pull。
不再靠 tar.gz 覆盖源码。压缩包只用于实验数据、证据备份和阶段交付。

## 依赖与可复现性

`dependencies.lock.json` 固定 SimuHome、Agent Lightning 的上游提交。
这两个仓库独立管理，不将其模型、环境或整个源码重复加入本项目。
Lightning 的已有用户修改保存在 `patches/agent-lightning/user-changes.patch`，
额外测试保存在同目录的 `tests/server/test_http_limits.py`，原服务器依赖未修改。

新机器恢复依赖时，先检出固定提交；只在全新的、干净的 Lightning 工作区应用补丁：

```bash
git -C ../agent-lightning apply --check ../smarthome_agent_rl/patches/agent-lightning/user-changes.patch
git -C ../agent-lightning apply ../smarthome_agent_rl/patches/agent-lightning/user-changes.patch
cp patches/agent-lightning/tests/server/test_http_limits.py ../agent-lightning/tests/server/
```

当前服务器已经有这些修改，不能再次应用。模型 manifest 和 requirements lock 文件在仓库中；
模型下载、检索索引和原有环境继续保留在 Git 之外。

## 实验版本记录

每个新的实验开始前记录源码版本和是否有未提交改动：

```bash
git rev-parse HEAD
git status --porcelain
```

将结果存入该实验的运行目录。已经完成的 P0/P1 不补写成“运行于本次初始提交”：
它们发生在建库之前，仍以各自源码快照、原轨迹与校验记录为准。
