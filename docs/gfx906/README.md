# gfx906 (MI50 / MI60) 优化分支工作手册

本目录记录 `gfx906-opt` 分支的工作模型、可从外部 fork 移植的优化特性清单，以及本机（8×MI50）实测基线。
这里的所有内容都是**下游**内容：上游官方不提供，也不会被上游改动覆盖（`docs/gfx906/` 是我们的私有路径）。

## 1. 分支模型：只用 `upstream/master`（remote-tracking）+ `gfx906-opt`

| ref | 角色 | 允许的操作 |
|---|---|---|
| `upstream/master`（remote-tracking） | **纯净上游**：diff 基准、rebase 目标、纯净基线（用 `--detach` 检出即可，不需要分支） | 只 `git fetch upstream`；**永不提交** |
| `gfx906-opt`（本地分支，主力） | 上游基线 + 全部优化 patch，**线性历史** | 日常开发；同步上游用 `git rebase upstream/master` |
| `origin/*`（PowerfulGhost/llama.cpp） | 个人 fork 远端 | 推送**必须先经确认** |
| `mxxm/master` 等 | 素材库（mxxm fork）：只取材 | 只 `git fetch`；**绝不 merge 进 `gfx906-opt`** |

> 另有历史分支 `master`（= `origin/master`，升级前的旧上游点 `8a1a9b512`），仅作回滚参考；
> 想留回滚点又不想留分支，可以打 tag：`git tag rollback/pre-30217 8a1a9b512`。

### 为什么不需要一个本地“镜像分支”

`upstream/master` 本身就是 `git fetch` 维护的本地引用，可以直接用于 diff、range-diff、`git worktree add --detach`、`git rebase`。
一个额外本地分支只在“想对它做 `--ff-only` 前移”时有意义，而那件事 `git fetch` 已经做了。

**纯净基线怎么拿（不建分支）**：

```bash
# 只读的纯净基线工作树（detached HEAD，不会污染分支）
git worktree add --detach ../base upstream/master
# 或者临时切过去：
git checkout --detach upstream/master
```

> ⚠️ 绝不要在 `upstream/master`（或它的 detached HEAD）上提交：那会产生一个不属于任何分支的孤儿提交；
> 也不要试图 `git commit` 到 remote-tracking ref（git 会拒绝）。

### 为什么同步 `gfx906-opt` 不能用 `--ff-only`

`--ff-only` 成立的前提是「目标分支是上游的**后代**」。一旦 `gfx906-opt` 上落了 patch、而上游又往前动了，
两者就是**分叉**（behind + ahead），git 会直接拒绝：

```
$ git merge --ff-only upstream/master        # 老上游基点 + 一个 patch
fatal: Not possible to fast-forward, aborting.
```

「把上游的新提交放到我们 patch 的**下面**」只有一个动作：**rebase**。

### 日常同步（唯一需要的分支就是 `gfx906-opt`）

```bash
# 0) 记下当前状态（reflog 会保留旧 tip，也可另存一份 diff 比对）
git worktree list                       # 确认没有别的分支占着主工作树
git fetch upstream                      # 只更新 upstream/master

# 看两边差在哪里
git log --oneline --no-merges upstream/master..gfx906-opt   # 我们的 patch 列表
git log --oneline gfx906-opt..upstream/master               # 上游新增的提交

# 1) 把我们的 patch 重放到新上游之上
git rebase upstream/master
#   显式等价写法：git rebase --onto upstream/master <上一次的上游基点> gfx906-opt

# 2) 校验 patch 没被吃掉/变形（upstream/master@{1} = fetch 前的旧上游）
git range-diff upstream/master@{1}...gfx906-opt@{1} upstream/master...gfx906-opt
#   退而求其次的两条不变量：patch 列表行数不变 + 累积 diff 不意外增长
git log --oneline --no-merges upstream/master..gfx906-opt
git diff --stat upstream/master...gfx906-opt

# 3) 出问题就地回滚
git rebase --abort                      # 还在 rebase 途中
git reset --hard ORIG_HEAD              # 已经结束但结果不对（rebase 会留下 ORIG_HEAD）
git reset --hard gfx906-opt@{1}         # 更早：从 reflog 里挑一个 tip
```

推荐开局先打开冲突缓存：

```bash
git config rerere.enabled true      # 同一处冲突只解一次
git config rebase.autostash true    # 有杂项改动时自动 stash
```

> ⚠️ `gfx906-opt` 已推送到 `origin/gfx906-opt`。rebase 会**改写历史**，同步后推送需要 `--force-with-lease`——属于必须显式确认的操作。

## 2. 加一个 patch 的完整流程

1. 先对齐：`git fetch upstream && git rebase upstream/master`
2. 写 patch：**优先新增独立文件**，其次改局部函数；一个特性一个提交，控制在数百行以内
3. 编译（容器 `qwen-flash-llama-builder:v1`）：

   ```bash
   cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
         -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DLLAMA_CURL=OFF
   cmake --build /build --target llama-cli llama-bench llama-server -j 24
   ```

4. 冒烟：`llama-bench -p 512,2048,4096 -n 128 -r 3`（同一模型、同一 `-lm` 口径，否则不可比）
5. **一致性校验**（并行/数值路径改动必做）：同一 prompt 与 `-sm layer` 参考配置做 token diff
6. 把数字与结论写进 [`baselines.md`](baselines.md)，**必须带 commit hash**
7. patch 若被上游吸收或不再需要：`git rebase -i` 里 drop

### 验收清单

- [ ] `cmake --build` 无新增 warning / error
- [ ] 冒烟性能不低于基线（对比 `baselines.md`）
- [ ] 并行/数值路径改动：token 输出与参考配置一致
- [ ] 本目录文档同步更新

## 3. 目录

| 文件 | 内容 |
|---|---|
| [`features.md`](features.md) | 可移植特性清单：来源 commit、涉及文件、冲突等级、实测收益、采纳状态 |
| [`baselines.md`](baselines.md) | 实测基线：环境、命令、数字、适用范围与坑 |

## 4. 相关路径（本机）

| 路径 | 内容 |
|---|---|
| `llama.cpp` (本仓库) | `gfx906-opt` 工作树 |
| `../base`（可选） | 纯净上游基线工作树：`git worktree add --detach ../base upstream/master` |
| `../mxxm-master` | mxxm fork 的 detached worktree（取材用，`git worktree remove mxxm-master` 可清理） |
| `../build-tp` | 官方代码的 gfx906+HIP+RCCL 构建 |
| `../build-mxxm` | mxxm fork 的构建 |
| `../KNOWLEDGE.md` | 项目知识库（性能/负面结论），与本目录互为补充 |
