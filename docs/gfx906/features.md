# 可移植特性清单（取自 mxxm fork）

素材库：`mxxm`（https://github.com/mxxm-t/mx-llama.cpp.git），分叉点 `0f3a71be1`（2026-09-02），
取材时点 `eefc4e732`（2026-09-20，**落后当前上游 799 个提交**）。
fork 自分叉点以来：**115 个文件、+18455 / −940**；同期上游：1306 个文件、+196093 / −73832；
**双方都动过的文件 83 个**（占 fork 改动文件的 72%）——这就是"整体 merge 不可取"的量化依据。

## 冲突分级

| 级别 | 含义 | 处理方式 |
|---|---|---|
| A 无冲突 | fork 新增的独立文件，上游不存在 | 直接取（仍要接线 + 验证） |
| B 可解 | 小改 / 新增开关，改动局部 | 手工 port，偶尔解 rebase 冲突 |
| C 结构性 | 同一机制被两边按不同设计重写 | 二选一，且要接受长期 rebase 成本 |

## 总表

| # | 特性 | 级别 | 来源（fork commit） | 实测/文档收益 | 状态 |
|---|---|---|---|---|---|
| F2 | 自定义 AllReduce（PCIe 友好） | A/B | `04f89ab03` `5f65f9fa3` `41c46cedb` `92607b5d1` `19d784ad0` `743037eae` | 本机 A/B 在噪声内；fork 文档 +5%（decode） | **已移植** `75aba2e4e` |
| F3 | gfx906 内核（MMQ/Q8 repack/gated-delta chunk） | A/B | `2335544ab` `fcb857560` `55b275262` `1248e5e37` `b8baef09f` | 文档 +6~8%（decode） | 未开始 |
| F1 | 多级 TP `-tps`（TP×PP） | C | `04f89ab03` + 后续 | **本机实测 +8.5% decode**（见 baselines） | 未开始 |
| F4 | meta/调度流水线修复（依附 F1） | C | `c7069d868` `7f117dcd2` `27e980a21` `46b95d97e` `2d05207b6` `a355590d2` `be8ff98aa` `62d4be47d` `b8efb9510` | 文档 +37% layer prefill 等 | 依赖 F1 |
| F5 | PLE 分片 `LLAMA_PLE_SHARD` | C | `51b31fc47` | 本机 8 卡 **±2%（不采纳）** | 不采纳 |
| F6 | loader `-lm dio` / `-lzm` 策略 | B | 多提交 | 未量化 | 暂不 |
| F7 | MTP / dspark 投机解码优化 | C | 多提交 | 未量化 | 暂不 |
| F8 | DeepSeek-V4 专项（topk-moe / hc 融合 / engram） | C | `75db26192` `5542318e7` `5c8b24021` 等 | 与本模型无关 | 不采纳 |

推荐次序：**F2 → F3 → F1(+F4)**。先把"独立新文件"和接线跑通，再啃 meta 设备的重写。

---

## F2 自定义 AllReduce

- **做什么**：在 `-sm tensor` 的集合通信上，除上游的 RCCL/NCCL 与 internal/butterfly 之外，加一套
  peer-write 广播 + two-shot reduce-scatter/allgather 实现；F32 on the wire；面向 PCIe（无 xGMI）拓扑。
- **开关**：默认开（`41c46cedb`）；`GGML_ENABLE_CUSTOM_AR=0` 关；`GGML_TP_AR_TWOSHOT=0/1` 强制模式；
  `GGML_TP_AR_TWOSHOT_MIN_NE` 设切 two-shot 的元素数阈值；快路径需要 `HSA_FORCE_FINE_GRAIN_PCIE=1`。
- **文件**：新增 `ggml/src/ggml-cuda/tp-allreduce.cu`、`tp-allreduce.cuh`；接线在 `ggml/src/ggml-cuda/ggml-cuda.cu`
  （fork 1496 行 / 上游同期 496 行 —— 这是 B 级里最需要注意的地方）。
- **为什么值得做**：这是唯一"纯新文件"的高价值项，且本机是纯 PCIe（RCCL ring 在这种拓扑上并不优）。
- **移植要点**：先只落新文件 + 选择链挂点，保持 `-tps 0`（= 官方 `-sm tensor`）路径，跑通并 A/B 出数字后再做 F1。
- **验证**：`-sm tensor` 下 token 一致性（求和顺序改变）+ `-p 512,2048 -n 128` 对比。

### 移植结果（2026-10-12，`75aba2e4e`）

- `tp-allreduce.cu/.cuh` **逐字节照搬** fork（便于以后三方比对更新）；`.cu` 由现有的 `file(GLOB "*.cu")` 自动纳入构建。
- `ggml-cuda.cu` **+147/−1**，只有三处接线：comm_context 加 `ar_plan`/`custom_ar`/`use_custom_ar`（析构里 destroy）、`comm_allreduce_custom{,_prepare}` 插在 `try_allreduce` 指针**之前**（按 F32 / 尺寸门限 / peer-write 可用性自行拒绝）、`comm_init` 里按 `GGML_ENABLE_CUSTOM_AR`（默认开）初始化。
- **刻意排除**（属 F1/F4）：fork 的 `xfer_staging_*` 暂存环、`staging_depth`、`set_pipeline_stages`、以及给 meta 后端用的 fused-dispatch 入口（`allreduce_prepare`/`launch_rank`）。因此本 patch **不依赖 `-tps`**，对 `-tps 0`（= 官方 `-sm tensor`）的调用方完全透明。
- **实测（8×MI50，`-sm tensor`，`-lm dio`，`-p 512,2048 -n 128 -r 3`）**：
  - `HSA_FORCE_FINE_GRAIN_PCIE=1` → 日志 `path = broadcast F32 + twoshot F32 (peer-write, size-adaptive, lossless)` + `auto-selected two-shot for 8-rank decode-sized tensors`
  - 不设该变量 → `path = one-shot`，且因 RCCL 可用，所有集合通信用回 RCCL（**设计如此**）
  - A/B（开/`GGML_ENABLE_CUSTOM_AR=0`）：pp512 543.5/539.4、pp2048 518.7/519.8、tg128 33.80/32.44 —— tg 上 +4.2% 但落在 ±3.1 噪声内，**TP8 + RCCL 场景未见显著收益**
- **结论/下一步**：该 patch 的价值主要在 (a) 作为 F1 `-tps` 的前置（fork 的 +5% 是在多级 TP 下测的），(b) 提供一条**不依赖 RCCL/`--ipc=host`** 的通路（强制 one-shot：`GGML_TP_AR_ONESHOT=1`，可用于 RCCL 起不来或 shm 受限的环境）。要判定 tg 上那约 4% 是否真实，需把 `-r` 提到 8 或改测 `-n 256`。
  - 相关开关：`GGML_ENABLE_CUSTOM_AR=0` 关、`GGML_TP_AR_NO_GATE=1` 关尺寸门限、`GGML_TP_AR_ONESHOT=1` 强制 one-shot、`GGML_TP_AR_TWOSHOT=0/1` 强制模式、`GGML_TP_AR_TWOSHOT_MIN_NE` 改阈值、`GGML_TP_AR_KTIMING` 打印计时。
- **token 一致性验收：通过** ✅（脚本 `scripts/f2-custom-ar-consistency.sh`）。
  贪心解码（`--temp 0 --seed 1 -st`）32 token，比对**生成文本**：
  `-sm layer`(PP8) ≡ `-sm tensor` + 自定义 AR ≡ `-sm tensor` + `GGML_ENABLE_CUSTOM_AR=0`(RCCL)，
  三者逐字节一致（生成：`We need respond to user: "alpha bravo charlie delta echo foxtrot". Likely they are
  spelling NATO phonetic alphabet first six letters. Need maybe continue`）。
  > 方法上的坑：第一次直接比对 stdout 的 md5 会“四组全不一致”——因为 stdout 里含性能行
  > `[ Prompt: x t/s | Generation: y t/s ]`，那里每次必然不同；剔除该行后完全一致。

## F3 gfx906 内核

- **内容**：
  - `2335544ab` MMQ gfx906 配置（Q8_0 8 warps + wide tiles）→ 新增 `ggml/src/ggml-cuda/mmq-config-gfx906.cuh`
  - `fcb857560` Q8 repack 体系 → 新增 `ggml/src/ggml-cuda/q8_repack/*`
  - `55b275262` + `1248e5e37` chunked gated-delta-net prefill 内核 → 新增 `gated_delta_net_chunk.cu/.cuh`
    （**与本模型的 SSM/gated-delta 路径直接相关**）
  - `b8baef09f` K-quant / IQ4_NL MoE up 融合 + gfx906 mat-vec 门控（文档：+6~8% decode）
- **文件**：以新文件为主；另有 `ggml/src/ggml-cuda/mmq.cu`（274/176）、`ggml-cuda/common.cuh`（135/108）。
- **注意**：fork 整体的 **layer 模式 decode 反而比上游慢 8.8%**（31.31 vs 34.33 t/s），
  说明这些内核**不是无条件收益**，必须逐个 A/B 单独量化，不能整包照搬。
- **验证**：同一模型、同一 `-lm` 口径，逐内核开关对比；关注 qwen4exp 的 gated-delta 分块 prefill。

## F1 多级 TP `-tps`（TP×PP）

- **语义**（fork `common/arg.cpp`）：`-sm tensor -tps T` → 每 T 张卡组成一个 TP 组，其余卡组成 pipeline stage；
  `T=0`（默认）等价官方"单 TP 组"。要求 `n_gpus % T == 0`。`-tpsd` 单独给 draft 模型设宽度。
  `llama_model_params.tensor_parallel_size`（`include/llama.h:324`）。
- **实现方式**：把 `tps` 做进 meta 设备本身——`ggml_backend_meta_device(devs, n_devs, tps, ...)`，
  split-state 里多一个 `n_stages`，跨 stage 传递走 staging ring。
- **文件（冲突最集中处）**：
  | 文件 | fork | 上游同期 |
  |---|---:|---:|
  | `ggml/src/ggml-backend-meta.cpp` | 3182 | 105 |
  | `src/llama-context.cpp` | 714 | 553 |
  | `ggml/src/ggml-backend.cpp` | 682 | 446 |
  | `src/llama-model.cpp` | 660 | 796 |
  | `ggml/src/ggml-alloc.c` | 381 | 140 |
  | `src/llama.cpp` | 94 | **4** |
  | `ggml/include/ggml-backend.h` | （API 增参） | 小改 |
- **收益（本机 8×MI50 实测，`-lm dio`）**：`tg128` 37.25（`-tps 4`）/ 36.60（`-tps 2`）对官方最好配置 34.33 → **+8.5%**；
  `pp4096` 1204（`-tps 2`）对上游 1095。
- **风险**：改了 `ggml_backend_meta_device()` 的**公开签名**，属 API 级分歧；上游把 meta 设备列为待重构对象
  （相关 TODO 与调度器 PR 仍在推进），且官方 issue #23568（hybrid TP+PP）已 **closed as not planned**
  → 这套注定是**永久下游补丁**，要按"每次上游动 meta/allreduce 都要跟"来预算维护成本。
- **验证**：token 一致性 + 4/8 卡各跑一轮；`tps=0` 必须与官方 `-sm tensor` 数值一致（回归护栏）。

## F4 meta / 调度 流水线修复

- 依附 F1，只有 `-tps` 落地后才有意义：
  - `c7069d868`（输入环按 pipeline 宽度，文档 +37% layer prefill）、`7f117dcd2`（staging ring 按 pipeline 深度）
  - `27e980a21`（stage 边界不进 layer cache 生命周期）、`46b95d97e`（写型 graph 放到拥有该 buffer 的 stage）
  - `2d05207b6`（跨 stage seam 的 token graph）、`a355590d2`、`be8ff98aa`、`62d4be47d`、`b8efb9510`
- **两个未合并分支**（都停在旧基线，落后 master 672 提交）：
  - `perf/meta-layer-seam-transfers` @ `50da967f3`（2026-08-14）
  - `fix/tp-allreduce-system-barrier` @ `5a18dbe5c`（2026-08-15，仅 14 行 `tp-allreduce.cu`：全局排序 TP all-reduce 完成）
    → 需要时按 diff 择优 cherry-pick，不要整支拉。

## 不采纳项（记录原因，避免以后重复评估）

- **F5 `LLAMA_PLE_SHARD`**（`51b31fc47`，2026-08-29）：把 27465 MiB 的 `per_layer_token_embd.weight`
  从 host demand-paged 改成按整 hash head 分片进各卡显存，各卡 gather 自己拥有的 head、非自己的行清零，
  再由 AllReduce 求和。**本机 8 卡热缓存实测 ±2%**（特性确实生效，日志 `sharding the PLE gather table across the TP group`
  出现过）；文档的 4×MI50 冷缓存收益（gen 18→37 t/s、prefill 224→500）在本机复现不了。
  采纳它的代价是 split state 改动 + 放宽一个 GET_ROWS 支持性检查。**若将来要在 4 卡/冷缓存跑，再回来取。**
  兄弟开关：`LLAMA_PLE_PREFAULT=1`（加载时预 touch 表页）。
- **F6 loader 策略**：上游已有 `--load-mode`（含 dio）与 `TENSOR_READ_LAZY`；fork 的 `-lm dio`/`-lzm` 面广而收益未量化。
  注意：**跑 fork 时必须 `-lm dio`**（其文档称 mmap 在该栈上会 hang）——这是使用素材库时的操作约束，不是要移植的内容。
- **F7 MTP / dspark**：`common/speculative.cpp`（751/506）等面广，收益未量化，且与 `-tpsd` 的 pipeline 交互复杂。
- **F8 DeepSeek-V4 专项**：`src/models/deepseek4.cpp`、`deepseek41.cpp`、`llama-kv-cache-dsv4.cpp`、topk-moe 等，与本模型无关。

## 取材方法

```bash
# 看某个文件在 fork 里的最终形态
git show mxxm/master:<path>

# 某个新文件是被哪个提交引入的
git log --oneline --diff-filter=A mxxm/master -- <path>

# 对照 fork 自分叉点以来的全部改动
git diff 0f3a71be1 mxxm/master -- <path>

# 在本机直接读代码（detached worktree）
ls ../mxxm-master/<path>
```
