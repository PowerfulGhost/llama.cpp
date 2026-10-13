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
| F3 | gfx906 内核（MMQ/Q8 repack/gated-delta chunk） | A/B | `2335544ab` `fcb857560` `55b275262` `1248e5e37` `b8baef09f` | MMQ 配置**已被上游吸收**；GDN chunk 本模型 0 收益 | **不采纳**（见下） |
| F1 | 多级 TP `-tps`（TP×PP） | C | `04f89ab03` + 后续 | **本机实测 +8.5% decode**（见 baselines） | 未开始 |
| F4 | meta/调度流水线修复（依附 F1） | C | `c7069d868` `7f117dcd2` `27e980a21` `46b95d97e` `2d05207b6` `a355590d2` `be8ff98aa` `62d4be47d` `b8efb9510` | 文档 +37% layer prefill 等 | 依赖 F1 |
| F5 | PLE 分片 `LLAMA_PLE_SHARD` | C | `51b31fc47` | 本机 8 卡 **±2%（不采纳）** | 不采纳 |
| F6 | loader `-lm dio` / `-lzm` 策略 | B | 多提交 | 未量化 | 暂不 |
| F7 | MTP / dspark 投机解码优化 | C | 多提交 | 未量化 | 暂不 |
| F8 | DeepSeek-V4 专项（topk-moe / hc 融合 / engram） | C | `75db26192` `5542318e7` `5c8b24021` 等 | 与本模型无关 | 不采纳 |
| F9 | gfx906 权重重排（Q4_K/Q5_1/Q8_0） | A/B | `2e9d29fe7` `fcb857560` `b9a990b28` `b8baef09f` `481f684d3` | **本机 +34% prefill**（pp512 337→451），decode 持平 | **已移植** `9706f6734` |

推荐次序：**F2（已落）→ F9（已落）→ F3/F3c（已否决）→ F1(+F4)**。
F9 一次性拿到了 prefill 的 +34%，也顺带把配置结论改成 **layer 模式（PP8）优于 TP8**（见 F9 移植结果）。
F1 只在需要「TP 的短上下文 prefill 优势」时才值得啃（它是最大的重写，且是永久下游补丁）。

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

### 移植结论（2026-10-12）：两项都不采纳

**F3a MMQ gfx906 配置 —— 已被上游吸收（无需移植）。**
上游把 VEGA20 归入 GCN（`GGML_CUDA_CC_IS_GCN`：`cc > OFFSET_AMD && cc < CDNA1`，906 < 908），
所以 gfx906 走 `mmq-config-gcn.cuh`，而它**已经有** Q8_0 / `nthreads=512` / `J` 最高 128 的条目
（`CASE(GGML_TYPE_Q8_0, 512, 2, 128, 128, ...)`）——正是 fork 新增 `mmq-config-gfx906.cuh` 想做的那件事。
fork 剩下的只有 J 扫描里的 VEGA20 占用门控（`J > 64` 时要求 tile 总数 ≥ nsm），
而 gfx906 上只有 Q8_0 会提供 `J > 64`，**本模型（UD-Q4_K_XL）无法验证** → 不采纳。

**F3b chunked gated-delta-net prefill —— 已移植、已验证生效、但回退。**
- 移植：`gated_delta_net_chunk.{cu,cuh}`（366 行）+ `gated_delta_net.cu` 34 行 hook；编译通过。
- **确认生效**（临时插桩，需 `-v` 才可见）：`launch_gated_delta_net: DBG n_tokens=512 S_v=128 KDA=0 keep_rs=0 eligible=1`。
- **A/B（`-sm layer`，`-lm dio`，`-p 512,2048,4096 -n 128 -r 3`）**：
  | | pp512 | pp2048 | pp4096 | tg128 |
  |---|---:|---:|---:|---:|
  | 有 chunk | 340.50 ± 1.57 | 854.04 ± 6.76 | 1096.56 ± 1.97 | 34.38 ± 0.06 |
  | 无 chunk | 339.23 ± 1.98 | 852.89 ± 7.00 | 1094.90 ± 1.96 | 34.46 ± 0.04 |
  → 全部在噪声内（≤ 0.4%）。**本模型的 prefill 不是被 GDN 串行扫描主导**（48 层里 MoE 512 experts + 27 GiB host PLE gather 才是）。
- **token 一致性：不一致**。200-token prompt、贪心 32 token：
  chunk = `...repeated NATO alphabet sequence...`，serial = `...repeated NATO phonetic alphabet sequence...`。
  分块改变浮点求和顺序，不是 bug，但**会破坏输出可复现性**（对投机解码接受率/对比测量是麻烦）。
- 结论：0 收益 + 非 bit-exact → 回退（代码已从 `gfx906-opt` 中移除）。将来若跑 delta-net 为主的模型（如 Qwen3Next）再取。
- **长上下文补测（2026-10-12）**：`-p 16384,65536 -n 128 -r 2`、`-sm layer`、`-lm dio`：

  | test | chunked | 串行 | 差值 |
  |---|---:|---:|---:|
  | pp16384 | 1282.71 ± 33.07 | 1339.60 ± 0.92 | **−4.2%** |
  | pp65536 | 1105.29 ± 4.09 | 1150.01 ± 2.29 | **−3.9%** |
  | tg128 | 33.79 ± 0.49 | 34.45 ± 0.24 | −1.9% |

  → 长上下文下 chunked **反而更慢**，且方差很小（±0.9%~±2.3%），−4% 是真实的。
  原因：chunk 核每个 (head, seq) 只起一个 CTA，并行度固定为 H×z，token 越多每 CTA 的串行 chunk 循环越长；
  而串行核沿 token 展开，CTA 更多。这也解释了 fork 自己 pp4096=370 的异常。
  → 结论由“0 收益”升级为“短上下文持平、长上下文 **−4%** 且非 bit-exact”，**回退**无疑。
- 顺带得到两个新的长上下文基线（上游+F2、`-sm layer`、`-lm dio`）：**pp16384 1339.60**、**pp65536 1150.01 t/s**。

### F3b 的 kernel 级隔离测量（`test-backend-ops perf`）

端到端之外，kernel 本身的差距可以用仓库自带的 op 级 harness 直接量：
`test-backend-ops perf -o GATED_DELTA_NET`（用例在 `tests/test-backend-ops.cpp:4899`）。
同 harness、同 shape、只换 kernel；每 shape 取中位数（ROCm0）。为找交叉点，临时在用例列表里加了一组
`H × tok × n_seqs` 探针（测完已还原，未入库）：

| H | S_v | tok | n_seqs | 串行 µs | chunked µs | chunk/串行 | 谁赢 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 128 | 128 | 1 | 158.3 | 124.4 | 0.79x | chunk 快 1.27x |
| 4 | 128 | 512 | 1 | 853.7 | 571.4 | 0.67x | chunk 快 1.49x |
| 4 | 128 | 1024 | 1 | 1875.9 | 1148.6 | 0.61x | chunk 快 1.63x |
| 4 | 128 | 512 | 4 | 951.8 | 811.4 | 0.85x | chunk 快 1.17x |
| 8 | 128 | 512 | 1 | 950.0 | 574.3 | 0.60x | chunk 快 1.65x |
| 8 | 128 | 512 | 4 | 1062.4 | 1462.7 | 1.38x | 串行 快 1.38x |
| 16 | 128 | 512 | 1 | 1017.3 | 813.2 | 0.80x | chunk 快 1.25x |
| 16 | 128 | 512 | 4 | 2003.2 | 3041.7 | 1.52x | 串行 快 1.52x |
| 32 | 128 | 256 | 1 | 538.7 | 728.2 | 1.35x | 串行 快 1.35x |
| 32 | 128 | 512 | 1 | 1053.0 | 1469.9 | 1.40x | 串行 快 1.40x |
| 32 | 128 | 1024 | 1 | 2092.6 | 3002.9 | 1.44x | 串行 快 1.44x |
| 32 | 128 | 512 | 4 | 3537.1 | 6998.1 | 1.98x | 串行 快 1.98x |
| 32 | 128 | 64 | 1 / kda=1 tok=64 | 161.4 / 277.5 | 161.5 / 296.3 | 1.00x / 1.07x | 不满足 / 串行快 1.07x |

**对照组**：`n_tokens < 2·CS`（kda=0 时 <128，kda=1 时 <32）的 shape 根本不会走 chunk，
两侧逐位相同（1.00x）——证明这张表是干净的同 harness 对照。

**结论：什么时候 chunk 更好？**
1. **硬条件**：`S_v ≤ 128` 且 `n_tokens ≥ 2·CS`（kda=0 需 ≥128 token，kda=1 需 ≥32）。不满足则根本不用它，也就无所谓好坏。
2. **value head 数 H 是主因**：H ≤ 16 时 chunk 赢（串行核 CTA 不足、每 token 还要把 state 写回 HBM）；
   H = 32 时串行赢。H=4/8、n_seqs=1、tok≥512 时 chunk 可以快到 **1.5~1.65x**。
3. **n_seqs（并发序列数）是第二因子且与 H 交互**：n_seqs=4 时即使 H=8 也变成串行赢（1.38x），
   H=32/n_seqs=4 时 chunk 慢到 **2x**——串行核的 grid 随 n_seqs 扩张得更有效。
4. **token 越长收益越大**（H=4：tok 128→1024 时 1.27x→1.63x）。

本模型 48 层里 36 层带 GDN，长上下文 E2E 是 −4%（即落在 H 大的一侧；按 op 级约 1.4x 反推 GDN op 约占 prefill 10%），
且 server 场景 n_seqs>1 会更糟 → **不采纳**。这个 kernel 面向的是 head 数少、单序列、长 prompt 的模型
（上游那条 PR 的目标场景）。

> 测量卫生（踩过一次坑）：`build-f3b` 曾被误用另一棵树重新构建，导致 op 级对比出现“全部 1.00x”的假结果。
> 每次重建后必须验证 chunk 符号存在（`strings libggml-hip.so | grep -c launch_gated_delta_net_chunk`）
> 或抽查一个已知有差异的 shape（H=32/tok=512 应为 ~1480 µs 而不是 ~1053 µs）。

### 新线索（待评估）：BF16 compute on gfx906（F3c）

fork `FEATURES.md` 有独立一节：*“On AMD parts predating CDNA and RDNA3, BF16 matmuls compute in F32 …
F32 is both faster and more faithful. Worth **+18-19% prefill on UD / `*_XL` quants that keep BF16 tensors**.
Automatic, no flag. `GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16` selects the old compute type.”*
我们的模型正是 **UD-Q4_K_XL**（UD 量化会保留部分 BF16 张量），且我们自己测到 fork 的 layer 模式 prefill 比上游快 30%+（450 vs 339 @pp512）。
→ 这是目前性价比最高的下一个候选。

## F9 gfx906 权重重排（weight repack）

- **做什么**：Q8_0/IQ4_NL/Q4_K/Q5_K/Q5_1（以及 MXFP4）的权重在**上传时**重排成「quants 与 scales 分平面、行去别名」的布局，
  gfx906 的 MMQ / mat-vec 内核直接读，省掉 per-block scale gather。走的是上游本来就有的 extra-buffer-type 机制
  （`--no-repack` / llama-bench `-nr 1` 可关）。
- **上游状态**：**只有 CPU 侧**（`ggml_backend_cpu_get_extra_buffer_types()` + `make_cpu_buft_list()`，`src/llama-model.cpp:1119`）；
  GPU 侧没有 → 需要给 `make_gpu_buft_list()` 加 `use_extra_bufts`（fork 已做，`src/llama-model.cpp:1524`）。
- **实测**（fork `eefc4e732`，8×MI50，`-sm layer`，`-lm dio`，`-r 3`，**同 build 内 `-nr 1` 对照**）：见 baselines §3.3 →
  **+23% prefill / +4% decode**，是所有候选里最大的一项；关掉 repack 后 fork 只剩 +3%（≈上游水平）。
- **我们模型的类型分布**：Q4_K 44.6%、IQ4_NL 28.9%（= PLE 表那 1 个张量）、Q5_1 20.4%、Q8_0 5.1%、Q5_K 0.9%
  —— 全在 fork 的支持集合内。
- **移植面**：新目录 `ggml/src/ggml-cuda/q8_repack/`（4516 行，**无冲突**）+ 钩子（`ggml-cuda.cu`、`common.cuh`、
  `CMakeLists.txt`、`src/llama-model.cpp` 的 GPU extra-buft、`ggml-backend-meta.cpp` 的 TP 切片 repack）。
  内核本身是"纯新文件"，钩子才落在上游重构热点上（`llama-model.cpp` 796/660、`ggml-cuda.cu` 496/1496）。
- **验收优势**：对照组在**同一个 build 内**（`-nr 1`），不依赖两棵树/两份构建的一致性 —— 正好避开 F3b 踩过的构建污染坑。
- **增量计划**：① 骨架（CUDA extra-buft 接线 + 上传/重排 infra）+ **Q4_K**（44.6% 的字节）→ ② **Q5_1**（20.4%；fork 文档对
  "43 个 Q5_1 张量的 Qwen3.8-Flash-Next MoE" 报 +11% prefill，就是本模型）→ ③ Q8_0（503 个张量）→ ④ IQ4_NL / Q5_K →
  ⑤（可选）MoE up/gate 融合（实测只值 ~2% decode，优先级最低）。每步都用 `-nr 1` 出数字。
- **数值验收**：K-quant 重排在 fork 文档里是 "byte-identical boots"、PPL 不变；本机实测 greedy 32 token 生成
  **逐字节相同**（ON×2 / OFF 三组一致），比预期更干净。

### 移植结果（2026-10-12，`9706f6734`）

- **照搬**：`ggml/src/ggml-cuda/q8_repack/` 8 个文件（4516 行）字节级来自 fork，只改了三处依赖 fork-only 基础设施的地方，
  且在原地写了注释说明：① `repack_quantize_src1_q8_1()` 里 **q8_1 激活缓存**（那是另一个独立特性）未移植，改为每次量化进自己的
  fallback buffer；② 显存压力下的**列分块 + `try_alloc` 重试**未移植，改为一次性全量分配（prefill 宽度下只有几 MiB）；
  ③ `ggml_backend_buffer_type_i` 的 vtable 按**上游新增的 `alloc_buffer_n`/`get_alloc_size_n` 槽位**补齐（7 参数顺序）。
- **接线**（上游只有 CPU 侧有这套机制）：`ggml-cuda.cu` 的 include / `device_supports_buft` / `device_supports_op`（repacked 权重只能做
  `MUL_MAT`/`MUL_MAT_ID` 的 src0 且 src1/dst 为 F32）/ `get_extra_bufts` proc / `mul_mat(_id)` 分发 / 屏蔽 canonical fused MMV；
  `src/llama-model.cpp` 的 `make_gpu_buft_list(..., use_extra_bufts)`（extra buft 放在默认 buft **之前**，`--repack` 可关）；
  两个 `CMakeLists.txt` 加 `q8_repack/*.cu|cuh` 的 glob。
- **类型门**：`ggml_cuda_repack_tensor_supported()` 只放行 **Q4_K / Q5_1 / Q8_0** —— 这是实测出来的组合：
  单独加 Q4_K +2%、再加 Q5_1 +9%、再加 Q8_0 **+34%**；本模型剩下的 IQ4_NL（PLE 表，走 GET_ROWS 本来就不会被 repack）与 Q5_K（0.9%）
  放开后反而 **−2%**（两次复现、回退后复现原值），MXFP4/Q6_K 本模型没有、无法验证，故一并排除。要放开是一行改动。
- **实测**（8×MI50，`-sm layer`，`-lm dio`，`-r 3`，同 build 内 `--repack 0` 对照）：

  | 配置 | pp512 | pp2048 | pp4096 | tg128 |
  |---|---|---|---|---|
  | repack off | 337.2 | 850.3 | 1092.1 | 34.50 |
  | repack on（Q4_K+Q5_1+Q8_0） | **451.3** | **1161.6** | **1468.4** | 35.26 |

- **配置结论变了**：repack 只在 layer 模式下生效（`-sm tensor` 下 ON/OFF 完全相同，因为 meta 设备没接这套），
  而 repack 后的 PP8 在 pp2048/4096/decode 上全面超过 TP8（1162/1468/35.3 对 519/514/34.1）→ **目标配置从 TP8 改为 PP8**。
### MoE / 稠密 up+gate 融合 mat-vec（`9706f6734` 之后追加）

- **做了什么**：把 repack 的融合 mat-vec 接进上游 `ggml_cuda_try_fuse` 的**裸** `{MM(_ID), MM(_ID), GLU}` 分支
  （`ggml-cuda.cu`：一个 `ggml_cuda_repack_fusion_glu_ok()` 判别 + 一段分发，约 35 行）。该分支无 bias、无 scale，
  正好是融合核支持的形状；**带 bias/scale 的 GLU 分支保持 canonical**，不替换（那些形状融合核没有实现，替换会算错）。
- **触发条件**：up/gate 两个权重都在 repack 布局、类型被 `ggml_cuda_repack_mmv{,_id}_fusion_supported()` 放行
  （MoE 侧本模型专家是 Q4_K ✓；稠密侧只放 Q8_0/MXFP4 —— 本模型注意力投影是 Q8_0 ✓）、宽度在
  `ggml_cuda_repack_mmv_fusion_width_ok()` 内（稠密 1 token，MoE ≤8 token）。
- **实测**（8×MI50，`-sm layer`，`-lm dio`，repack 全开，r=3；对照 `GGML_CUDA_REPACK_KQUANT_MOE_FUSION=0`）：

  | | tg128 | pp512 | pp2048 | pp4096 |
  |---|---|---|---|---|
  | 融合 off | 35.26 | 450.7 | 1160.7 | 1467.2 |
  | 融合 on | **35.87 / 36.30** | 451.6 | 1161.7 | 1467.7 |

  prefill 不变（融合只在窄 batch 存在）。witness：`llama-bench -v` 下 `[repack-moe-fusion] type=q4_K cols=1 fused up/gate mat-vec engaged`。
- **数值代价（重要）**：width-1（= 普通 decode）下融合核的**归约顺序**与 canonical 不同 → greedy 输出会分叉
  （前约 150 字符一致，然后某个 token 翻转、之后文本完全不同）。batch=1 的 PPL（8×512 token）：**4.2282（融合）vs 4.2120（不融合），+0.38%**，
  误差棒 ±0.23；batch=512 的 PPL **完全相同 3.7710**（那条路径不触发融合）。
  与 fork 文档 "differ at width 1 by reduction order" 一致，属固有性质而非算错。
- **开关：默认关**（`GGML_CUDA_REPACK_MMV_FUSION`，未设=0）。理由：本项目对数值路径的默认要求是逐位一致，而这一项在 width 1 不满足；
  需要 decode 那 2~3%（实测 34.97 → 36.56，+4.5%）时显式开。专家侧另有细开关 `GGML_CUDA_REPACK_KQUANT_MOE_FUSION`（在融合已开的前提下，默认 1）。

- **仍未移植**：① `ggml-backend-meta.cpp` 的 TP 接线（~188 行，TP 下 repack 才生效）；② Q8_1 激活复用（独立特性，fork 称 +2.2~2.6%）；
  ③ 显存压力列分块与 `try_alloc` 重试；④ 稠密 bias-only（`{MM, ADD}`）融合。

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

# 在本机直接读代码（fork 快照的源码已归档，用 tag 取）
git show archive/mxxm-fork-base:<path>            # mxxm fork
# furnace：相关文件可直接读源码树
ls ../forks/furnace/ggml/src/ggml-cuda/<file>
```
