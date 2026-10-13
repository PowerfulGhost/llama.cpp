# gfx906 实验记录（2026-10-13 会话，含对 10-12 会话 E1 的更正）

> **目标声明（用户明确）**：目标是**改造推理引擎**，让 `Qwen3.8-Flash-Next`(qwen4exp) 在本机
> （8×MI50 gfx906，纯 PCIe，ROCm 7.14）上跑出好效率；**不是"在现有引擎上挑一套好参数"**。
> **有决心改代码**，必要时换引擎（如 vLLM）并移植 kernel。
> 因此本页的实验都以"**该改哪条执行路径 / 哪个 kernel**"为导向。

环境与基线见 `baselines.md`；分支/同步流程见 `README.md`；特性清单见 `features.md`。
模型：`UD-Q4_K_XL`（103.68 GiB）；48 层中 36 层是 SSM/GDN，12 层带 attention；MoE 512 专家、每 token 激活 10 个。
关键张量形状（GGUF 实测，ggml 约定 ne[0]=输入维、ne[1]=输出维）：
`attn_qkv`[2560×10240] Q8_0、`attn_gate`[2560×6144] Q8_0、`ssm_out`[6144×2560] Q8_0、
`ffn_gate_exps`/`ffn_up_exps`[2560×640×512] **Q4_K**、`ffn_down_exps`[640×2560×512] **Q5_1**、
`ffn_*_shexp`[2560×640] Q8_0、`ffn_gate_inp`[2560×512] F32、`token_embd`/`output`[2560×248320] Q8_0。

---

## 0. 方法学更正：llama-bench **不能**做批量 decode（E1 作废）

10-12 会话的 E1 用 `llama-bench -b {1..32}` 扫描"decode 宽度"，得到"吞吐对宽度是平的（35 t/s）"，
并据此推断"批处理无收益、投机解码必亏"。**这个结论作废**，原因在源码里：

```cpp
// tools/llama-bench/llama-bench.cpp : test_gen()
for (int i = 0; i < n_gen; i++) {
    batch.clear();
    batch.add(token, pos++, 0, true);        // ← 永远一次一个 token
    llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
```

`test_gen` 每次只喂 1 个 token，`-b/-ub` 只影响 **prefill**（`test_prompt`）。所以 E1 的 6 行数据其实是
同一个 1-token decode 用 6 种 prefill 批量测出来的。
**要测 decode 宽度，只能用 `llama-server -np N` + 并发请求**（见 E5）。

---

## E2. 融合核在宽度 ≥2 时逐位一致（llama-perplexity，有效）

命令：`llama-perplexity -f corpus -c 512 --chunks 2 -b B`，对照 `GGML_CUDA_REPACK_MMV_FUSION`。

| 宽度 | 融合 ON | 融合 OFF | 判定 |
|---:|---|---|---|
| 1 | 3.5869 ±0.36028 | 3.6241 ±0.36672 | 不同（+1.0%） |
| 2 | 3.5886 ±0.36072 | **3.5886 ±0.36072** | **逐位一致** ✅ |
| 4 | 3.5793 ±0.35953 | **3.5793 ±0.35953** | **逐位一致** ✅ |

**结论**：宽度 ≥2 时连误差棒都相同 ⇒ 每个 chunk 的 loss 相同 ⇒ forward 逐位一致；只有宽度 1 因归约顺序不同而分叉。
⇒ **多流并发（槽位≥2）或投机验证（宽度≥2）时融合零数值代价**，值 +3.5~4%（E5 表里的"融合 ON"列就是它）。

---

## E3. MoE 专家 GEMM 的形状效率（`test-backend-ops perf -o MUL_MAT_ID`）

本模型真实形状：512 专家、每 token 10 激活、Q4_K gate/up（k=2560）、Q5_1 down（k=640）。
"完整宽度"= layer 模式（我们现在跑的），"1/8 切片"= TP8 下每卡的形状。单卡、GPU 空闲。

| 算子 | ntok | 完整宽度（m=640 / 2560） | 1/8 切片（m=80 / 320） | 切片 wall 比 |
|---|---:|---|---|---:|
| Q4_K gate/up | 1 | 49.8 µs / 0.658 TF | 195.8 µs / 0.021 TF | **3.9× 慢** |
| Q4_K gate/up | 8 | 225.6 µs / 1.160 TF | 226.4 µs / 0.145 TF | 1.0× |
| Q4_K gate/up | 512 | 10653 µs / 1.570 TF | 4908 µs / 0.427 TF | 0.5× |
| Q4_K gate/up | 2048 | 11828 µs / **5.670 TF** | 5805 µs / 1.450 TF | 0.5× |
| Q5_1 down | 1 | 71.2 µs / 0.460 TF | 196.6 µs / 0.021 TF | 2.8× 慢 |
| Q5_1 down | 2048 | 11363 µs / **5.910 TF** | 2853 µs / 2.940 TF | 0.3× |

**结论**
1. **decode 宽度（ntok 小）下，TP 切片对专家 GEMM 是灾难**：1/8 的工作量却要 2.8~3.9× 的时间
   （per-FLOP 差 20~30 倍）。这从"专家 GEMM 形状"角度**支持 EP/PP（保持完整宽度），不支持 TP**——
   与 `features.md` 里"EP 通信是 TP 的 3.7 倍"合起来看：TP 在 decode 侧被两头夹击。
2. **prefill 宽度下切片反而 wall-clock 更快**（因为只做 1/8 的工作，1.45~2.94 TFLOPS 的绝对效率虽低，但每卡工作量也低）。
3. **最刺眼的一点：即使完整宽度，专家 GEMM 也只跑到 5.7~5.9 TFLOPS ≈ 单卡 int8 峰值（MI50 约 53 TOPS）的 11%**。
   这是 prefill 的主瓶颈（见 E4b）。

（探针已从 `tests/` 移除，副本：仓库外 `probe-e3-e4.cpp.txt`，共 15 行，可直接贴回 `make_test_cases_perf()` 的 GATED_DELTA_NET 段之后。）

---

## E4a. decode 的时间归因（rocprofv3 kernel trace，PP8）

做法：`rocprofv3 --kernel-trace` 采一次 `llama-bench -p 0 -n 8`，按"最大时间间隔"切出工作负载窗口，
按 kernel 名字归类（窗口内 36091 次 dispatch，GPU 忙时 307 ms）。
⚠️ 绝对毫秒含 profiler 开销与预热，只看**相对占比**。

| 类别 | 占比 |
|---|---:|
| 稠密 Q8_0 mat-vec（`mul_mat_vec_q8_0_repacked`，即 attention/SSM 的 Q8_0 投影） | **31.3%** |
| 其它（`dsv4_hc_pre/post`、`unary_gated_op`(silu)、`concat` …） | 12.8% |
| 逐元素（`scale_f32`/`k_bin_bcast`/sigmoid…） | 11.9% |
| 稠密 mat-vec f16/f32（`mul_mat_vec_f`） | 10.8% |
| 激活量化 `quantize_q8_1` | 8.0% |
| MoE mat-vec（Q4_K，`mul_mat_vec_rp<12,…>`） | 6.8% |
| `rms_norm_f32` | 4.0% |
| 拷贝 | 3.3% |
| MoE mat-vec（Q5_1） | 3.5% |
| MoE 路由 `topk_moe` | 2.9% |
| GET_ROWS（PLE 表 gather） | 2.6% |
| SSM/GDN | 1.9% |
| Flash-Attention | 0.9% |

**结论**：decode 是 **GPU 忙满**的（GPU 忙时 ≈ 墙钟时间），但时间花在**普通稠密 mat-vec（42%）+ 各种零碎逐元素/量化/归一化/拷贝（约 27%）**上，
**不是**花在这个模型的"异形结构"（PLE 2.6%、SSM/GDN 1.9%、FA 0.9%）。
⇒ decode 侧的引擎改造方向是"**减少 kernel 数量 + 提高稠密 mat-vec 效率**"（融合、更少的小 kernel），
而不是去优化 PLE/SSM。

## E4b. prefill 的归因（op 微基准重建）

用 E3 的单卡微基准，把 48 层 × 3 个专家 GEMM 加起来：
`48 × (11828 + 11828 + 11363) µs ≈ 1.68 s`（ntok=2048），外推到 4096 token ≈ 3.3 s。
实测 pp4096 一趟 = 2.79~3.03 s（1468~1351 t/s）。**两者同量级 ⇒ prefill 基本由 MoE 专家 GEMM 决定**，
且它只有 11% 的峰值效率（E3 结论 3）。

**结论**：prefill 的最大杠杆是**专家 GEMM 的绝对效率**（skinny per-expert GEMM：M=640、N≈40、K=2560）。
这是一条**明确的 kernel/算法级改造方向**（grouped-GEMM / 更好的分块与并行度、或者改数据结构让每专家 N 变大）。
次要方向：pipeline 串行化问题（layer 模式是否让 8 张卡串行工作，见"未解决"一节）。

---

## E5. 真并发压测（`llama-server -np 8` + N 路并发请求）

这是**唯一有效的 decode 宽度实验**（E1 作废后）。每请求 `n_predict=128, ignore_eos=true`，短 prompt（11 token）。
`-c 32768 -np 8`，PP8 与 TP8（无 repack，因为 TP 下 repack 未接线）各一轮。

| 并发 | PP8 总吞吐 | PP8 单请求 | TP8 总吞吐 | TP8 单请求 |
|---:|---:|---:|---:|---:|
| 1 | 32.6 | 32.6 | 27.4 | 27.4 |
| 2 | 47.5 (1.46×) | 23.8 | 36.6 (1.34×) | 18.3 |
| 4 | 54.9 (1.68×) | 13.7 | 49.3 (1.80×) | 12.3 |
| 8 | **83.1 (2.55×)** | 10.4 | 81.4 (2.97×) | 10.2 |

**结论**
1. **并发确实提升总吞吐**（PP8 32.6 → 83.1 t/s，2.55×），彻底推翻 E1 的"批处理无收益"。单请求延迟也随并发上升（3.9 s → 12.3 s 完成 8 个请求）。
2. **PP8 ≥ TP8 在所有并发度**（8 路时 83.1 vs 81.4，差 2%）⇒ 即使目标场景是多流并发，layer 模式仍是（至少并列）最优。
3. 亚线性（8 路只有 2.55×，理想 8×）⇒ decode 仍是访存主导，仍有优化空间；结合 E4a，"减少零碎 kernel"是主要方向。
4. 多流场景下**建议开融合**（E2 证明宽度 ≥2 逐位一致，值 +3.5~4%）。
5. 投机解码的账要重算：宽度 8 时每 token 成本 ≈ 12 ms，而宽度 1 是 39 ms ⇒ verify 步不是线性成本，
   **投机大概率是正收益**（之前基于 E1 的"净亏"推断随之作废）。

---

## E7. 引擎选型（**已修订**：gfx906 社区生态盘点，2026-10-13）

### E7-原始结论（作废）
只查官方 vLLM → "gfx906 不被支持"（issue #22458 closed not planned、PR #33811、MI50 自 ROCm 6.0 弃用）。
**该结论作废**：用户指出存在社区 fork（并可复现），实测 GitHub 搜索 `gfx906` 有 **93 个仓库**。

### 生态盘点（GitHub API 实测，2026-10-13）

| 仓库 | star | 状态 | 关键信息 |
|---|---:|---|---|
| `mixa3607/ML-gfx906` | 352 | 活跃 | llama.cpp/vLLM/PyTorch 的 gfx906 构建镜像 —— **我们用的 builder 镜像就来自这里** |
| `nlzy/vllm-gfx906` | 435 | **已归档**（2026-02） | 初代 gfx906 vLLM；需 ROCm 6.3 + `nlzy/triton-gfx906` v3.5.0；量化只推荐 GPTQ/AWQ(W4A16 INT)；**其 README 明确写 "All MoE quantization models are significantly slow"** |
| `nlzy/triton-gfx906` | 48 | 已归档（2025-12） | gfx906 版 Triton |
| **`sixvolts/llamacpp-gfx906-furnace`** | 17 | **活跃**（2026-10-11 仍在推） | **见下方专节：与本项目正面重合** |
| `sixvolts/reinstinct` | 17 | 活跃（2026-10-04） | 自研 HIP 推理引擎（MI50/MI60）；furnace 的 repack/DPP 技术"从这里的生产实现移植" |
| `ttdxq/gfx906-vllm` | 43 | **活跃**（2026-09-29） | nlzy fork 的继任者，持续跟上游；**已支持 Qwen3.5 MoE 架构的 GGUF（含 MTP/多模态）并在单卡 32 GiB gfx906 实测可用** |
| `0c33/mi50-vllm-lab` | 4 | 活跃（2026-10-11） | vLLM 自定义 kernel：2×MI50 上 **DFlash2 投机 73.7 t/s、MTP 65.1 t/s**；200k 上下文（int8 KV）；**TP=2 all-reduce 28 µs（RCCL 72 µs，逐位一致）**（≈我们的 F2 思路） |
| `nick413-bit/gfx906-fa-vllm` | 16 | 活跃 | vLLM 的 FlashAttention 风格自定义 attention backend |
| `Wulfsta/vllm-flake` | 18 | 活跃 | Nix 构建 gfx906 vLLM |
| `xxDoman/strata-gfx906` | 8 | 活跃（2026-10-12） | 第三方引擎 **Strata，明确支持 Qwen3.8-Flash-Next**；2 卡 IQ3_XXS + `--spec 4`：decode 41~44 t/s、4K prompt 473~547 t/s；带 `--mmap-experts` |
| `milpster/gfx906-llama-cpp`, `skyne98/wiki-gfx906`, `kyuz0/mi50-gfx906-toolboxes` … | — | — | 更多构建/资料 |

### **`sixvolts/llamacpp-gfx906-furnace` 与本项目正面重合（最重要发现）**

其 README 自述：为 Vega20 重调的 llama.cpp，"**加入 GCN 权重重排 GPU buffer type 与 DPP warp reduce**（从 `reinstinct` 生产实现移植），以及**在 4 张 MI50 上服务 177B 稀疏注意力 MoE（Qwen3.8-Flash-Next / Swift 1.5，`qwen4exp`）所需的 kernel、内存布局与服务端改动**"。
即：**同一个模型、同一类卡、同一个技术路线（权重重排）**，而且已经解决/覆盖了我们这次测出来的全部三个缺口：

| 我们测出的缺口 | furnace 的对应工作 | 他们的实测 |
|---|---|---|
| **N1** layer 模式的 8 卡被串行化（我们未证实） | `GGML_SCHED_SYNC_INPUTS`：图输入改为从 pinned staging 异步下发 | **pp8192 877 → 1978 t/s** |
| **N2** 专家 GEMM 只有 ~11% 峰值 | MoE 直接 in-kernel 路由（decode）+ **grouped 16-token-tile GEMM**（prefill）；`GGML_CUDA_NO_MOE_SMALL` 可关 | 见下表 |
| **N4** 投机解码的账 | 移植上游 PR #28243 的 NextN/MTP draft head + **adaptive MTP** `--spec-draft-max-slots` | 单用户 decode +17%~+40%（取决于文本），多用户下不亏 |
| repack（我们的 F9） | **three-plane K-quant repack**（Q4_K/Q5_K/Q6_K，含 MoE 专家堆叠），`GGML_CUDA_REPACK=0` 关；Q8_0/Q5_1 repack 为**可选**（`GGML_CUDA_REPACK_Q8_0=1`） | 声称 on-disk 布局把 GCN matvec 限制在 ~58% HBM 带宽，重排后 ~89%；单卡 Qwen3.6-35B-A3B UD-Q4_K_XL pp512 864→968 (+12%)、tg 持平 |
| qwen4exp 异形结构 | LDS-resident GDN prefill/decode、QSA indexer 池化/块打分融合、块布局缓存、**稀疏 decode attention**、并行 top-k radix select、hyper-connection/共享专家尾部融合 | pp4096@48K 深度 622→1150 t/s |
| 多卡 gfx906 | **`-DGGML_CUDA_NO_PEER_COPY=ON`**（PCIe peer copy 在该 HIP 运行时崩溃）+ 异步分阶段拷贝替代 | — |

**其 4×MI50 上的生产数据（同一模型 UD-Q4_K_XL）**：pp8192 1977 t/s、tg64 49.2 t/s、6 路并发 decode 104.6 t/s 聚合；
server 端 prefill 8K 1251 / 32K 1561 t/s，decode 1/2/6 用户 62/78/98 t/s，6 流混合 20 分钟完成 191 请求。

**对照我们（8×MI50，PP8+repack）**：pp4096 1468、tg128 35.3（单流）、8 槽并发 83.1 t/s。
**即：他们用 4 张卡就超过我们用 8 张卡的 prefill 与并发 decode。**

### 结论与规划影响

1. **"从 mxxm fork 逐个移植"的路线应当停止**——它在做同样的事、且更先进；继续走下去是重复造轮子。
2. **我们的测量结论被独立验证**：N1（串行化）、N2（MoE 效率）、N4（投机）全都被他们以真实优化解决，方向完全一致。
3. **我们的资产应重新定位**：① 可复现的测量与验收方法（E2/E4/E5 + `acceptance.sh`，未见他们有公开同类套件）；
   ② 具体发现（Q8_0 是 repack 最大贡献 +25%、融合在宽度≥2 逐位一致、专家 GEMM 仅 11% 峰值、EP/TP 通信账）；
   ③ 可反哺社区（issue/PR）。
4. **下一步必须先做"采用评估"（E9），再谈引擎横向对比（E8）**，详见下节。

### E9. 采用评估（新增，最高优先）

| 步骤 | 内容 |
|---|---|
| E9-1 | clone `sixvolts/llamacpp-gfx906-furnace`（`gfx906-perf`），用我们的容器配方构建（**记得加 `-DGGML_CUDA_NO_PEER_COPY=ON`**），并在 8 卡上跑我们自己的 `acceptance.sh` + 标准 bench（pp512/2048/4096/8192、tg128、8 槽并发、MTP） |
| E9-2 | 与我们的 `gfx906-opt` 逐项对比，确认差距（预期长上下文/并发/投机全面落后） |
| E9-3 | 决定采用方式：**(a)** 直接以 furnace 为基座（最快，但把维护押给单维护者）；**(b)** 把它的关键提交 cherry-pick 到我们 rebase 风格的 `gfx906-opt`（可控，但工作量大、且它已含大量上游 merge）；**(c)** 只取我们缺口对应的改动（async input staging、MoE grouped GEMM、MTP head），其余保留上游 + 我们的 F2/F9 |
| E9-4 | 反向输出：把我们的验收套件/测量结论以 issue 或 PR 形式提给他们（他们的验证纪律与我们互补） |

### E8. 引擎横向对比（在 E9 之后）

在同一台机器上跑 **llama.cpp(furnace) vs vLLM(`ttdxq/gfx906-vllm`，或 `0c33` 的自定义 kernel 方案)**，用同类 MoE 模型（若有 GGUF MoE 支持则尽量用我们的 UD-Q4_K_XL）。重点看：
- MoE 量化的 prefill/decode 绝对效率（nlzy fork 的旧结论是"MoE 量化显著慢"，ttdxq 的 2026-09 更新称已可用，需实测）；
- 多流调度（`-np 8` vs vLLM continuous batching）；
- 投机解码（他们 2 卡就到 65~74 t/s）；
- vLLM 侧是否已有 **EP**（我们 E3 已证明完整宽度对专家 GEMM 至关重要）。

## E6. 验收套件（已落地）

`docs/gfx906/scripts/acceptance.sh`：一条命令跑完
① 构建 → ② repack A/B（对照 `--repack 0`，带阈值判定）→ ③ 融合宽度 1 应有差异 + 宽度 1 的 PPL 基线
→ ④ 8 卡是否都用了 repack buffer → 输出 PASS/FAIL 汇总。
每次上游 rebase 后跑它，把验收从"半天"压到"一条命令 + ~30 分钟机器时间"。

---

## E9. furnace 采用评估（**已实测**，2026-10-13）

`sixvolts/llamacpp-gfx906-furnace` @ `7ba588b8d`（2026-10-11），按它的配方 + 我们的容器构建：
`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_NO_PEER_COPY=ON -DGGML_HIP_RCCL=ON -DLLAMA_CURL=OFF`。
同机同模型（8×MI50，UD-Q4_K_XL，`-lm dio`，`-fa on`，`-sm layer`，`-r 2`）：

| 指标 | **我们**（上游+F2+F9） | furnace 默认 | furnace 关异步输入 | furnace 最佳 |
|---|---|---|---|---|
| pp512 | 451.6 | **487.7** | — | 487.7 |
| pp2048 | 1169.6 | **1263.8** | — | 1263.8 |
| pp4096 | **1472.0** | 686.4 | 686.4 | 743.1（ub1024） |
| pp8192 | **1672.5** | 345.4 | 1027.9 | 1128.8（ub1024） |
| 单流 tg128/64 | 35.2（**MTP 40.8**） | 40.6 | **45.1** | 45.1 |
| 8 槽并发总吞吐 | 83.1（MTP 46.9） | **95.5** | — | — |

**已排除的假设（逐个实测，都不是原因）**
- 它的 qwen4exp 专用内核开关全部无效：`GGML_CUDA_NO_FA_SPARSE_PREFILL`、`NO_QSA_POOL`、`NO_QSA_SCORE`、`LLAMA_QSA_NO_MS_CACHE`、
  `NO_MOE_SMALL`、`NO_Q8_NC`、`NO_F32_NC`、`NO_Q8_MULTI`、`NO_Q8_EMIT`、`NO_HC_UP_PRE`、`NO_REPACK_FLATTEN` → pp4096 恒为 685~686。
- 它的生产配置（`-b 16384 -ub 1024`）只把 pp4096 提到 743、pp8192 到 1129。
- `GGML_SCHED_SYNC_INPUTS=1`（关掉它的异步输入 staging）在 pp4096 无变化，但把 **pp8192 从 345 → 1028**：
  即它引以为傲的异步 staging 在**我们 8 卡的长上下文上是负优化**（它是在 4 卡上调的）。
- 重排的影响：`GGML_CUDA_REPACK=0` → pp512 382.7 / pp2048 966 / pp4096 536 / pp8192 807
  ⇒ 它的重排在短上下文 +27~31%，在长上下文反而 **−57%**（与我们 F9 的 +22~34% 相反）。
- `-lm dio` vs 默认加载：无实质差别。

**结论**
1. **不整体采用**：长上下文 prefill 在我们 8 卡上退化 2~5 倍，对"长上下文为主"的负载是硬伤。
2. **摘取三块**（见下节 E10 与"下一步"）：① MTP（我们已有 +25%）；② 多槽位下的 draft 限流开关（~10 行）；
   ③ 它在 8 槽并发上 +15% 的 decode 内核（MoE 小批量/in-kernel 路由、多列 Q8_0 matvec、GDN decode、稀疏 decode attention、hc/shexp 尾部融合）。
3. **反馈一个 bug**：它的异步输入 staging 在 8 卡上把 pp8192 从 1028 拉到 345（可复现），值得报到它的 issue。

## E10. MTP 实测（**我们自己的树就支持**，上游已合 qwen4exp NextN/MTP）

命令：`llama-server ... -md /models/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf --spec-draft-n-max 2`（短 prompt + `ignore_eos`，128 token）。

| 场景 | 无 MTP | 有 MTP | 变化 |
|---|---:|---:|---|
| 单流 decode | 32.6 t/s | **40.8 t/s** | **+25%** |
| 8 槽并发总吞吐 | 83.1 t/s | 46.9 t/s | **−44%** |

- 接受率 **51.6%**（64 accepted / 124 generated），平均接受长度 **2.03** —— 与 furnace 报告的 60~65%（原始文本续写）一致量级。
- 8 槽下变慢的原因与 furnace README 一致：verify 批次让**每个槽**都付 draft 的 matvec 与专家开销，聚合吞吐反而下降。
- **修复只需 furnace 的 `--spec-draft-max-slots N`（"最多 N 个槽在生成时才 draft"），实现约 10 行**：
  `common/common.h` 加一个 `int32_t max_slots`（已在其 328 行）→ `common/arg.cpp` 注册选项 →
  `tools/server/server-context.cpp` 两行判断（`draft_max_slots` / `draft_allowed`）+ props 一行。
  ⇒ 这是**当前性价比最高的一步**：单流 +25%，多流不退化。

## 已定论（本轮之前算清，不必重测）

1. **TP 的集合通信无法被计算掩盖，而且掩盖了也翻不了盘**：meta 后端是"每子图 8 卡并发算 → AllReduce（host 发起，屏障）→ 下一子图"；
   TP 的通信/计算比与序列长度无关（两者都 ∝ token 数）⇒ TP8 prefill 平线；即使把 1.2 s 通信全藏掉，
   pp4096 也只能到 ~871 t/s，仍低于 PP8+repack 的 1468 t/s。
2. **EP 在 8 卡上通信是 TP 的 ~3.7 倍**（49 KB vs 184 KB / token / 层，k=10）；计算量、显存、每卡权重读取量与 TP 相同。
   叠加 E3 的结论（完整宽度 vs 切片），**在 decode 侧 EP 才真正有意义**，但 8 卡 + 纯 PCIe 上仍不建议（llama.cpp 也没有 EP，要从零造）。
3. **repack 在 `-sm tensor` 下用不了**：meta 虚拟设备 `reg = nullptr`（extra-buft 通道不存在）+ 重排布局依赖完整 `ne1`、不可切分。
   fork 用 ~188 行 `ggml-backend-meta.cpp` 解决，未移植。
4. **PP8 + repack 是当前结构性最优**：pp512 451.6 / pp2048 1161.7 / pp4096 1467.7 / pp16384 1739.6 / pp65536 1410.7 / tg128 35.3
   （对照上游 master + F2：339.3 / 853.2 / 1094.9 / 1339.6 / 1150.0 / 34.3）。

---

## 未解决 / 下一步

| # | 待办 | 为什么重要 | 建议做法 |
|---|---|---|---|
| **N1** | layer 模式的 8 张卡是否被**串行化**（prefill 时是否只有 1 张卡在算） | 若成立，是比 kernel 优化更大的杠杆（理论上最多 ~8×）；E4b 的"微基准之和 ≈ 整趟时间"支持这一猜测，但 rocm-smi/amd-smi 采样不可靠、`--device` 在本 build 上加载失败，**未证实** | 用 `rocprofv3` 的 `--kernel-trace` 看不同 Agent 的时间区间是否重叠；或在 sched 层加计时打点 |
| **N2** | 专家 GEMM 为什么只有 11% 峰值效率 | prefill 的第一瓶颈（E4b） | 用 rocprof 采样 `mmq_gemm_repacked` / `mul_mat_vec_rp` 的 occupancy/带宽；试不同 tiling、或改用"按专家分组的一次性 grouped GEMM" |
| **N3** | decode 里的零碎 kernel（逐元素/量化/拷贝 ≈ 27%） | decode 的第二瓶颈（E4a） | 统计 kernel 数量与平均耗时；找可融合的（如 norm+quantize、scaled-add） |
| **N4** | 投机解码（MTP）在本机的实际收益 | E5 表明 width 成本非线性，投机可能正收益 | 用模型自带的 MTP 头跑 `--draft-max 3`，比较 8 槽位下的总吞吐 |
| **N5** | pp65536 掉到 1410（pp16384 是 1740） | 长上下文是主场配置 | FA 与 KV 的分段测量（`-p` 曲线 + 消融） |

---

## 测量卫生（本轮新增，务必遵守）

- **`llama-bench` 的 `test_gen` 永远单 token decode**（源码见 §0）：不要用它做"宽度/批处理"实验，用 `llama-server -np N` + 并发请求。
- `llama-bench` 关 repack 用 **`--repack 0`**（不是 `-nr`/`--no-repack`，也不是 `LLAMA_ARG_REPACK`）。
- `llama-bench` 默认吞 `GGML_LOG_WARN`：要看 witness / `Repacked model buffer size` 必须加 `-v`。
- `llama-perplexity` 的 PPL 随 `-b` 变：on/off 对照**必须同 `-b`**；误差棒相同即逐位一致。
- **rocprofv3 的 attach 模式在容器里不可用**（`librocprofiler-sdk-rocattach.so` 返回 1，即使给 `--cap-add SYS_PTRACE` + `seccomp=unconfined`），需要目标进程预载 profiler。
- **用 profiler 采"加载 + 工作负载"整段代价极高**（加载阶段被拖到 10~20 分钟，因为 dio/repack 期间有大量 HIP 调用被拦截）。
  可行做法：采一次短工作负载，然后按"最大时间间隔"切出窗口（本页 E4a 就是这么做的）。
- `rocm-smi --showuse` / `amd-smi metric --usage` 在本机上**采样不到真实忙时**（多数为 0，偶发单卡尖峰），不能用来做利用率归因。
- 重启仓库后必须抽查一个已知有差异的 shape（如 `test-backend-ops perf -o GATED_DELTA_NET` 的 H=32/tok=512，chunk 应 ~1480 µs 而非 ~1053 µs），避免"用错树构建"的假结果。
