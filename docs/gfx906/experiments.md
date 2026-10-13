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

## E7. 引擎选型：vLLM 是不是选项

- **模型侧：vLLM 有**。`vllm/model_executor/models/qwen3_next.py` 实现了 Qwen3-Next 的
  `Qwen3NextSparseMoeBlock`；GDN 层已抽成共享的 `GatedDeltaNetAttention`（PR #37975）；官方博客/Recipes 有 Qwen3-Next 支持与 MTP。
- **硬件侧：明确不支持 gfx906**。
  - vLLM issue #22458（"Please add support for AMD Instinct MI50"，2025-08）→ **closed as not planned**；
  - PR #33811 专门加注释说明 gfx906(MI50/MI60) 不被支持（op 里说明即便列在 HIP 架构里也跑不起来）；
  - MI50 自 ROCm 6.0 起被 deprecated；vLLM 要求 ROCm 6.3+；
  - PR #12431（"add support for AMD MI25/50/60"）亦被关闭。
- **结论**：换 vLLM = fork vLLM + 把它的 Triton/量化 kernel 移植到 gfx906（Triton 无 gfx906 后端）+ 适配新架构变体，
  是**数月级、且收益未证**的工程。**llama.cpp 仍是基座**，优化应针对 E3/E4 测出的瓶颈（专家 GEMM 效率、decode 的零碎 kernel）。

---

## E6. 验收套件（已落地）

`docs/gfx906/scripts/acceptance.sh`：一条命令跑完
① 构建 → ② repack A/B（对照 `--repack 0`，带阈值判定）→ ③ 融合宽度 1 应有差异 + 宽度 1 的 PPL 基线
→ ④ 8 卡是否都用了 repack buffer → 输出 PASS/FAIL 汇总。
每次上游 rebase 后跑它，把验收从"半天"压到"一条命令 + ~30 分钟机器时间"。

---

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
