# gfx906 实验记录（2026-10-12 会话）

> **目标声明（用户 2026-10-12 明确）**：目标是**改造推理引擎**，让 `Qwen3.8-Flash-Next`(qwen4exp) 在
> 本机（8×MI50 gfx906，纯 PCIe，ROCm 7.14）上跑出好效率；不是"在现有引擎上挑一套好参数"。
> **有决心改代码**，必要时甚至换引擎（如 vLLM）并移植 kernel。
> 因此本页的实验都以"**该改哪条执行路径 / 哪个 kernel**"为导向，而不是"选哪组参数"。

环境与基线见 `baselines.md`；分支/同步流程见 `README.md`；特性清单见 `features.md`。
模型：`UD-Q4_K_XL`（103.68 GiB，Q4_K 44.6% / IQ4_NL 28.9%（PLE 表）/ Q5_1 20.4% / Q8_0 5.1% / Q5_K 0.9%），
48 层中 36 层带 SSM/GDN，MoE 512 专家（~10 激活），`n_head_kv=2`。

---

## E1. decode 的批量宽度扫描（PP8 / PP8+融合 / TP8）

**目的**：多流并发与投机推理的每个 forward 宽度 = 并发槽位数 ×（1+草稿深度）；`llama-bench -b B` 正是
"一个 forward 处理 B 个 token"，所以在同一宽度上比较就等于在该并发度上比较（维度对齐）。
命令：`-lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -p 16 -n 128 -r 3 -b B -ub B -o md`。

| 宽度 B | PP8 | PP8+融合 | TP8 |
|---:|---:|---:|---:|
| 1 | 35.01 ±0.20 | （未测） | 34.14 ±3.25 |
| 2 | 35.41 ±0.02 | **36.63 ±0.02** | 34.12 ±3.34 |
| 4 | 35.30 ±0.09 | **36.60 ±0.05** | 34.14 ±3.28 |
| 8 | 35.11 ±0.29 | **36.39 ±0.30** | 34.21 ±3.34 |
| 16 | 35.24 ±0.04 | **36.55** | 33.48 |
| 32 | 35.56 | **36.30** | 33.95 |

**结论**

1. **PP8 的吞吐对宽度是平的（35.0 → 35.6）**：每个 forward 的耗时**线性正比于宽度**，批处理**没有任何摊薄**。
   推论：本机的 decode 总吞吐被钉在 **≈35–36 t/s，与并发槽位数无关**——多流并发只能提高单请求的占用比，
   不能提高整机吞吐。这条直接否掉了"靠批处理/调度改进拿 decode 收益"的路线。
2. **推测其瓶颈是"每 token 的专家权重读取"**：MoE 每 token 选 ~10 个专家，不同 token 选不同专家，
   所以权重流量 ∝ token 数（不像 dense 权重那样可被 batch 摊薄）。
   ⚠️ **这是假设，未证实**——必须由 E4 坐实。
3. **推论（重要）：投机解码在本机上大概率是净亏**。若宽度 4 的 verify 步耗时 ≈ 4× 宽度 1 的步耗时
   （E1 实测就是线性的），而 4 宽只提交 ~3 个 token（3 中 3），则有效吞吐 ≈ 3/0.113 s ≈ 26 t/s < 35 t/s。
   只有验收率极高（或 verify 步亚线性）才划算。**在投入 MTP/draft 之前必须先测 verify 步的宽度成本。**
4. **TP8 在所有宽度上都不优于 PP8**（34.1 对 35.0 @b=1；33.5–34.0 对 35.1–35.6 @b=16/32），
   且方差大（±3.3，RCCL 初始化）。⇒ 即便将来补上 TP+repack，decode 侧也没有翻盘空间
   （repack 本身是 prefill 优化，decode 只值 1~3%）。**TP 的唯一价值仍在短宽度 prefill。**
5. **融合在宽度 ≥2 时值 +3.5~4%**（36.3–36.6 对 35.1–35.4），而 E2 证明该宽度区间**逐位一致** ⇒
   **多流/投机场景可以零数值代价打开融合**（`GGML_CUDA_REPACK_MMV_FUSION=1`）。

---

## E2. 融合核在宽度 ≥2 时是否逐位一致（llama-perplexity）

**目的**：决定"融合默认关"这个策略在多流场景下是否还需要。
命令：`llama-perplexity -f corpus-perplexity.txt -c 512 --chunks 2 -b B`，对照 `GGML_CUDA_REPACK_MMV_FUSION`。

| 宽度 | 融合 ON | 融合 OFF | 判定 |
|---:|---|---|---|
| 1 | 3.5869 ±0.36028 | 3.6241 ±0.36672 | 不同（+1.0%） |
| 2 | 3.5886 ±0.36072 | **3.5886 ±0.36072** | **逐位一致** ✅ |
| 4 | 3.5793 ±0.35953 | **3.5793 ±0.35953** | **逐位一致** ✅ |

**结论**：宽度 2 与 4 时连误差棒都完全相同 ⇒ 每个 chunk 的 loss 相同 ⇒ forward **逐位一致**；
只有宽度 1 因归约顺序不同而分叉。**多流并发（槽位≥2）或投机验证（宽度≥2）时融合零代价**
（`features.md` 里"默认关"的理由只对单流宽度 1 成立）。

---

## 已定论（本轮讨论中算清，不必重测）

1. **TP 的集合通信无法被计算掩盖，而且掩盖了也翻不了盘。**
   - llama.cpp 的 meta 后端执行是"每子图：8 卡并发算 → AllReduce（host 发起，屏障）→ 下一子图"
     （`ggml-backend-meta.cpp` 主循环），通信在关键路径上，零重叠。
   - TP 的通信/计算比 ρ 与序列长度**无关**（两者都 ∝ token 数）⇒ 这就是 TP8 prefill 平线（721/691/693）的成因；
     PP 的 ρ 小 1~2 个数量级（每层 96 次集合通信 vs 每 forward 7 次边界传递）⇒ 它能吃下"序列变长"的 GEMM 效率红利（452→1162→1468）。
   - 定量上界（pp4096，fork 实测 TP8+repack 692.8 t/s = 5.91 s/趟；通信估算 ≈1.2 s/趟）：
     **即使通信 100% 被掩盖 → 4.7 s ≈ 871 t/s，仍 < PP8+repack 的 1468 t/s（2.79 s）**。
     结论：TP 的劣势不只是通信，还叠加每层多次子图下发/同步、per-card GEMM 变窄、GQA 头数少（`n_head_kv=2`）导致的注意力并行度浪费。
2. **专家并行（EP）在 8 卡上价值为负。**
   - EP 与 TP 的**计算量、显存、每卡权重读取量完全相同**（TP 每卡算"被选中专家的 1/8 切片"，EP 每卡算"自己那 1/8 专家的整张"）。
   - 通信：TP = 2×AllReduce ≈ **49 KB**/token/层（d=6144，f32）；EP = dispatch+combine ≈ 2×(去重 rank 数 7~8)×d×2B ≈ **184 KB**（bf16）
     ⇒ **EP 是 TP 的 ~3.7 倍**（k=10、8 卡是最差比例；EP 只在 rank≫k 时才有优势）。
   - EP 唯一的潜在收益是"保住专家 GEMM 的完整形状"（我们专家 ne0=2560，TP 切 8 份后 N≈192 偏窄）→ **交给 E3 判定**。
   - llama.cpp 没有 EP（只有 none/layer/row/tensor），实现等于从零造 all-to-all dispatch/combine + 专家分区 + 负载均衡。
3. **repack 在 `-sm tensor` 下用不了**：TP 走 meta 虚拟设备，而 `ggml_backend_meta_device()` 造出来的设备 `reg = nullptr`
   （无 backend registry）⇒ `make_gpu_buft_list` 的 extra-buft 通道根本不存在；加之重排布局**依赖完整 ne1、不可切分**，
   必须"每 lane 各自重排切片"并绕过 meta 的分块上传。fork 用 ~188 行 `ggml-backend-meta.cpp` 解决。
   实测：layer 模式 8 行 `Repacked model buffer size`，tensor 模式 0 行、权重落 `Meta()`。
4. **PP8 + repack 是当前结构性最优**（pp16384 +30.4%、pp65536 +22.5%、pp4096 +34%、decode +3~6%）。

---

## 待做实验（下次会话按此顺序）

| # | 实验 | 怎么做 | 能得出的结论 | 为什么这个测量能支撑该结论 |
|---|---|---|---|---|
| **E4** | **时间归因 / 屋顶线**（最高优先） | ① `-p` 分段曲线看拐点；② rocprof 采样 1~2 个 case 看 kernel 占比；③ 消融（关 PLE、改 n_expert_used、改 `-b`） | decode 35 t/s 与 pp65536 掉到 1410 各自被什么限制（MoE 专家权重读取 / PLE host 表 / SSM-GDN 串行算子 / attention） | 只有知道时间归属，才能决定"改哪个 kernel / 哪条执行路径"。这是唯一可能推翻 E1 那条假设的测量 |
| **E3** | MoE GEMM 形状效率（EP 的动机） | `test-backend-ops perf -o MUL_MAT_ID`，加探针用例比较 N=1536 与 N=192 | 形状损失幅度；与已知的通信差（3.7×）直接对比 | EP 与 TP 除"形状"和"通信"外全部相同；若形状损失 <10%，EP 收益上限远小于其通信代价 ⇒ 直接否决 |
| **E5** | 真并发压测（生产等价） | `llama-server -np 8`（PP8/TP8 各一轮）+ 8 路并发客户端，量总吞吐/TTFT/TPOT；长上下文探 KV 容量边界 | 多流场景最终配置 + KV 容量账（TP 在 `n_head_kv=2` 时 KV 可能被复制） | E1 的 `-b` 只模拟宽度，不模拟"多序列 KV 分散访问 + prefill/decode 交错调度" |
| **E7** | 引擎选型可行性 | 查 vLLM 对 gfx906 的支持面（无 MFMA）+ 该架构是否在模型库；评估"移植 kernel vs 改 llama.cpp"的工程量 | 要不要换引擎 / 换哪条路 | 模型是混合 GDN + 512 专家 MoE + hash 式 PLE 表，属非主流架构；gfx906 无矩阵核，两边都要自研 kernel |
| **E6** | 验收套件脚本化（工程卫生） | 一条命令跑完：构建 → repack A/B（`--repack 0`）→ token 一致性 → F2 一致性 → 与基线表对比 | 把每次上游 rebase 的验收压到"一条命令 + 30 分钟机器时间" | 让"同步成本"可被持续验证，而不是靠估计（当前估计：2~4 个小 hunk，见 README） |

---

## 测量卫生（本轮新增的坑）

- `llama-bench` 的关 repack 开关是 **`--repack 0`**；它**不接受** `-nr`/`--no-repack`（那是 llama-cli/server 通用参数），
  也**不认** `LLAMA_ARG_REPACK` 环境变量。
- `llama-bench` 默认吞掉 `GGML_LOG_WARN`：要看 witness/`Repacked model buffer size` 必须加 `-v`。
- `llama-perplexity` 的 PPL **随 `-b` 变化**（b=1 约 3.59 / b=2 约 3.59 / b=512 约 3.77），
  所以 on/off 对照**必须在同一个 `-b` 下**比较；误差棒相同即逐位一致。
- 融合的 witness：`[repack-moe-fusion] type=q4_K cols=1 fused up/gate mat-vec engaged`。
- 重构仓库后必须抽查一个已知有差异的 shape（如 `test-backend-ops perf -o GATED_DELTA_NET` 的 H=32/tok=512），
  避免再次出现"用错树构建"导致的假结果（F3b 期间踩过）。

---

## 本轮的状态快照

- 分支 `gfx906-opt`（相对 `upstream/master` = `f2cc7282c`）：代码 patch 只有 **F2**（自定义 AllReduce）
  和 **F9**（gfx906 权重重排 + MoE/稠密融合接线），热点文件 +284 行，其余是新目录 `q8_repack/`(4516 行) 与 `tp-allreduce.*`。
  **未推送**（rebase 过，将来要推需 `--force-with-lease`）。
- 当前最佳配置：`-sm layer`（PP8）+ repack（默认开）+ 多流场景下 `GGML_CUDA_REPACK_MMV_FUSION=1`。
  实测：pp512 451.6 / pp2048 1161.7 / pp4096 1467.7 / pp16384 1739.6 / pp65536 1410.7 / tg128 35.6–36.6。
- 对照（上游 master + F2）：339.3 / 853.2 / 1094.9 / 1339.6 / 1150.0 / 34.3。
