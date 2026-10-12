# gfx906 实测基线

> **数字绑定 commit**：换 commit、换量化、换装载模式都必须重测。本页记录测量口径，未标注口径的数字不可互比。

## 1. 测试环境

| 项 | 值 |
|---|---|
| GPU | 8× AMD Instinct MI60/MI50（`gfx906:sramecc+:xnack-`），32752 MiB/卡，**纯 PCIe，无 xGMI** |
| 主机 | AMD EPYC 7532 32-core，kernel 6.8.0-142，amdgpu 驱动 6.16.13 |
| ROCm | 7.14 定制 gfx906 构建（2026-08-02），HIP `7.14.60850`，clang 23.0.0git，RCCL 2.30.4 |
| 容器 | `qwen-flash-llama-builder:v1`（构建 + 运行合一） |
| 模型 | `models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`（4 shard，合计 103.68 GiB） |
| 参数 | `-ngl 99 -fa on -ctk f16 -ctv f16 -lm dio`，`-p 512,2048,4096 -n 128 -r 3` |

### 容器配方（本机）

```bash
docker run --rm --ipc=host -w /build/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add 993 \
  -v $PWD/build-tp:/build:ro \
  -v $PWD/models:/models:ro \
  --entrypoint bash qwen-flash-llama-builder:v1 -lc '
    export LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib
    ./llama-bench -m /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
      -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -p 512,2048,4096 -n 128 -r 3 -o md'
```

**四个必踩的坑**：

1. `--ipc=host`：默认 `/dev/shm` 只有 64 MB，RCCL 初始化会失败（`NCCL init failed (unhandled system error)`），
   然后回退 internal（只支持 2 卡）→ meta butterfly。加了它 RCCL 才起来（日志 `NCCL INFO RCCL Tuning index`）。
2. `-w /build/bin`（容器默认 WORKDIR 是 `/app`）：ggml 的后端搜索路径含"可执行文件目录 + 当前目录"，
   从 `/app` 启动会加载镜像里**旧的** `libggml-cpu-haswell.so` / `libggml-rpc.so`。
3. `--group-add 993`（render 的 GID），**不能写 `--group-add render`**（容器内无该组名）。
4. `LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib`，避免混到 `/app` 的旧库。

## 2. 官方上游 `f2cc7282c`（= upstream/master @ 2026-10-11）

加载模式 `-lm dio`（HSA_FORCE_FINE_GRAIN_PCIE 对上游是 no-op，已验证：339.39/852.82/33.75）：

| 配置 | pp512 | pp2048 | pp4096 | tg128 |
|---|---:|---:|---:|---:|
| `-sm layer`（PP8） | 339.27 ± 1.89 | 853.18 ± 6.79 | 1094.92 ± 2.06 | **34.33 ± 0.03** |
| `-sm tensor`（TP8 + RCCL） | 539.87 ± 8.50 | 519.82 ± 3.11 | 514.42 ± 1.35 | 32.68 ± 2.83 |

参考（更早一次，默认 mmap 加载 + `-v`，口径不同，仅作趋势参考）：
`-sm layer` pp512 290.31 / pp2048 802.65 / tg128 34.04；`-sm tensor` pp512 543.32 / pp2048 521.49 / tg128 32.54。

**要点**：TP8 的小 batch prefill 明显强于 PP8（+59% @pp512），但 pp2048 之后被 PCIe 上的 allreduce 拖住；
decode 两者接近（PP8 略优）。**tg8 之类的短测会被一次性开销污染**（RCCL 首次建链约 0.8 s，
`tg8=7.64` 反推：8 token 用了 1.05 s），decode 一律用 `tg128`。

## 3. mxxm fork `eefc4e732`（2026-09-20，落后上游 799 提交）

同一口径（`-lm dio`，含 `HSA_FORCE_FINE_GRAIN_PCIE=1`、`GPU_MAX_HW_QUEUES=8`）：

| 配置 | pp512 | pp2048 | pp4096 | tg128 |
|---|---:|---:|---:|---:|
| `-sm layer`（PP8） | 450.41 | 1107.35 | 370.33 ⚠️ | 31.31 |
| `-sm tensor -tps 0`（TP8） | 705.71 | 687.92 | 691.36 | 35.84 |
| `-sm tensor -tps 0` + `LLAMA_PLE_SHARD=1` | 720.92 | 691.10 | 692.78 | 35.57 |
| `-sm tensor -tps 4`（**TP4×PP2**） | 593.69 | 863.07 | 951.16 | **37.25** |
| `-sm tensor -tps 4` + `LLAMA_PLE_SHARD=1` | 605.10 | 862.00 | 944.49 | 37.00 |
| `-sm tensor -tps 2`（TP2×PP4） | 483.22 | 972.09 | **1204.39** | 36.60 |
| `-sm tensor -tps 2` + `LLAMA_PLE_SHARD=1` | 490.69 | 978.90 | 1210.70 | 36.43 |

⚠️ fork PP8 的 pp4096=370 与它自己的 pp2048=1107 自相矛盾（上游是 853→1095），是第一个 session 的
冷启/页缓存伪影，**不可用**。

**结论**：
- 最优单流 decode = fork `-tps 4`（TP4×PP2）**37.25 t/s**，比官方最好配置（上游 PP8 34.33）**+8.5%**，
  比 fork 自己的 PP8（31.31）**+19%**。
- fork 的 TP 路径整体更快（TP8 35.84 vs 上游 32.68，+9.7%），但 **layer 模式反而更慢**（31.31 vs 34.33，−8.8%）
  → 用 fork 就必须用 `-sm tensor`。
- `LLAMA_PLE_SHARD` 在 8 卡热缓存下 **±2%（无收益）**，特性确实生效（日志 `sharding the PLE gather table across the TP group`），
  但文档的 4×MI50 冷缓存 2 倍收益复现不了（见 `features.md` F5）。

## 3.1 F2 自定义 AllReduce 的 A/B（本仓库 `75aba2e4e`，2026-10-12）

上游 `f2cc7282c` + 单个 patch（`tp-allreduce.{cu,cuh}` + `ggml-cuda.cu` 147 行），
`-sm tensor`、`-lm dio`、`HSA_FORCE_FINE_GRAIN_PCIE=1`、`-p 512,2048 -n 128 -r 3`：

| 配置 | pp512 | pp2048 | tg128 |
|---|---:|---:|---:|
| 自定义 AR 开（默认） | 543.46 ± 2.37 | 518.70 ± 3.05 | 33.80 ± 3.15 |
| `GGML_ENABLE_CUSTOM_AR=0` | 539.44 ± 8.95 | 519.83 ± 2.80 | 32.44 ± 2.79 |

**结论：在 8 卡 TP8 + RCCL 场景下差异落在噪声内**（tg 上 +4.2% 但 σ≈3；pp 基本不变）。
路径确认已生效（日志）：

```
TP custom AllReduce: initialized for 8 GPUs, path = broadcast F32 + twoshot F32 (peer-write, size-adaptive, lossless)
TP custom AllReduce: auto-selected two-shot for 8-rank decode-sized tensors
```

不设 `HSA_FORCE_FINE_GRAIN_PCIE=1` 时：`path = one-shot (PCIe without HSA_FORCE_FINE_GRAIN_PCIE=1, ...)`，
且因为 RCCL 可用，所有集合通信仍然走 RCCL（prepare 自行拒绝）。

> 需要更多重复才能判定这 ~4% 是否真实：把 `-r` 提到 8，或改测 `-n 256`。

### 3.1.1 token 一致性（同一 patch，脚本 `scripts/f2-custom-ar-consistency.sh`）

贪心解码 `--temp 0 --seed 1 -n 32 -st`，prompt = `alpha bravo charlie delta echo foxtrot`：

| 配置 | 结果 |
|---|---|
| `-sm layer`（PP8） | 与下面两者逐字节一致 |
| `-sm tensor` + 自定义 AR | 与上、下两者逐字节一致 |
| `-sm tensor` + `GGML_ENABLE_CUSTOM_AR=0`（RCCL） | 与上两者逐字节一致 |

结论：**通过** ✅ —— 三条路径给出完全相同的 32 token 输出（连 PP8 与 TP8 在这个短贪心生成上也一致）。

> 坑：不要直接对 stdout 取 md5。stdout 里含 `[ Prompt: x t/s | Generation: y t/s ]`，该行每次必然不同，
> 会出现“三组全不一致”的假警报；剔除该行后完全一致。
> 另外，默认 verbosity 会吞掉 `TP custom AllReduce: initialized ...` 这条 INFO，取证时需要给 `llama-cli` 加 `-v`。

## 4. 与模型相关的关键事实（供以后判断）

- 权重里约 **27465 MiB（26.8 GiB）是 host-mapped 的 PLE 表**（`per_layer_token_embd.weight`，
  16 head × 约 2000 万行 × 160 宽 IQ4_NL ≈ 90 B/行）；上游默认把它钉在 CPU buffer 上按需换页，
  每个 graph 里有约 **4380 处 `rs_s_copy` / `GET_ROWS`** 走主机侧 gather。
- 8 卡 layer 模式的显存占用：每卡 9.05–11.81 GiB（47 层 + output），另有 26.8 GiB 在主机。
- 结论上的含义：decode 的天花板与主机侧 PLE gather 强相关；**TP 对 decode 的收益有限，主要收益在
  小 batch prefill**（`-tps 2` 的 pp4096 最高）。

## 5. 重测清单

1. `git fetch upstream && git rebase upstream/master`（见 `README.md`）
2. 重新配置构建（`-DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DGGML_HIP_RCCL=ON`）
3. 用上面的容器配方跑 `-sm layer` 与 `-sm tensor`（TP 需 `--ipc=host`）
4. 结果写入本页，并**标注 commit hash + 加载模式 + 是否 `--ipc=host`**
5. 页缓存冷热会显著影响 `pp4096` 与任何 PLE 相关结论：比较不同配置时保持同一缓存状态，
   必要时先 `echo 3 > /proc/sys/vm/drop_caches`（需 sudo，会扰动整机，慎用）
