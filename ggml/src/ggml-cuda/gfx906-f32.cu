// GCN (gfx906) F32 mat-vec family, ported from the gfx906-perf/furnace family.
// The generic MMVF path walks each row with one small workgroup and re-reads x
// per row; on wave64 devices this is latency-bound for thin-but-wide F32
// matrices (qwen4exp: hc injections [10240 -> 4], MoE router [2560x512],
// GDN beta/alpha [2560x48]). Kernels here give each row one wave (or one
// 1024-thread workgroup for few long rows), weight rows read all columns
// together in the batched variants, and grouped launches serve up to three
// mat-vec products over the same activation in one go.
#include "common.cuh"
#include "mmf.cuh"
#include "gfx906-f32.cuh"
#include "../ggml-backend-impl.h"

// E17.2 (2026-10-17) BUGFIX: the guard used to be `defined(GGML_USE_HIP) && defined(__gfx906__)`.
// `__gfx906__` is only defined in the HIP *device* pass, so in the host pass the whole
// translation unit (kernels + host dispatch) compiled away: gfx906-f32.cu.o was 1.5 KB and
// ZERO gcn_f32_* kernels ever appeared in a decode profile. That is why T10 measured
// "neutral" and why the f32m grouping never formed - the port was dead code.
// Host dispatch + kernel declarations must live in the host pass too (cf.
// gfx906-hc-up-pre.cu, which only guards kernel *bodies*). Runtime arch gating is
// done by GGML_CUDA_CC_IS_GCN(cc).
#if defined(GGML_USE_HIP)
#define GCN_F32_ENABLED 1
#endif

// Tolerant env parsing: GFX906_F32 is DEFAULT-ON; only an explicit 0/off/false/no
// disables it. `getenv(...) != nullptr` would make GFX906_F32=0 *enable* it.
// Exposed through gfx906-f32.cuh so that the `_OFF`-style switches (e.g.
// GFX906_F32_MULTI_OFF, read from ggml-cuda.cu) use the SAME parser instead of
// the "present => off" form that PLAYBOOK section 3 forbids.
// Semantics: returns def when the variable is unset/empty; false when the value
// is 0/off/false/no (any case); true otherwise. For an `_OFF` switch the caller
// inverts the result, i.e. `!env_on("..._OFF", false)` == "feature is on".
bool ggml_cuda_f32_env_on(const char * name, bool def) {
    const char * v = getenv(name);
    if (v == nullptr || *v == '\0') {
        return def;
    }
    if (v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F' ||
            strcmp(v, "off") == 0 || strcmp(v, "OFF") == 0) {
        return false;
    }
    return true;
}

#if GCN_F32_ENABLED

// F32 matvec with few long rows: one 1024-thread workgroup per row
static __global__ void __launch_bounds__(1024) gcn_f32_matvec_fewrows(
        const float * __restrict__ A, const float * __restrict__ x, float * __restrict__ y,
        const int K, const int64_t lda) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const float4 * a4 = reinterpret_cast<const float4 *>(A + (size_t) blockIdx.x * lda);
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float acc = 0.0f;
    for (int i = threadIdx.x; i < K / 4; i += 1024) {
        const float4 a = a4[i];
        const float4 b = x4[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = warp_reduce_sum<warp_size>(acc);
    __shared__ float part[1024 / ggml_cuda_get_physical_warp_size()];
    if (threadIdx.x % warp_size == 0) {
        part[threadIdx.x / warp_size] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 1024 / ggml_cuda_get_physical_warp_size(); w++) {
            sum += part[w];
        }
        y[blockIdx.x] = sum;
    }
}

// one wave per row; ITERS > 0: K/4 == ITERS*warp_size, loads issued before FMAs
template <int ITERS>
static __global__ void __launch_bounds__(256) gcn_f32_matvec_rows(
        const float * __restrict__ A, const float * __restrict__ x, float * __restrict__ y,
        const int K, const int64_t lda, const int M) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int row = blockIdx.x * (256 / warp_size) + threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    if (row >= M) {
        return;
    }
    const float4 * a4 = reinterpret_cast<const float4 *>(A + (size_t) row * lda);
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float acc = 0.0f;
    if constexpr (ITERS > 0) {
        float4 a[ITERS];
        float4 b[ITERS];
#pragma unroll
        for (int j = 0; j < ITERS; j++) {
            a[j] = a4[lane + j * warp_size];
            b[j] = x4[lane + j * warp_size];
        }
#pragma unroll
        for (int j = 0; j < ITERS; j++) {
            acc += a[j].x * b[j].x + a[j].y * b[j].y + a[j].z * b[j].z + a[j].w * b[j].w;
        }
    } else {
#pragma unroll 4
        for (int i = lane; i < K / 4; i += warp_size) {
            const float4 a = a4[i];
            const float4 b = x4[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
    acc = warp_reduce_sum<warp_size>(acc);
    if (lane == 0) {
        y[row] = acc;
    }
}

// few rows x 2..16 columns: each weight row is read once for all columns
template <int NC>
static __global__ void __launch_bounds__(1024) gcn_f32_matvec_fewrows_nc(
        const float * __restrict__ A, const float * __restrict__ x, float * __restrict__ y,
        const int K, const int64_t lda, const int64_t sx, const int64_t sy) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const float4 * a4 = reinterpret_cast<const float4 *>(A + (size_t) blockIdx.x * lda);
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.0f;
    }
    for (int i = threadIdx.x; i < K / 4; i += 1024) {
        const float4 a = a4[i];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float4 b = reinterpret_cast<const float4 *>(x + (size_t) c * sx)[i];
            acc[c] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
    __shared__ float part[NC][1024 / ggml_cuda_get_physical_warp_size()];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_reduce_sum<warp_size>(acc[c]);
        if (threadIdx.x % warp_size == 0) {
            part[c][threadIdx.x / warp_size] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < NC) {
        const int c = threadIdx.x;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 1024 / ggml_cuda_get_physical_warp_size(); w++) {
            sum += part[c][w];
        }
        y[c * sy + blockIdx.x] = sum;
    }
}

template <int NC>
static __global__ void __launch_bounds__(256) gcn_f32_matvec_rows_nc(
        const float * __restrict__ A, const float * __restrict__ x, float * __restrict__ y,
        const int K, const int64_t lda, const int M, const int64_t sx, const int64_t sy) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int row = blockIdx.x * (256 / warp_size) + threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    if (row >= M) {
        return;
    }
    const float4 * a4 = reinterpret_cast<const float4 *>(A + (size_t) row * lda);
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.0f;
    }
#pragma unroll 2
    for (int i = lane; i < K / 4; i += warp_size) {
        const float4 a = a4[i];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float4 b = reinterpret_cast<const float4 *>(x + (size_t) c * sx)[i];
            acc[c] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_reduce_sum<warp_size>(acc[c]);
        if (lane == 0) {
            y[c * sy + row] = v;
        }
    }
}

// grouped: up to three F32 single-column mat-vec products over the same x,
// one wave per row, weights dereferenced per group member (GDN beta+alpha,
// MoE router + shared-expert gate; K == 2560 -> ITERS specialization)
struct gcn_f32_multi_args {
    const float * A[3];
    float *       y[3];
    int64_t       lda[3];
    uint32_t      rows[3];
    uint32_t      start[3];
};

template <int ITERS>
static __global__ void __launch_bounds__(256) gcn_f32_matvec_rows_multi(const gcn_f32_multi_args args, const float * __restrict__ x) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ITERS_L = ITERS > 0 ? (2560 / 4 / warp_size) : 0;
    const uint32_t gw   = blockIdx.x * (256 / warp_size) + threadIdx.x / warp_size;
    const int      lane = threadIdx.x % warp_size;
    const int t = gw >= args.start[2] ? 2 : (gw >= args.start[1] ? 1 : 0);
    const uint32_t row = gw - args.start[t];
    if (row >= args.rows[t]) {
        return;
    }
    const float4 * a4 = reinterpret_cast<const float4 *>(args.A[t] + (size_t) row * args.lda[t]);
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float acc = 0.0f;
    if constexpr (ITERS_L > 0) {
        float4 a[ITERS_L];
        float4 b[ITERS_L];
#pragma unroll
        for (int j = 0; j < ITERS_L; j++) {
            a[j] = a4[lane + j * warp_size];
            b[j] = x4[lane + j * warp_size];
        }
#pragma unroll
        for (int j = 0; j < ITERS_L; j++) {
            acc += a[j].x * b[j].x + a[j].y * b[j].y + a[j].z * b[j].z + a[j].w * b[j].w;
        }
    } else {
#pragma unroll 4
        for (int i = lane; i < 2560 / 4; i += warp_size) {
            const float4 a = a4[i];
            const float4 b = x4[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
    acc = warp_reduce_sum<warp_size>(acc);
    if (lane == 0) {
        args.y[t][row] = acc;
    }
}

#endif // GCN_F32_ENABLED

// F32 mat-vec family entry point: DEFAULT-ON, `GFX906_F32=0` opts out. Not bitwise
// vs MMVF (different summation order). Needs the hook in mmvf.cu as well as
// mmf.cu: single-token decode (ncols_dst == 1) never reaches ggml_cuda_mul_mat_f.
bool ggml_cuda_f32_gcn_matvec(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, int cc) {
#if GCN_F32_ENABLED
    static const bool enabled = ggml_cuda_f32_env_on("GFX906_F32", true);
    GGML_TENSOR_BINARY_OP_LOCALS;
    // E17.2 diagnosis: dump every F32 mat-vec candidate shape that reaches this hook,
    // with whether the family fired. Bounded so graph capture stays readable.
    static const bool dbg = getenv("GFX906_F32_DEBUG") != nullptr;
    static int dbg_n = 0;
    if (dbg && dbg_n < 400) {
        const bool shape_ok = (ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1);
        GGML_LOG_WARN("[gfx906-f32] cand ne00=%lld ne01=%lld ne11=%lld ne02=%lld nb01=%llu nb0=%llu shape_ok=%d fire=%d\n",
                      (long long) ne00, (long long) ne01, (long long) ne11, (long long) ne02,
                      (unsigned long long) nb01, (unsigned long long) nb0, (int) shape_ok, (int) enabled);
        dbg_n++;
    }
    if (!enabled) {
        return false;
    }
    constexpr int warp_size = 64;
    static const bool no_rows  = !ggml_cuda_f32_env_on("GFX906_F32_ROWS", true);
    static const bool no_nc    = !ggml_cuda_f32_env_on("GFX906_F32_NC", true);

    if (cc == 0 || !GGML_CUDA_CC_IS_GCN(cc) || src0->type != GGML_TYPE_F32 ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            ne02 != 1 || ne03 != 1 || ne12 != 1 || ne13 != 1) {
        return false;
    }

    // 2..16 columns (speculative verify, few-token prefill): weights read once for all columns
    if (!no_nc && ne11 >= 2 && ne11 <= 16 && (ne01 <= 16 || ne01 > 16) &&
            ne00 >= 2048 && ne00 % 16 == 0 &&
            nb00 == sizeof(float) && nb10 == sizeof(float) && nb01 % 16 == 0 && nb11 % 16 == 0 && nb0 == sizeof(float) &&
            ((uintptr_t) src0->data % 16 == 0) && ((uintptr_t) src1->data % 16 == 0)) {
        const float * A  = (const float *) src0->data;
        const float * x  = (const float *) src1->data;
        float       * y  = (float *) dst->data;
        const int64_t lda = nb01 / sizeof(float);
        const int64_t sx  = nb11 / sizeof(float);
        const int64_t sy  = nb1  / sizeof(float);
        const int rows_per_block = 256 / warp_size;
        auto launch = [&](auto nc) {
            constexpr int NC = decltype(nc)::value;
            if (ne01 <= 16) {
                gcn_f32_matvec_fewrows_nc<NC><<<ne01, 1024, 0, ctx.stream()>>>(A, x, y, (int) ne00, lda, sx, sy);
            } else {
                gcn_f32_matvec_rows_nc<NC><<<(ne01 + rows_per_block - 1) / rows_per_block, 256, 0, ctx.stream()>>>(
                    A, x, y, (int) ne00, lda, (int) ne01, sx, sy);
            }
        };
        switch (ne11) {
            case 2:  launch(std::integral_constant<int, 2>{});  break;
            case 3:  launch(std::integral_constant<int, 3>{});  break;
            case 4:  launch(std::integral_constant<int, 4>{});  break;
            case 5:  launch(std::integral_constant<int, 5>{});  break;
            case 6:  launch(std::integral_constant<int, 6>{});  break;
            case 7:  launch(std::integral_constant<int, 7>{});  break;
            case 8:  launch(std::integral_constant<int, 8>{});  break;
            case 9:  launch(std::integral_constant<int, 9>{});  break;
            case 10: launch(std::integral_constant<int, 10>{}); break;
            case 11: launch(std::integral_constant<int, 11>{}); break;
            case 12: launch(std::integral_constant<int, 12>{}); break;
            case 13: launch(std::integral_constant<int, 13>{}); break;
            case 14: launch(std::integral_constant<int, 14>{}); break;
            case 15: launch(std::integral_constant<int, 15>{}); break;
            default: launch(std::integral_constant<int, 16>{}); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    // 1 column, few long rows (hc injections): one 1024-wide workgroup per row
    if (ne11 == 1 && ne01 <= 16 && ne00 >= 2048 && ne00 % 16 == 0 &&
            nb00 == sizeof(float) && nb10 == sizeof(float) && nb01 % 16 == 0 && nb0 == sizeof(float) &&
            ((uintptr_t) src0->data % 16 == 0) && ((uintptr_t) src1->data % 16 == 0)) {
        gcn_f32_matvec_fewrows<<<ne01, 1024, 0, ctx.stream()>>>(
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
            (int) ne00, nb01 / sizeof(float));
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    // 1 column, many rows: one wave per row
    if (!no_rows && ne11 == 1 && ne01 > 16 && ne01 <= 16384 && ne00 >= 1024 && ne00 % 16 == 0 &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            nb00 == sizeof(float) && nb10 == sizeof(float) && nb01 % 16 == 0 && nb0 == sizeof(float) &&
            ((uintptr_t) src0->data % 16 == 0) && ((uintptr_t) src1->data % 16 == 0)) {
        const int rows_per_block = 256 / warp_size;
        const dim3 grid((ne01 + rows_per_block - 1) / rows_per_block);
        if (ne00 == 10240) {
            gcn_f32_matvec_rows<40><<<grid, 256, 0, ctx.stream()>>>(
                (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
                (int) ne00, nb01 / sizeof(float), (int) ne01);
        } else if (ne00 == 6144) {
            gcn_f32_matvec_rows<24><<<grid, 256, 0, ctx.stream()>>>(
                (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
                (int) ne00, nb01 / sizeof(float), (int) ne01);
        } else if (ne00 == 2560) {
            gcn_f32_matvec_rows<10><<<grid, 256, 0, ctx.stream()>>>(
                (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
                (int) ne00, nb01 / sizeof(float), (int) ne01);
        } else {
            gcn_f32_matvec_rows<0><<<grid, 256, 0, ctx.stream()>>>(
                (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
                (int) ne00, nb01 / sizeof(float), (int) ne01);
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
#else
    GGML_UNUSED_VARS(ctx, src0, src1, dst, cc);
#endif
    return false;
}

bool ggml_cuda_f32_multi_ok(const ggml_tensor * mm) {
#if GCN_F32_ENABLED
    if (!ggml_cuda_f32_env_on("GFX906_F32", true)) {
        return false;
    }
    static const bool disabled = !ggml_cuda_f32_env_on("GFX906_F32_MULTI", true);
    if (disabled || mm->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    return w->type == GGML_TYPE_F32 && w->ne[0] == 2560 && w->ne[1] <= 16384 &&
        w->ne[2] == 1 && w->ne[3] == 1 &&
        w->nb[0] == sizeof(float) && w->nb[1] % 16 == 0 &&
        (!w->buffer || !ggml_backend_buft_is_host(w->buffer->buft)) &&
        mm->src[1]->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32;
#else
    GGML_UNUSED(mm);
    return false;
#endif
}

void ggml_cuda_mul_mat_f_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * mms, int n) {
#if GCN_F32_ENABLED
    GGML_ASSERT(n >= 2 && n <= 3);
    gcn_f32_multi_args args = {};
    uint32_t rows = 0;
    for (int i = 0; i < 3; i++) {
        if (i < n) {
            args.A[i]   = (const float *) mms[i]->src[0]->data;
            args.y[i]   = (float *) mms[i]->data;
            args.lda[i] = mms[i]->src[0]->nb[1] / sizeof(float);
            args.rows[i] = (uint32_t) mms[i]->src[0]->ne[1];
            args.start[i] = rows;
            rows += args.rows[i];
        } else {
            args.start[i] = UINT32_MAX;
        }
    }
    bool aligned = (uintptr_t) mms[0]->src[1]->data % 16 == 0;
    for (int k = 0; k < n; k++) {
        aligned &= (uintptr_t) mms[k]->src[0]->data % 16 == 0 && mms[k]->src[1] == mms[0]->src[1];
    }
    GGML_ASSERT(aligned);
    const int64_t ne11 = mms[0]->src[1]->ne[1];
    if (ne11 == 1) {
        gcn_f32_matvec_rows_multi<1><<<rows, 256, 0, ctx.stream()>>>(args, (const float *) mms[0]->src[1]->data);
    } else {
        GGML_ABORT("f32 multi with columns is not wired up"); // grouped nc left to would-be follow-up
    }
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, mms, n);
#endif
}
