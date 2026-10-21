#include "gfx906-hc-up-pre.cuh"
#include <cstring>
#include <cstdlib>
#include "unary.cuh"
#include "quantize.cuh"
#include "gfx906-repack/repack.cuh"
#include "ggml-impl.h"
#include "../ggml-backend-impl.h"

// [gfx906 T3] qwen4exp hyper-connection mix, after the down projection (one token):
//   SCALE -> SILU -> MUL_MAT(hc up, repacked Q8_0, K = 320) -> RESHAPE -> DSV4_HC_PRE(gated, 4 streams)
// A block quantizes silu(s*lo + b) itself (silu + fp32 scale arithmetic), computes the 4 stream rows of
// 32 output dims (the repacked flat q8_0 dot/sum order) and mixes them as our dsv4_hc_pre kernel does.
// Ported from furnace ggml-cuda/repack-gcn.cu hc_up_pre_f32 (same element order as the launches above).
//
// NON-BITWISE STATUS (E7/E8 closing, 2026-10-15):
//   - The fused chain reads the scale bias from op_params[1] exactly like the solo
//     ggml_cuda_op_scale kernel, so that known divergence source is copied byte-for-byte.
//   - The residual sensitivity: this kernel's dot product accumulates each repacked Q8_0
//     weight block in its whole 32-value order, while the canonical
//     mul_mat_vec_q8_0_repacked path splits every block into two 16-value halves and
//     reduces with a 64-lane DPP butterfly. The two are mathematically equal but round
//     differently, so greedy generations may diverge from the canonical path (~token 60
//     on some prompts). Removing the difference would require restructuring this kernel
//     to the canonical lane order, which costs the fusion's benefit -- so the whole HC
//     chain ships permanently default-on (opt-out: GFX906_HC_UP_PRE_OFF /
//     GFX906_HC_POST_NORM_OFF; the wide-batch helper stays opt-in: GFX906_SCALE_SILU;
//     all read through the tolerant parser gfx906_env_on in common.cuh).
//   - Where the fused kernel writes yq it must equal a canonical quantize_q8_1 over the
//     F32 result it just produced; the runtime emit-honesty check lives in
//     opt/diag-instrumentation (plan §15), not in this branch.
static constexpr int HC_UP_K  = 320;
static constexpr int HC_UP_NB = HC_UP_K/32;
static constexpr int HC_UP_HC = 4;
static constexpr int HC_UP_D  = 32; // output dims per block

static __global__ void __launch_bounds__(256) hc_up_pre_f32(
        const float * __restrict__ lo, const float scale, const float bias,
        const uint8_t * __restrict__ wbase, const uint32_t ne1,
        const float * __restrict__ xn, const int64_t sx1,
        const float pre_scale, float * __restrict__ dst, block_q8_1 * __restrict__ yq, const int64_t n_embd,
        const int64_t sx2) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    constexpr int ROWS = HC_UP_HC*HC_UP_D;
    __shared__ block_q8_1 aq[HC_UP_NB];
    __shared__ float      part[ROWS*HC_UP_NB];
    __shared__ float      gate[ROWS];

    const int t = threadIdx.x;
    // blockIdx.y is the token: its own lo row, xn slab, dst row and q8 row
    {
        const int it = blockIdx.y;
        lo  += (int64_t) it*HC_UP_K;
        xn  += (int64_t) it*sx2;
        dst += (int64_t) it*n_embd;
        yq   = yq ? yq + (int64_t) it*(n_embd/QK8_1) : yq;
    }

    const int4     * qs4     = reinterpret_cast<const int4 *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * HC_UP_NB * 32);
    const int64_t d0 = (int64_t) blockIdx.x * HC_UP_D;

    // 1280 (row, sub-block) items over 256 threads: all five weight loads of a thread go out
    // together, ahead of the activation quantize
    static_assert(ROWS*HC_UP_NB == 5*256, "five items per thread");
    int4     w0[5];
    int4     w1[5];
    uint16_t db[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int item = t + 256*k;
        const int r    = item / HC_UP_NB;
        const int sb   = item % HC_UP_NB;
        const size_t row = (size_t) (r / HC_UP_D) * n_embd + d0 + r % HC_UP_D;
        const size_t u   = row * HC_UP_NB + sb;
        w0[k] = qs4[u * 2 + 0];
        w1[k] = qs4[u * 2 + 1];
        db[k] = d_plane[u];
    }

    // [gfx906 T3 fix, e10] per-32-value-group quantization: each warp must own exactly one
    // 32-value QK8_1 block. The original loop (i = t; i < 320; i += 256) let lane t carry
    // a value from block (t/32) AND from block (t/32 + 8) in the same warp reduction, so
    // the amax/sum of 8 of the 10 blocks were mixed pairwise: the fused gate diverged
    // from canonical (E7's unresolved last-bit mismatch), and any emitted q8_1 failed a
    // canonical-quantize comparison (caught by the e10 emit-honesty harness, now in
    // opt/diag-instrumentation).
    // Pass 1: values 0..255 = blocks 0..7 (one per warp, lanes aligned to block lanes).
    for (int i = t; i < 256; i += blockDim.x) {
        const float v = ggml_cuda_op_silu_single(scale * lo[i] + bias);
        float amax = fabsf(v);
        float sum  = v;
        amax = warp_reduce_max(amax);
        sum  = warp_reduce_sum(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
        aq[i / QK8_1].qs[i % QK8_1] = q;
        if (i % QK8_1 == 0) {
            aq[i / QK8_1].ds = make_half2(d, sum);
        }
    }
    // Pass 2: values 256..319 = blocks 8..9 (only threads t<64; warps 0 and 1).
    if (t < 64) {
        const int i = 256 + t;
        const float v = ggml_cuda_op_silu_single(scale * lo[i] + bias);
        float amax = fabsf(v);
        float sum  = v;
        amax = warp_reduce_max(amax);
        sum  = warp_reduce_sum(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
        aq[i / QK8_1].qs[i % QK8_1] = q;
        if (i % QK8_1 == 0) {
            aq[i / QK8_1].ds = make_half2(d, sum);
        }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int item = t + 256*k;
        const int sb   = item % HC_UP_NB;
        const block_q8_1 * xb = aq + sb;
        const int4 * x4 = reinterpret_cast<const int4 *>(xb->qs);
        const int4 a0 = x4[0];
        const int4 a1 = x4[1];
        int idot = 0;
        idot = ggml_cuda_dp4a(w0[k].x, a0.x, idot); idot = ggml_cuda_dp4a(w0[k].y, a0.y, idot);
        idot = ggml_cuda_dp4a(w0[k].z, a0.z, idot); idot = ggml_cuda_dp4a(w0[k].w, a0.w, idot);
        idot = ggml_cuda_dp4a(w1[k].x, a1.x, idot); idot = ggml_cuda_dp4a(w1[k].y, a1.y, idot);
        idot = ggml_cuda_dp4a(w1[k].z, a1.z, idot); idot = ggml_cuda_dp4a(w1[k].w, a1.w, idot);
        part[item] = __half2float(*reinterpret_cast<const __half *>(&db[k])) * __low2float(xb->ds) * (float) idot;
    }
    __syncthreads();

    if (t < ROWS) {
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < HC_UP_NB; j++) {
            sum += part[t * HC_UP_NB + j];
        }
        gate[t] = sum;
    }
    __syncthreads();

    if (t < HC_UP_D) {
        const int64_t i0 = d0 + t;
        float sum = 0.0f;
        for (int ih = 0; ih < HC_UP_HC; ++ih) {
            const float xv = xn[i0 + ih*sx1];
            const float wv = 1.0f / (1.0f + expf(-gate[ih*HC_UP_D + t]));
            sum += xv * wv;
        }
        const float v = pre_scale * sum;
        dst[i0] = v;
        if (yq != nullptr) {
            float amax = fabsf(v);
            float bsum = v;
            amax = warp_reduce_max(amax);
            bsum = warp_reduce_sum(bsum);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
            yq[i0 / QK8_1].qs[i0 % QK8_1] = q;
            if (i0 % QK8_1 == 0) {
                yq[i0 / QK8_1].ds = make_half2(d, bsum);
            }
        }
    }
#else
    GGML_UNUSED_VARS(lo, scale, bias, wbase, ne1, xn, sx1, pre_scale, dst, yq, n_embd, sx2);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(__gfx906__)
}

// host side ---------------------------------------------------------------

bool ggml_cuda_hc_up_pre_ok(const ggml_tensor * scale, const ggml_tensor * silu, const ggml_tensor * up, const ggml_tensor * pre) {
    // [gfx906 T3] single switch now: GFX906_HC_UP_PRE_OFF=1 turns the fusion off
    // (tolerant parse: 1/on/true/yes = off, 0/off/false/no/unset = on), so there is
    // exactly one way to disable it. The legacy opt-in spellings of this switch are
    // retired (no aliases, plan §3.1.1).
    static const bool disabled = gfx906_env_on("GFX906_HC_UP_PRE_OFF");
    const ggml_tensor * w  = up->src[0];
    const ggml_tensor * lo = scale->src[0];
    if (disabled || !w->buffer || !ggml_backend_buft_is_cuda_repack(w->buffer->buft) || w->type != GGML_TYPE_Q8_0 ||
            w->ne[0] != HC_UP_K || w->ne[2] != 1 || w->ne[3] != 1 || up->src[1] != silu) {
        return false;
    }
    const bool gated = ggml_get_op_params_i32(pre, 1) != 0;
    const int64_t n_embd = pre->ne[0];
    const int64_t n_tok  = pre->ne[1]; // one row per token (single-token decode; verify batches up to 16)
    const ggml_tensor * xn = pre->src[0];
    return gated && n_tok >= 1 && n_tok <= 16 && lo->type == GGML_TYPE_F32 && ggml_is_contiguous(lo) &&
        ggml_nelements(lo) == HC_UP_K*n_tok &&
        up->type == GGML_TYPE_F32 && ggml_nelements(up) == w->ne[1]*n_tok &&
        xn->type == GGML_TYPE_F32 && xn->ne[0] == n_embd && xn->ne[1] == HC_UP_HC && xn->ne[2] == n_tok &&
        xn->ne[3] == 1 && xn->nb[0] == sizeof(float) &&
        w->ne[1] == HC_UP_HC*n_embd && n_embd % HC_UP_D == 0 && pre->ne[2] == 1 && pre->ne[3] == 1 &&
        pre->type == GGML_TYPE_F32 && ggml_is_contiguous(pre) && pre->src[1]->ne[1] == HC_UP_HC &&
        silu->type == GGML_TYPE_F32 && scale->op == GGML_OP_SCALE;
}

// launch the fused chain; yq (optional): also emit the result as flat q8_1 blocks
// (see ggml_cuda_hc_up_pre_emit_target) so the downstream mat-vec skips its quantize
void ggml_cuda_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, const ggml_tensor * up,
        ggml_tensor * pre, void * yq) {
    // read the bias bytes exactly as the solo ggml_cuda_op_scale kernel does
    // (op_params[1]: zero in most arena layouts, but reproduced, not assumed)
    const float  s         = ggml_get_op_params_f32(scale, 0);
    float        bias      = 0.0f;
    memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));
    const float  pre_scale = ggml_get_op_params_f32(pre, 0);
    const ggml_tensor * xn = pre->src[0];
    const int64_t n_embd = pre->ne[0];
    hc_up_pre_f32<<<dim3(n_embd / HC_UP_D, pre->ne[1], 1), 256, 0, ctx.stream()>>>(
        (const float *) scale->src[0]->data, s, bias, (const uint8_t *) up->src[0]->data, (uint32_t) up->src[0]->ne[1],
        (const float *) xn->data, xn->nb[1] / sizeof(float), pre_scale, (float *) pre->data, (block_q8_1 *) yq, n_embd,
        xn->nb[2] / sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}

void * ggml_cuda_hc_up_pre_emit_target(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const ggml_tensor * t) {
    if (!cgraph || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->data == nullptr ||
            ggml_nelements(t) % QK8_1 != 0) {
        return nullptr;
    }
    // find a repack-route MUL_MAT whose flat src1 aliases t exactly (plain MM only;
    // MUL_MAT_ID reads padded expert layouts which the producer kernel cannot write)
    const ggml_tensor * x = nullptr;
    for (int i = 0; i < cgraph->n_nodes && x == nullptr; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_MUL_MAT || !n->src[0]->buffer || !ggml_backend_buft_is_cuda_repack(n->src[0]->buffer->buft)) {
            continue;
        }
        const ggml_tensor * xs = n->src[1];
        const ggml_tensor * rx = xs; while (rx->view_src) { rx = rx->view_src; }
        const ggml_tensor * rt = t;  while (rt->view_src)  { rt = rt->view_src;  }
        if (rx == rt && xs->data == t->data && xs->type == GGML_TYPE_F32 && ggml_is_contiguous(xs) &&
                xs->ne[2] == 1 && xs->ne[3] == 1 && xs->ne[0] % MATRIX_ROW_PADDING == 0 &&
                xs->ne[0]*xs->ne[1] <= ggml_nelements(t) &&
                GGML_PAD(xs->ne[0], MATRIX_ROW_PADDING)/QK8_1 * xs->ne[1] == ggml_nelements(t)/QK8_1) {
            x = xs; // emit covers the full consumer rows: no padding gap left unwritten
        }
    }
    if (x == nullptr) {
        return nullptr;
    }
    bool capturing = false;
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus st;
    if (hipStreamIsCapturing(ctx.stream(), &st) == hipSuccess) {
        capturing = st != hipStreamCaptureStatusNone;
    }
#else
    cudaStreamCaptureStatus st;
    if (cudaStreamIsCapturing(ctx.stream(), &st) == cudaSuccess) {
        capturing = st != cudaStreamCaptureStatusNone;
    }
#endif
    const size_t bytes = (size_t) ggml_nelements(t)/QK8_1 * sizeof(block_q8_1);
    auto & e = ctx.repack_xq[ctx.repack_xq_next];
    ctx.repack_xq_next = (ctx.repack_xq_next + 1) % ggml_backend_cuda_context::REPACK_XQ_CACHE_N;
    e.gen = 0;
    if (e.cap < bytes) {
        if (capturing) {
            return nullptr; // consumer falls back to its own quantize
        }
        if (e.buf != nullptr) {
            ctx.repack_xq_retired.push_back(e.buf);
        }
        CUDA_CHECK(cudaMalloc(&e.buf, bytes));
        e.cap = bytes;
    }
    e.gen              = ctx.graph_gen;
    e.data             = x->data;
    memcpy(e.ne, x->ne, sizeof(e.ne));
    for (int d = 1; d < 4; ++d) {
        e.nb[d] = x->nb[d];
    }
    e.bytes            = bytes;
    // producer metadata = t (what the fused kernel writes on the F32 side), so that the
    // graph loop's per-node range invalidation keeps this entry honest
    e.producer         = t;
    e.producer_data    = t->data;
    e.producer_nbytes  = (size_t) ggml_nbytes(t);
    return e.buf;
}

// [gfx906 T3] SCALE -> SILU (qwen4exp hc mix, wider batches where hc_up_pre does
// not apply): one pass, same elementwise arithmetic as the two separate launches
static __global__ void scale_silu_f32(const float * x, float * dst, const float scale, const float bias, const int64_t nelements) {
    int64_t tid = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    int64_t stride = (int64_t) blockDim.x * (int64_t) gridDim.x;

    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = ggml_cuda_op_silu_single(scale * x[i] + bias);
    }
}

void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * scale_node, ggml_tensor * silu_node) {
    const ggml_tensor * src0 = scale_node->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && silu_node->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(silu_node));
    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(silu_node));

    float scale;
    float bias;
    memcpy(&scale, (float *) scale_node->op_params + 0, sizeof(float)); // exactly as ggml_cuda_op_scale
    memcpy(&bias,  (float *) scale_node->op_params + 1, sizeof(float));
    const int64_t nelements = ggml_nelements(src0);
    const int64_t num_blocks = (nelements + 256 - 1) / 256;
    scale_silu_f32<<<MIN(0x7FFFFFFF, num_blocks), 256, 0, ctx.stream()>>>(
        (const float *) src0->data, (float *) silu_node->data, scale, bias, nelements);
    CUDA_CHECK(cudaGetLastError());
}
