// Grouped repacked-Q8_0 mat-vec (ported to our layout).
// Up to three dense MUL_MAT nodes that share the same F32 activation (same src1
// pointer) and the same ne00 are collected by the graph scan in ggml-cuda.cu and
// served by ONE launch: per row the arithmetic is identical to
// mul_mat_vec_q8_0_repacked<16,1,false> (one warp, LANES=64, no fusion), so
// results stay bitwise equal to the ungrouped path. Adapted from the
// gfx906-perf/furnace family (q8_multi); the weight layout is our repack
// (repack_qs_stride), not the fork's.
#include "repack.cuh"
#include "repack-common.cuh"
#include "repack-kernels.cuh"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

struct q8_multi_args {
    const uint8_t * w[3];
    float *         y[3];
    uint32_t        ne1[3];
    uint32_t        start[3];
};

static __global__ void __launch_bounds__(64) mul_mat_vec_q8_0_repacked_multi(
        const q8_multi_args args, const block_q8_1 * __restrict__ xq, const uint32_t ne0) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const int t = blockIdx.x >= args.start[2] ? 2 : (blockIdx.x >= args.start[1] ? 1 : 0);
    const uint8_t * wbase = args.w[t];
    const uint32_t  ne1   = args.ne1[t];
    const uint32_t  row   = blockIdx.x - args.start[t];

    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t qs_str   = repack_qs_stride(ne0); // bytes per row

    const uint4    * w_row  = reinterpret_cast<const uint4 *>(wbase + (size_t) row * qs_str);
    const uint16_t * d_row  = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * qs_str) + (size_t) row * n_blocks;

    const int lane = threadIdx.x;
    const uint32_t n_half = n_blocks * 2;

    float acc = 0.0f;
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        const uint32_t half = hb & 1;
        const block_q8_1 * xb = xq + (hb >> 1);
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;
        const int x0 = xq32[0], x1 = xq32[1], x2 = xq32[2], x3 = xq32[3];
        const uint4 wv = w_row[hb];
        const uint16_t db = d_row[hb >> 1];
        const float dw = __half2float(*reinterpret_cast<const __half *>(&db));
        acc += dw * dx * (float) mmvq_dp4a_u4(wv, x0, x1, x2, x3);
    }
    const float a = rp_warp_reduce_sum<64>(acc);
    if (lane == 0) {
        args.y[t][row] = a;
    }
#else
    GGML_UNUSED_VARS(args, xq, ne0);
    NO_DEVICE_CODE;
#endif
}

// Unrolled variant: every load issued before the first dot. ITERS = half
// sub-blocks per lane; lanes past the row end re-read its first half with a
// zero activation scale (identical acc contribution but a real load).
template <int ITERS>
static __global__ void __launch_bounds__(64) mul_mat_vec_q8_0_repacked_multi_u(
        const q8_multi_args args, const block_q8_1 * __restrict__ xq, const uint32_t ne0) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const int t = blockIdx.x >= args.start[2] ? 2 : (blockIdx.x >= args.start[1] ? 1 : 0);
    const uint8_t * wbase = args.w[t];
    const uint32_t  ne1   = args.ne1[t];
    const uint32_t  row   = blockIdx.x - args.start[t];

    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t qs_str   = repack_qs_stride(ne0);

    const uint4    * w_row  = reinterpret_cast<const uint4 *>(wbase + (size_t) row * qs_str);
    const uint16_t * d_row  = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * qs_str) + (size_t) row * n_blocks;

    const uint32_t lane   = threadIdx.x;
    const uint32_t n_half = n_blocks * 2;

    uint4     w[ITERS];
    int4      xv[ITERS];
    uint16_t  db[ITERS];
    float     dx[ITERS];
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        const uint32_t hb  = lane + j * 64;
        const bool     ok  = hb < n_half;
        const uint32_t hbc = ok ? hb : 0u;
        w[j]  = w_row[hbc];
        db[j] = d_row[hbc >> 1];
        const block_q8_1 * xb = xq + (hbc >> 1);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs) + (hbc & 1) * 4;
        xv[j] = make_int4(xq32[0], xq32[1], xq32[2], xq32[3]);
        dx[j] = ok ? __low2float(xb->ds) : 0.0f;
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        const int idot = mmvq_dp4a_u4(w[j], xv[j].x, xv[j].y, xv[j].z, xv[j].w);
        acc += __half2float(*reinterpret_cast<const __half *>(&db[j])) * dx[j] * (float) idot;
    }
    const float a = rp_warp_reduce_sum<64>(acc);
    if (lane == 0) {
        args.y[t][row] = a;
    }
#else
    GGML_UNUSED_VARS(args, xq, ne0);
    NO_DEVICE_CODE;
#endif
}

bool ggml_cuda_repack_multi_ok(const ggml_tensor * mm) {
    static const bool disabled = getenv("GFX906_Q8_MULTI_OFF") != nullptr;
    if (disabled || mm->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    if (w == nullptr || w->view_src != nullptr) {
        return false;
    }
    if (!ggml_cuda_repack_mul_mat_should_fire(w)) {
        return false;
    }
    if (w->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if (mm->src[1]->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32) {
        return false;
    }
    if (mm->src[1]->ne[1] != 1) { // one-token decode only for this first cut
        return false;
    }
    if (!ggml_is_contiguous(mm)) {
        return false;
    }
    return true;
}

static void q8_multi_log(ggml_tensor * const * mms, int n) {
    static std::mutex mtx;
    static std::set<std::string> seen;
    std::string key;
    for (int k = 0; k < n; k++) {
        key += std::string(mms[k]->name) + "|";
    }
    std::lock_guard<std::mutex> lck(mtx);
    if (seen.insert(key).second && seen.size() <= 24) {
        fprintf(stderr, "[q8-multi] n=%d :", n);
        for (int k = 0; k < n; k++) {
            fprintf(stderr, " %s(0x%x rows, ne0=0x%x)", mms[k]->name,
                (uint32_t) mms[k]->src[0]->ne[1], (uint32_t) mms[k]->src[0]->ne[0]);
        }
        fprintf(stderr, "\n");
    }
}

void ggml_cuda_mul_mat_repacked_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * mms, int n) {
    GGML_ASSERT(n >= 2 && n <= 3);

    const ggml_tensor * src1 = mms[0]->src[1];
    const int64_t ne00 = mms[0]->src[0]->ne[0];
    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_q8_1> src1_q8_1_own;
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);
    block_q8_1 * xq = repack_quantize_src1_q8_1(ctx, mms[0]->src[0], src1, ne00, ne10_padded,
        s11, s12, s13, src1->ne[1], src1->ne[2], src1->ne[3],
        (size_t) (src1->ne[3] * src1->ne[2] * src1->ne[1] * (ne10_padded / QK8_1)), src1_q8_1_own, stream);

    q8_multi_args args = {};
    uint32_t rows = 0;
    for (int i = 0; i < n; i++) {   // geometry guards (fail loud rather than corrupt)
        const ggml_tensor * mm = mms[i];
        GGML_ASSERT(mm->src[1]->ne[0] == ne00);
        GGML_ASSERT(mm->src[1]->nb[0] == sizeof(float));
        GGML_ASSERT(ggml_is_contiguous(mm->src[1]) || mm->src[1]->ne[1] == 1);
        GGML_ASSERT(mm->type == GGML_TYPE_F32);
        GGML_ASSERT(mm->nb[0] == sizeof(float));
        GGML_ASSERT(mm->nb[1] == (size_t) mm->ne[0] * sizeof(float));
        GGML_ASSERT(mm->ne[2] == 1 && mm->ne[3] == 1);
        GGML_ASSERT(mms[i]->src[1] == src1);
    }
    for (int i = 0; i < 3; i++) {
        if (i < n) {
            args.w[i]   = (const uint8_t *) mms[i]->src[0]->data;
            args.y[i]   = (float *) mms[i]->data;
            args.ne1[i] = (uint32_t) mms[i]->src[0]->ne[1];
            args.start[i] = rows;
            rows += args.ne1[i];
        } else {
            args.start[i] = UINT32_MAX;
        }
    }
    q8_multi_log(mms, n);

    static const bool no_unroll = getenv("GFX906_Q8_MULTI_NO_UNROLL") != nullptr;
    static const bool no_triples = getenv("GFX906_Q8_MULTI_NO_TRIPLES") != nullptr;
    if (no_triples && n == 3) { GGML_ABORT("q8-multi: triple groups disabled but one ran"); }
    // n_half = 2 * ne00/32 (16-weight half sub-blocks). The old `ne00/8` here was 2x
    // oversized (E8 latency note): ITERS picked >=2x the real need — either falling
    // off the unrolled ladder entirely or inflating register/wasted load work.
    const uint32_t n_half = (uint32_t)(ne00 / 16);
    const uint32_t iters  = (n_half + 63) / 64;      // half sub-blocks per lane
    if (!no_unroll && iters >= 1 && iters <= 4) {
        switch (iters) {
            case 1:  mul_mat_vec_q8_0_repacked_multi_u<1><<<rows, 64, 0, stream>>>(args, xq, (uint32_t) ne00); break;
            case 2:  mul_mat_vec_q8_0_repacked_multi_u<2><<<rows, 64, 0, stream>>>(args, xq, (uint32_t) ne00); break;
            case 3:  mul_mat_vec_q8_0_repacked_multi_u<3><<<rows, 64, 0, stream>>>(args, xq, (uint32_t) ne00); break;
            default: mul_mat_vec_q8_0_repacked_multi_u<4><<<rows, 64, 0, stream>>>(args, xq, (uint32_t) ne00); break;
        }
    } else {
        mul_mat_vec_q8_0_repacked_multi<<<rows, 64, 0, stream>>>(args, xq, (uint32_t) ne00);
    }
    CUDA_CHECK(cudaGetLastError());
    // (HIP-only project: q8_repack/ always builds with HIP; the MUSA path goes
    // through the repack.cuh inline stubs and never reaches this TU.)
}
