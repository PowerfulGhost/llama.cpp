// Dense (non-MoE) entry point for repacked Q8_0 weights: quantize src1, then dispatch
// the mat-vec (single token) or tiled GEMM (multi-token) path per 2D slice.
#include "repack.cuh"
#include "repack-common.cuh"
#include "repack-kernels.cuh"
#include "../quantize.cuh"
#include "../mmq.cuh"

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, int64_t ne00, int64_t ne01, int64_t ne11,
        cudaStream_t stream);
template <ggml_type WT>
static void ggml_cuda_mul_mat_repacked_nc_t(
        const uint8_t * w, const block_q8_1 * xq, float * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const uint32_t xs, const uint32_t ys, cudaStream_t stream);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();

    // Views are re-packed into a persistent cache buffer (pool allocs are invalid
    // inside CUDA graph capture).
    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;

    if (ne11 >= 1 && ne11 <= MMQ_RP_Q8_MMV_MAX_TOKENS) {
        // One token, or a narrow batch: plain Q8_1 rows, one weight pass. The
        // tiled path below computes a full 32-wide tile whatever the batch is.
        ggml_cuda_pool_alloc<block_q8_1> src1_q8_1_own;
        block_q8_1 * src1_q8_1_d;
        {
            const int64_t s11 = src1->nb[1] / sizeof(float);
            const int64_t s12 = src1->nb[2] / sizeof(float);
            const int64_t s13 = src1->nb[3] / sizeof(float);
            src1_q8_1_d = repack_quantize_src1_q8_1(ctx, src0, src1, ne10, ne10_padded,
                s11, s12, s13, ne11, ne12, ne13,
                (size_t) (ne13 * ne12 * ne11 * x_stride), src1_q8_1_own, stream);
        }
        const uint32_t dst_s1 = dst->nb[1] / sizeof(float);
        for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const block_q8_1 * xq = src1_q8_1_d
                              + (i3 * ne12 + i2) * ne11 * x_stride;
            float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
            if (ne11 == 1) {
                ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
                    ne00, ne01, ne11, stream);
            } else {
                switch (src0->type) {
                    case GGML_TYPE_Q8_0:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q8_0>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_MXFP4:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_MXFP4>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_IQ4_NL:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_IQ4_NL>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_Q5_1:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q5_1>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_Q5_K:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q5_K>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_Q4_K:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q4_K>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_Q6_K:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q6_K>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    default: GGML_ABORT("unsupported repack type");
                }
            }
        }
        }
        return;
    }

    // Multi-token: grouped MMQ Q8_1 layout for the tiled GEMM.
    const uint64_t n_groups   = (uint64_t) ne10_padded / (4 * QK8_1);

    // [gfx906-opt step 1] The fork divides the columns into chunks under VRAM pressure, because it
    // also competes with its q8_1 activation cache. Neither that cache nor the retry allocation is
    // ported yet, so the full workspace is allocated up front (a few MiB at prefill widths).
    const int64_t chunk_ne11 = ne11;
    ggml_cuda_pool_alloc<block_q8_1_mmq_h> src1_q8_1(ctx.pool());
    src1_q8_1.alloc(ne13 * ne12 * chunk_ne11 * n_groups);

    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);
    for (int64_t col = 0; col < ne11; col += chunk_ne11) {
        const int64_t iter_ne11 = std::min(chunk_ne11, ne11 - col);
        quantize_mmq_q8_1_cuda((const float *) src1->data + col*s11, nullptr, src1_q8_1.get(),
            src0->type, ne10, s11, s12, s13, ne10_padded, iter_ne11, ne12, ne13, stream);

        for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const block_q8_1_mmq_h * slice_base = src1_q8_1.get()
                                  + (i3 * ne12 + i2) * iter_ne11 * n_groups;
            const block_q8_1 * xq = reinterpret_cast<const block_q8_1 *>(slice_base);
            float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2])
                           + col*dst_s1;
            ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
                ne00, ne01, iter_ne11, stream);
        }
        }
    }
}

// Narrow batch (2..8 tokens): one multi-column mat-vec pass per slice.
template <ggml_type WT>
static void ggml_cuda_mul_mat_repacked_nc_t(
        const uint8_t * w, const block_q8_1 * xq, float * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const uint32_t xs, const uint32_t ys, cudaStream_t stream) {
    // Geometry per width, measured on Qwen3.8-27B (pp512 at that ubatch, 4x
    // MI50 -sm tensor). The row count per lane is small once the tensor is
    // split, so scheduling granularity beats per-group reuse and 64-thread
    // workgroups spread over the CUs win. Two rows per lane reuse the
    // activation block across rows and pay from 3 tokens up; at 2 there is
    // too little to amortize, and 4 rows spills.
    // Lane width per row, chosen from the ROW LENGTH and the batch width.
    //
    // LANES is the number of lanes cooperating on one row. The strided loop
    // runs ne0/32/LANES iterations and is followed by a log2(LANES)-step DPP
    // reduction, so the useful knob is how much work a lane gets before it has
    // to reduce. Narrowing the lane group raises that, and raises occupancy,
    // but it also cuts the block count that hides memory latency.
    //
    // Which effect wins is decided by the row length, and two models disagree
    // about the batch width alone, so that is the wrong axis. Measured pp512
    // at the ubatch, 2x MI50 -sm layer, narrow-lane against the 64-lane shape:
    //
    //                       ne0    K-blocks   width 2   width 3
    //   Qwen3.6-27B        5120       160      +13.8%    +2.1%
    //   Qwen3.6-35B-A3B    2048        64       -9.5%    -0.7%
    //
    // A 2048-wide row is 64 K-blocks, so splitting it over 16 lanes leaves 4
    // iterations against a 4-step reduction while cutting the grid by 16x -
    // the row is simply too short to pay for the lost parallelism. Gate on the
    // block count so the choice follows the shape and not the model.
    // Narrowing the lane group raises the work a lane does before it reduces,
    // and multiplies the rows a block covers, which shrinks the grid. Both
    // have a floor and BOTH must hold.
    //
    // A lane accumulates n_blocks/LANES times, then pays a log2(LANES)-step
    // reduction, so require at least four accumulations at the widest group we
    // would narrow to. Measured on two models: 160 and 128 blocks win, 64
    // loses at every lane width, so the break is between them.
    //
    // The grid condition is what the row length alone cannot see - a 48-row
    // tensor is a 2-block grid however long its rows are.
    constexpr int RP_NC_MIN_ACCUM   = 4;
    constexpr int RP_NC_LANES_WIDE  = 32;   // widest group we ever narrow to
    const int     nsm       = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int64_t n_blocks  = ne00 / 32;
    const int64_t min_grid  = 2LL * nsm;
    const bool    work_ok   = n_blocks >= RP_NC_MIN_ACCUM * RP_NC_LANES_WIDE;
    const bool    lanes16_ok = work_ok && (ne01 + 31) / 32 >= min_grid;
    const bool    lanes32_ok = work_ok && (ne01 +  7) /  8 >= min_grid;
    switch (ne11) {
        case 2: {
            if (lanes16_ok) {
                // 16 lanes to a row, 32 rows per block, one row per lane.
                const dim3 grid((ne01 + 31) / 32, 1, 1);
                mul_mat_vec_repacked_nc<32, 8, 2, 1, 16, WT><<<grid, 512, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            } else {
                const dim3 grid((ne01 + 1) / 2, 1, 1);
                mul_mat_vec_repacked_nc<2, 2, 2, 1, 64, WT><<<grid, 128, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            }
        } break;
        case 3: {
            if (lanes32_ok) {
                // 32 lanes to a row, 8 rows per block, one row per lane.
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_repacked_nc<8, 4, 3, 1, 32, WT><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            } else {
                const dim3 grid((ne01 + 1) / 2, 1, 1);
                mul_mat_vec_repacked_nc<2, 1, 3, 2, 64, WT><<<grid, 64, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            }
        } break;
        case 4: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 4, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 5: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 5, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 6: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 6, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 7: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 7, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 8: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 8, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        default: GGML_ABORT("nc mat-vec: unsupported batch width");
    }
}

// [gfx906 E16.6] furnace dense-Q8_0 short/long kernels (flat/seg/splitk/rowu) ported
// onto OUR Q8_0 repack layout (qs rows at repack_qs_stride, d plane at ne1*pad then
// row*n_blocks+sb): flat: K=320 (hc_up), seg: K<=1024 (shexp 640x2560), splitk:
// few long rows (hc_downs 320x10240), rowu: K%2048==0 && K<=8192 (ssm_out 6144x2560).
// A/B via GFX906_Q8_SHAPE_MMV=x (furnace-exact ladder) vs default 'g'.
static __global__ void __launch_bounds__(256) gfx906_q8_flat(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    constexpr int NB = 10, R = 256 / NB;
    const int4     * qs4 = reinterpret_cast<const int4 *>(wbase);
    const uint16_t * d_row = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * ((ne0 % 128 == 0) ? (size_t) ne0 + 16 : (size_t) ne0));
    __shared__ float part[R * NB];
    const int t   = threadIdx.x;
    const uint32_t row0 = blockIdx.x * R;
    if (t < R * NB) {
        const uint32_t row = row0 + t / NB;
        const int      sb  = t % NB;
        float acc = 0.0f;
        if (row < ne1) {
            const block_q8_1 * xb = xq + sb;
            const int4 * x4 = reinterpret_cast<const int4 *>(xb->qs);
            const size_t u = (size_t) row * NB + sb;
            const int4 w0 = qs4[u * 2 + 0];
            const int4 w1 = qs4[u * 2 + 1];
            const int4 a0 = x4[0];
            const int4 a1 = x4[1];
            int idot = 0;
            idot = ggml_cuda_dp4a(w0.x, a0.x, idot); idot = ggml_cuda_dp4a(w0.y, a0.y, idot);
            idot = ggml_cuda_dp4a(w0.z, a0.z, idot); idot = ggml_cuda_dp4a(w0.w, a0.w, idot);
            idot = ggml_cuda_dp4a(w1.x, a1.x, idot); idot = ggml_cuda_dp4a(w1.y, a1.y, idot);
            idot = ggml_cuda_dp4a(w1.z, a1.z, idot); idot = ggml_cuda_dp4a(w1.w, a1.w, idot);
            const uint16_t db = *(d_row + (size_t) row * (ne0 >> 5) + sb);
            acc = __half2float(*reinterpret_cast<const __half *>(&db)) * __low2float(xb->ds) * (float) idot;
        }
        part[t] = acc;
    }
    __syncthreads();
    if (t < R && row0 + t < ne1) {
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < NB; j++) {
            sum += part[t * NB + j];
        }
        y[row0 + t] = sum;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1);
    NO_DEVICE_CODE;
#endif
}

template <int SEG>
static __global__ void __launch_bounds__(256) gfx906_q8_seg(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    constexpr int ROWS_PER_BLOCK = 256 / SEG;
    const uint32_t n_blocks = ne0 >> 5;
    const int4     * qs4   = reinterpret_cast<const int4 *>(wbase);
    const uint16_t * d_row = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * ((ne0 % 128 == 0) ? (size_t) ne0 + 16 : (size_t) ne0));
    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / SEG;
    const uint32_t sb  = threadIdx.x % SEG;
    float acc = 0.0f;
    if (row < ne1 && sb < n_blocks) {
        // our layout de-aliases qs rows by +16 B when ne0 % 128 == 0
        const size_t rs   = (ne0 % 128 == 0) ? (size_t) ne0 + 16 : (size_t) ne0;
        const int4 * roww = reinterpret_cast<const int4 *>(wbase + (size_t) row * rs + (size_t) sb * 32);
        const int4 w0 = roww[0];
        const int4 w1 = roww[1];
        const block_q8_1 * xb = xq + sb;
        const int4 * x4 = reinterpret_cast<const int4 *>(xb->qs);
        const int4 a0 = x4[0];
        const int4 a1 = x4[1];
        int idot = 0;
        idot = ggml_cuda_dp4a(w0.x, a0.x, idot);
        idot = ggml_cuda_dp4a(w0.y, a0.y, idot);
        idot = ggml_cuda_dp4a(w0.z, a0.z, idot);
        idot = ggml_cuda_dp4a(w0.w, a0.w, idot);
        idot = ggml_cuda_dp4a(w1.x, a1.x, idot);
        idot = ggml_cuda_dp4a(w1.y, a1.y, idot);
        idot = ggml_cuda_dp4a(w1.z, a1.z, idot);
        idot = ggml_cuda_dp4a(w1.w, a1.w, idot);
        const uint16_t db = *(d_row + (size_t) row * n_blocks + sb);
        acc = __half2float(*reinterpret_cast<const __half *>(&db)) * __low2float(xb->ds) * (float) idot;
    }
    acc = warp_reduce_sum<SEG>(acc);
    if (sb == 0 && row < ne1) {
        y[row] = acc;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1);
    NO_DEVICE_CODE;
#endif
}

static __global__ void gfx906_q8_verify_diff(const float * a, const float * b, int n) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    __shared__ float mx[256];
    const int t = threadIdx.x;
    float m = 0.0f, s0 = 0.0f, s1 = 0.0f;
    for (int i = t; i < n; i += 256) {
        float d = fabsf(a[i] - b[i]);
        m = fmaxf(m, d);
        s0 += a[i]; s1 += b[i];
    }
    mx[t] = m;
#pragma unroll
    for (int o = 128; o > 0; o >>= 1) mx[t] = fmaxf(mx[t], mx[t ^ o]);
    if (t == 0) {
        printf("[splitk-verify] n=%d maxdiff=%.6e sum_a=%.4f sum_b=%.4f rel=%.3e\n",
               n, (double) mx[0], (double) s0, (double) s1,
               (double) (fabsf(s0 - s1) / (fabsf(s1) + 1e-30f)));
    }
#else
    GGML_UNUSED_VARS(a, b, n);
#endif
}

static __global__ void __launch_bounds__(256) gfx906_q8_splitk(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const uint32_t n_blocks = ne0 >> 5;
    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_row   = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * ((ne0 % 128 == 0) ? (size_t) ne0 + 16 : (size_t) ne0));
    const uint32_t row = blockIdx.x;
    const size_t rs = (ne0 % 128 == 0) ? (size_t) ne0 + 16 : (size_t) ne0;

    float acc = 0.0f;
    for (uint32_t hb = threadIdx.x; hb < n_blocks * 2; hb += 256) {
        const uint32_t sb = hb >> 1, half = hb & 1;
        const block_q8_1 * xb = xq + sb;
        const int * xq32  = reinterpret_cast<const int *>(xb->qs) + half * 4;
        const int * w_int = qs_int + (size_t) row * (rs >> 2) + sb * 8 + half * 4;
        int idot = 0;
#pragma unroll
        for (int g = 0; g < 4; g++) {
            idot = ggml_cuda_dp4a(w_int[g], xq32[g], idot);
        }
        const uint16_t db = *(d_row + (size_t) row * n_blocks + sb);
        acc += __half2float(*reinterpret_cast<const __half *>(&db)) * __low2float(xb->ds) * (float) idot;
    }
    acc = warp_reduce_sum<64>(acc);
    __shared__ float part[4];
    if ((threadIdx.x & 63) == 0) {
        part[threadIdx.x >> 6] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        y[row] = part[0] + part[1] + part[2] + part[3];
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1);
    NO_DEVICE_CODE;
#endif
}

template <int LANES, int ITERS>
static __global__ void __launch_bounds__(64) gfx906_q8_rowu(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const uint32_t n_blocks = ne0 >> 5;
    const int lane = threadIdx.x % LANES;
    const uint32_t row = blockIdx.x * (64 / LANES) + threadIdx.x / LANES;
    const bool valid = row < ne1;
    const uint32_t rr = valid ? row : 0;
    const size_t rs = ((ne0 % 128 == 0) ? (size_t) ne0 + 16 : (size_t) ne0);
    const int4     * qs4 = reinterpret_cast<const int4 *>(wbase);
    const uint16_t * d_row = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * ((ne0 % 128 == 0) ? (size_t) ne0 + 16 : (size_t) ne0));

    int4     w[ITERS];
    int4     xv[ITERS];
    uint16_t db[ITERS];
    float    dx[ITERS];
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        const uint32_t hb = lane + j * LANES;
        const uint32_t sb = hb >> 1, half = hb & 1;
        w[j]  = *(qs4 + (size_t) rr * (rs >> 4) + sb * 2 + (size_t) half);
        db[j] = *(d_row + (size_t) rr * n_blocks + sb);
        const block_q8_1 * xb = xq + sb;
        xv[j] = *reinterpret_cast<const int4 *>(reinterpret_cast<const int *>(xb->qs) + half * 4);
        dx[j] = __low2float(xb->ds);
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        int idot = 0;
        idot = ggml_cuda_dp4a(w[j].x, xv[j].x, idot);
        idot = ggml_cuda_dp4a(w[j].y, xv[j].y, idot);
        idot = ggml_cuda_dp4a(w[j].z, xv[j].z, idot);
        idot = ggml_cuda_dp4a(w[j].w, xv[j].w, idot);
        acc += __half2float(*reinterpret_cast<const __half *>(&db[j])) * dx[j] * (float) idot;
    }
    acc = warp_reduce_sum<LANES>(acc);
    if (lane == 0 && valid) {
        y[row] = acc;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1);
    NO_DEVICE_CODE;
#endif
}

// Dispatch one 2D slice: mat-vec vs tiled GEMM.
static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, const int64_t ne00, const int64_t ne01, const int64_t ne11,
        cudaStream_t stream) {
    if (ne11 == 1) {
        switch (src0->type) {
            case GGML_TYPE_Q8_0: {
                // [gfx906-opt P5] shape->mapping ladder for the plain dense c=1 mat-vec.
                // Default 'g': every probe shape (K>1024) uses the one-wave-per-row mapping
                // <1,1> (64 threads, grid=ne01) — measured tg128 39.50/39.48 vs legacy
                // <16,16> (1024 threads, 16 rows/block) 39.08/39.00 = +1.3%, and **bitwise
                // identical** (greedy 5x3 md5 278be07e == legacy arm; WPR==1 keeps each
                // row's half-block partition unchanged).
                // 'w' = legacy <16,16> (A/B arm). 'f' (furnace ladder: rows<512 -> <2,4>)
                //                and 'o' (**always** <2,4>) are usable since the WPR>1 "corruption"
                // was located: it was never the s_part shared-memory reduction — it was the
                // dispatch grid. Our template semantics are NOT furnace's: here <ROWS, NWAVES>
                // covers ROWS rows per block (WPR = NWARPS/ROWS warps K-split per row), while
                // furnace's <ROWS, NWAVES> covers ROWS*NWAVES rows per block. P5 copied
                // furnace's grid ((ne01+7)/8 = ROWS*NWAVES=8 rows/block) onto our 2-row/block
                // template, so only ~(ne01+7)/4 rows were ever written (the rest kept stale
                // dst values → PPL 111/NaN; per-shape bisection could not fix a row-count bug).
                // Correct grid for our <2,4> is ceil(ne01/ROWS) = (ne01+1)/2.
                // GFX906_Q8_SHAPE_DEBUG=1 dumps per-shape dispatch; GFX906_Q8_SHAPE_ROWS
                // restricts the new mapping to shapes with an equal row count (bisection aid).
                static const char shape_mode = getenv("GFX906_Q8_SHAPE_MMV") ? getenv("GFX906_Q8_SHAPE_MMV")[0] : 'g';
                if (getenv("GFX906_Q8_SHAPE_DEBUG")) {
                    static int dbg_n = 0;
                    if (dbg_n++ < 40) {
                        fprintf(stderr, "[q8-shape] %s ne00=%d ne01=%d mode=%c\n", src0->name,
                            (int) src0->ne[0], (int) ne01, shape_mode);
                    }
                }
                const bool probe = (shape_mode == 'f' || shape_mode == 'o' || shape_mode == 'g') && ne00 > 1024;
                static const int row_bucket = getenv("GFX906_Q8_SHAPE_ROWS") ? atoi(getenv("GFX906_Q8_SHAPE_ROWS")) : 0;
                // [E16.6] furnace-exact dense ladder: all K, mapped as furnace maps them
                // GFX906_Q8_X_DISABLE bitmask: 1=flat 2=seg 4=splitk 8=rowu (bisection aid)
                static const int x_disable = getenv("GFX906_Q8_X_DISABLE") ? atoi(getenv("GFX906_Q8_X_DISABLE")) : 0;
                if (shape_mode == 'x' || shape_mode == 'h') {
                    if (ne00 == 320 && !(x_disable & 1)) {
                        gfx906_q8_flat<<<(ne01 + 24) / 25, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01);
                        break;
                    }
                    if (ne00 <= 1024 && !(x_disable & 2)) {
                        const uint32_t nb = (uint32_t) (ne00 >> 5);
                        const int seg = (nb <= 8) ? 8 : (nb <= 16) ? 16 : ((nb & (nb - 1)) == 0 ? (nb <= 32 ? nb : 32) : 32);
                        switch (seg) {
                            case 8:  gfx906_q8_seg<8> <<<((ne01 + 7) / 8), 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                            case 16: gfx906_q8_seg<16><<<((ne01 + 3) / 4), 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                            case 20: gfx906_q8_seg<20><<<((ne01 + 7) / 8), 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                            default: gfx906_q8_seg<32><<<((ne01 + 7) / 8), 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                        }
                        break;
                    }
                    if (shape_mode == 'x' && ne01 <= 1024 && ne00 >= 4096 && !(x_disable & 4)) {
                        // [E16.6-debug GFX906_Q8_SPLITK_VERIFY=1] compute splitk into dst, then
                        // the template <1,1> into a temp, device-compare and print once per shape
                        static const bool sk_verify = getenv("GFX906_Q8_SPLITK_VERIFY") != nullptr;
                        if (sk_verify) {
                            static int sk_v_n = 0;
                            if (sk_v_n++ < 12) {
                                gfx906_q8_splitk<<<ne01, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01);
                                ggml_cuda_pool_alloc<float> ref(ctx.pool(), ne01);
                                const dim3 g1(ne01, 1, 1);
                                mul_mat_vec_q8_0_repacked<1, 1, false><<<g1, 64, 0, stream>>>(
                                    w, xq, ref.get(), (uint32_t) ne00, (uint32_t) ne01,
                                    nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                                    nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
                                ggml_cuda_pool_alloc<float> rb(ctx.pool(), 2*ne01);
                                CUDA_CHECK(cudaMemcpyAsync(rb.get(), dst_d, ne01*sizeof(float), cudaMemcpyDeviceToHost, stream));
                                CUDA_CHECK(cudaMemcpyAsync(rb.get()+ne01, ref.get(), ne01*sizeof(float), cudaMemcpyDeviceToHost, stream));
                                CUDA_CHECK(cudaStreamSynchronize(stream));
                                float md = 0.0f; int mb = -1;
                                float s0 = 0.0f, s1 = 0.0f;
                                for (int i = 0; i < (int) ne01; i++) {
                                    float d = fabsf(rb.get()[i] - rb.get()[ne01 + i]);
                                    if (d > md) { md = d; mb = i; }
                                    s0 += rb.get()[i]; s1 += rb.get()[ne01 + i];
                                }
                                fprintf(stderr, "[splitk-hverify] ne00=%d ne01=%d maxdiff=%.6e at=%d sum_a=%.3f sum_b=%.3f\n",
                                    (int) ne00, (int) ne01, (double) md, mb, (double) s0, (double) s1);
                                break;
                            }
                        }
                        gfx906_q8_splitk<<<ne01, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01);
                        break;
                    }
                    if (shape_mode == 'x' && ne01 >= 512 && ne00 % 2048 == 0 && ne00 <= 8192 && !(x_disable & 8)) {
                        // [E16.6-debug GFX906_Q8_ROWU_VERIFY=1] rowu vs template <1,1>, host-compare
                        static const bool ru_verify = getenv("GFX906_Q8_ROWU_VERIFY") != nullptr;
                        static int ru_v_n = 0;
                        if (ru_verify && ru_v_n++ < 8) {
                            ggml_cuda_pool_alloc<float> rb(ctx.pool(), 2 * ne01);
                            switch (ne00 / 1024) {
                                case 2:  gfx906_q8_rowu<64, 2><<<ne01, 64, 0, stream>>>(w, xq, rb.get(), (uint32_t) ne00, (uint32_t) ne01); break;
                                case 4:  gfx906_q8_rowu<64, 4><<<ne01, 64, 0, stream>>>(w, xq, rb.get(), (uint32_t) ne00, (uint32_t) ne01); break;
                                case 6:  gfx906_q8_rowu<64, 6><<<ne01, 64, 0, stream>>>(w, xq, rb.get(), (uint32_t) ne00, (uint32_t) ne01); break;
                                default: gfx906_q8_rowu<64, 8><<<ne01, 64, 0, stream>>>(w, xq, rb.get(), (uint32_t) ne00, (uint32_t) ne01); break;
                            }
                            const dim3 g1r(ne01, 1, 1);
                            mul_mat_vec_q8_0_repacked<1, 1, false><<<g1r, 64, 0, stream>>>(
                                w, xq, rb.get() + ne01, (uint32_t) ne00, (uint32_t) ne01,
                                nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                                nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
                            CUDA_CHECK(cudaMemcpyAsync(rb.get() + 2*ne01, rb.get(), 2*ne01*sizeof(float), cudaMemcpyDeviceToHost, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                            float md = 0.0f; int mb = -1;
                            for (int i = 0; i < (int) ne01; i++) {
                                float d = fabsf(rb.get()[2*ne01 + i] - rb.get()[3*ne01 + i]);
                                if (d > md) { md = d; mb = i; }
                            }
                            fprintf(stderr, "[rowu-hverify] ne00=%d ne01=%d maxdiff=%.6e at=%d\n", (int) ne00, (int) ne01, (double) md, mb);
                            break;
                        }
                        switch (ne00 / 1024) {
                            case 2:  gfx906_q8_rowu<64, 2><<<ne01, 64, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                            case 4:  gfx906_q8_rowu<64, 4><<<ne01, 64, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                            case 6:  gfx906_q8_rowu<64, 6><<<ne01, 64, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                            default: gfx906_q8_rowu<64, 8><<<ne01, 64, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                        }
                        break;
                    }
                    // else fall through to the 'g' mapping below
                }
                if (shape_mode == 'g' || shape_mode == 'h' || (shape_mode == 'f' && ne01 >= 512)) {
                        const dim3 g1(ne01, 1, 1);
                        mul_mat_vec_q8_0_repacked<1, 1, false><<<g1, 64, 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                            nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                            nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
                    } else {
                        const dim3 g4((ne01 + 1) / 2, 1, 1); // ROWS=2 rows per block (ours), not furnace's ROWS*NWAVES=8
                        mul_mat_vec_q8_0_repacked<2, 4, false><<<g4, 256, 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                            nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                            nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
                    }
                    break;
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                // Measured A/B hook: route Q8_0 through the shared generic
                // mat-vec. If it holds parity the bespoke kernel retires.
                static const bool generic_mmv = getenv("GGML_RP_GENERIC_MMV") != nullptr;
                if (generic_mmv) {
                    mul_mat_vec_rp<GGML_TYPE_Q8_0, 16, 16, false><<<grid, 1024, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, 1, 0, 0, 0);
                    break;
                }
                mul_mat_vec_q8_0_repacked<16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                    nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
            } break;
            case GGML_TYPE_MXFP4: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_MXFP4, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_IQ4_NL: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_IQ4_NL, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q5_1: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q5_1, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q5_K: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q5_K, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q4_K: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q4_K, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q6_K: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q6_K, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    // Multi-token: 64-wide GEMM for large batches, 32-wide for small ones.
    const int nrl  = MMQ_RP_Q8_NROW_LANES;
    const int mmq_bm = MMQ_RP_Q8_BM;
    if (ne11 >= 128) {
        const int bn = 64 * MMQ_RP_Q8_TN;
        const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm,
                        (ne11 + bn - 1) / bn, 1);
        switch (src0->type) {
            case GGML_TYPE_Q8_0:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q8_0><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_MXFP4:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_MXFP4><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_IQ4_NL:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_IQ4_NL><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q5_1:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q5_1><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q5_K:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q5_K><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q4_K:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_K><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q6_K:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q6_K><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            default: GGML_ABORT("unsupported repack type");
        }
    } else {
        const int bn = 32;
        const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm,
                        (ne11 + bn - 1) / bn, 1);
        switch (src0->type) {
            case GGML_TYPE_Q8_0:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q8_0><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_MXFP4:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_MXFP4><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_IQ4_NL:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_IQ4_NL><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q5_1:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q5_1><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q5_K:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q5_K><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q4_K:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_K><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q6_K:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q6_K><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            default: GGML_ABORT("unsupported repack type");
        }
    }
    GGML_UNUSED(ctx);
}

// Fused MMV entry for repacked weights (dense): single-token only, fuses the up and
// gate dot products (shared quantized input) plus optional biases and the GLU op.
// Dispatch: ggml_cuda_try_fuse `{op,op,GLU}` and `{op,ADD}` repack branches.
void ggml_cuda_mul_mat_vec_repacked_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));
    GGML_ASSERT(dst->ne[1] == 1);   // fused MMV is single-token only
    GGML_ASSERT(fusion != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();

    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }

    const uint8_t * w_gate = nullptr;
    if (fusion->gate != nullptr) {
        const ggml_tensor * gate = fusion->gate;
        if (gate->view_src != nullptr && ggml_cuda_repack_tensor_supported(gate->view_src)) {
            w_gate = repack_view_get_cached(gate, gate->view_src, stream);
        } else {
            w_gate = (const uint8_t *) gate->data;
        }
    }

    const float * x_bias     = fusion->x_bias     ? (const float *) fusion->x_bias->data     : nullptr;
    const float * gate_bias  = fusion->gate_bias  ? (const float *) fusion->gate_bias->data  : nullptr;
    const ggml_glu_op glu_op = fusion->glu_op;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;

    ggml_cuda_pool_alloc<block_q8_1> src1_q8_1_own;
    block_q8_1 * src1_q8_1_d;
    {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        const int64_t s13 = src1->nb[3] / sizeof(float);
        src1_q8_1_d = repack_quantize_src1_q8_1(ctx, src0, src1, ne10, ne10_padded,
            s11, s12, s13, 1, ne12, ne13,
            (size_t) (ne13 * ne12 * x_stride), src1_q8_1_own, stream);
    }
    for (int64_t i3 = 0; i3 < ne13; i3++) {
    for (int64_t i2 = 0; i2 < ne12; i2++) {
        const block_q8_1 * xq = src1_q8_1_d
                          + (i3 * ne12 + i2) * x_stride;
        float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
        // Bias is shaped like the single-token dst ([ne0, 1, ne2]): per-channel stride ne01.
        const float * x_bias_s    = x_bias    ? x_bias    + (i3 * ne12 + i2) * ne01 : nullptr;
        const float * gate_bias_s = gate_bias ? gate_bias + (i3 * ne12 + i2) * ne01 : nullptr;

        const dim3 grid((ne01 + 15) / 16, 1, 1);
        if (src0->type == GGML_TYPE_MXFP4) {
            mul_mat_vec_rp<GGML_TYPE_MXFP4, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        } else if (src0->type == GGML_TYPE_IQ4_NL) {
            mul_mat_vec_rp<GGML_TYPE_IQ4_NL, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        } else if (src0->type == GGML_TYPE_Q5_K) {
            mul_mat_vec_rp<GGML_TYPE_Q5_K, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        } else if (src0->type == GGML_TYPE_Q4_K) {
            mul_mat_vec_rp<GGML_TYPE_Q4_K, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        } else if (src0->type == GGML_TYPE_Q6_K) {
            mul_mat_vec_rp<GGML_TYPE_Q6_K, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        } else {
            mul_mat_vec_q8_0_repacked<16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        }
    }
    }
}
