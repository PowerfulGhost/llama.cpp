#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

#include <atomic>
#include <cstdlib>

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// ---------------------------------------------------------------------------------------------
// [gfx906 A1] fused recurrent conv step (port of furnace ssm-conv.cu "conv_step_concat").
//
// The decode / verify graph of qwen4exp builds, per GDN layer:
//     GET_ROWS(conv cache, ids) -> RESHAPE -> CONCAT(state, transpose(x)) -> CPY(slot0..N) -> SSM_CONV -> SILU
// which is 4..6 launches per layer to move ~0.5 % of the model's bytes. This kernel does the whole
// step in one launch and reads the cache rows directly through the row indices, so the GET_ROWS can
// be skipped entirely. Arithmetic is preserved: the concat/cpy part is a pure copy, and the optional
// conv reproduces ssm_conv_f32<true> tap order + the ADD bias 0.0f + silu exactly.
//
// One thread per channel: it reads every sequence's state first and writes afterwards, so a write to
// sequence a's row can never land under another sequence's read.
// ---------------------------------------------------------------------------------------------

#define CONV_STEP_MAX_SEQS 8
#define CONV_STEP_MAX_T    16
#define CONV_STEP_MAX_CPY  8

struct conv_step_cpy {
    char *   dst;      // destination row of sequence 0
    int64_t  dst_nb1;  // bytes between sequences in the destination
    int      col0;     // first concat column of the tail
};

struct conv_step_args {
    const char *    cache;     // elided gather: cache base; else nullptr
    int64_t         cache_nb1; // bytes per cache row
    const int32_t * ids;       // elided gather: row index per sequence
    int64_t         ids_nb0;
    const char *    s0;        // not elided: the gathered state, [state_cols, C, n_seqs]
    int64_t         s0_nb1, s0_nb2;
    const char *    x;         // [T, C, n_seqs] (a transposed view)
    int64_t         x_nb0, x_nb1, x_nb2;
    float *         out;       // concat, contiguous [state_cols + T, C, n_seqs]
    int             C, T, n_seqs, n_cpy;
    conv_step_cpy   cpy[CONV_STEP_MAX_CPY];
    const float *   conv_w;    // fused SSM_CONV -> SILU: weights [SC + 1, C] (row stride conv_w_nb1); else nullptr
    int64_t         conv_w_nb1;
    float *         y;         // silu(conv) output, contiguous [C, T, n_seqs]
};

template <int SC>
static __global__ void conv_step_concat_f32(const conv_step_args a) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= a.C) {
        return;
    }
    float st[CONV_STEP_MAX_SEQS][SC];
#pragma unroll
    for (int b = 0; b < CONV_STEP_MAX_SEQS; b++) {
        if (b < a.n_seqs) {
            const float * src = a.cache != nullptr
                ? (const float *) (a.cache + (int64_t) *(const int32_t *) ((const char *) a.ids + b * a.ids_nb0) * a.cache_nb1) + (int64_t) c * SC
                : (const float *) (a.s0 + b * a.s0_nb2 + (int64_t) c * a.s0_nb1);
#pragma unroll
            for (int k = 0; k < SC; k++) {
                st[b][k] = src[k];
            }
        }
    }
    const int W = SC + a.T;
#pragma unroll
    for (int b = 0; b < CONV_STEP_MAX_SEQS; b++) {
        if (b >= a.n_seqs) {
            break;
        }
        // no per-thread arrays with runtime indices (they would live in scratch): write the concat row
        // straight to global memory and copy the tails back out of it (same thread, program order)
        float * o = a.out + ((int64_t) b * a.C + c) * W;
#pragma unroll
        for (int k = 0; k < SC; k++) {
            o[k] = st[b][k];
        }
        for (int t = 0; t < a.T; t++) {
            o[SC + t] = *(const float *) (a.x + b * a.x_nb2 + (int64_t) c * a.x_nb1 + t * a.x_nb0);
        }
        for (int j = 0; j < a.n_cpy; j++) {
            float * d = (float *) (a.cpy[j].dst + b * a.cpy[j].dst_nb1) + (int64_t) c * SC;
            const float * from = o + a.cpy[j].col0;
#pragma unroll
            for (int k = 0; k < SC; k++) {
                d[k] = from[k];
            }
        }
        if (a.conv_w != nullptr) {
            // ssm_conv_f32<true> arithmetic: sum over the window in tap order, + 0 bias, silu
            const float * w = (const float *) ((const char *) a.conv_w + (int64_t) c * a.conv_w_nb1);
            float wr[SC + 1];
#pragma unroll
            for (int k = 0; k <= SC; k++) {
                wr[k] = w[k];
            }
            for (int t = 0; t < a.T; t++) {
                float sumf = 0.0f;
#pragma unroll
                for (int k = 0; k <= SC; k++) {
                    sumf += o[t + k] * wr[k];
                }
                sumf += 0.0f; // the zero bias of ssm_conv_f32 (keeps the sign of a zero sum the same)
                a.y[((int64_t) b * a.T + t) * a.C + c] = ggml_cuda_op_silu_single(sumf);
            }
        }
    }
}

static const ggml_tensor * conv_step_root(const ggml_tensor * t) {
    while (t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

static bool conv_step_disabled() {
    static const bool disabled = getenv("GGML_CUDA_NO_CONV_STEP_FUSION") != nullptr ||
        (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION")) != 0);
    return disabled;
}

struct conv_step_plan {
    const ggml_tensor * gr   = nullptr; // the state GET_ROWS (nullptr when it lives in another split)
    const ggml_tensor * cat  = nullptr;
    const ggml_tensor * conv = nullptr; // fused SSM_CONV, or nullptr
    const ggml_tensor * silu = nullptr;
    int gi = -1, ci = -1, last = -1;
    int n_cpy = 0;
    conv_step_cpy cpy[CONV_STEP_MAX_CPY];
    bool elide = false;                 // the gather may be skipped (kernel reads the cache directly)
};

// Everything the launch needs, computed one way only, so the "elidable" predicate and the launch can
// never disagree (a disagreement would leave the CONCAT reading an unwritten gather output).
static bool conv_step_plan_concat(const ggml_cgraph * cgraph, int ci, conv_step_plan & p) {
    if (cgraph == nullptr || ci < 0 || ci >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * cat = cgraph->nodes[ci];
    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 || cat->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(cat) || cat->src[0] == nullptr || cat->src[1] == nullptr ||
            cat->src[0]->type != GGML_TYPE_F32 || cat->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * s0 = cat->src[0];
    const ggml_tensor * x  = cat->src[1];
    const int64_t C = s0->ne[1], ns = s0->ne[2];
    if (!(s0->ne[0] == 3 && x->ne[0] >= 1 && x->ne[0] <= CONV_STEP_MAX_T && ns >= 1 && ns <= CONV_STEP_MAX_SEQS &&
            s0->ne[3] == 1 && x->ne[1] == C && x->ne[2] == ns && x->ne[3] == 1 && s0->nb[0] == sizeof(float) &&
            x->nb[0] % sizeof(float) == 0 && ggml_is_contiguous(s0))) {
        return false;
    }
    const int T = (int) x->ne[0];

    // the gather this state comes from: it must be the only consumer's producer, in this graph
    const ggml_tensor * gr = conv_step_root(s0);
    int gi = -1;
    for (int k = 0; k < cgraph->n_nodes; k++) {
        if (cgraph->nodes[k] == gr) {
            gi = k;
            break;
        }
    }
    if (gr->op != GGML_OP_GET_ROWS || gi < 0 || gi >= ci) {
        return false;
    }
    if (!(gr->type == GGML_TYPE_F32 && gr->src[0] != nullptr && gr->src[0]->type == GGML_TYPE_F32 &&
            gr->src[1] != nullptr && gr->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(gr) &&
            gr->src[0]->nb[0] == sizeof(float) && gr->src[0]->ne[0] == gr->ne[0] && gr->ne[2] == 1 && gr->ne[3] == 1 &&
            gr->src[1]->ne[0] == gr->ne[1] && gr->ne[1] >= 1 && gr->ne[1] <= CONV_STEP_MAX_SEQS)) {
        return false;
    }
    // exactly one consumer, and it is this concat
    {
        int uses = 0;
        bool other = false;
        for (int k = gi + 1; k < cgraph->n_nodes && !other; k++) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (ggml_cuda_is_view_or_noop_public(n)) {
                continue;
            }
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (n->src[j] != nullptr && conv_step_root(n->src[j]) == gr) {
                    if (n == cat) {
                        uses++;
                    } else {
                        other = true;
                        break;
                    }
                }
            }
        }
        if (other || uses != 1) {
            return false;
        }
    }
    // nothing between the gather and the concat may write the cache rows or reuse the index memory
    const ggml_tensor * cache = conv_step_root(gr->src[0]);
    {
        const char * i0 = (const char *) gr->src[1]->data;
        const char * i1 = i0 + ggml_nbytes(gr->src[1]);
        for (int k = gi + 1; k < ci; k++) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (ggml_cuda_is_view_or_noop_public(n)) {
                continue;
            }
            if (conv_step_root(n) == cache) {
                return false;
            }
            if (n->view_src == nullptr && n->data != nullptr) {
                const char * a0 = (const char *) n->data;
                const char * a1 = a0 + ggml_nbytes(n);
                if (a0 < i1 && i0 < a1) {
                    return false;
                }
            }
        }
    }

    p = conv_step_plan{};
    p.gr = gr; p.cat = cat; p.gi = gi; p.ci = ci; p.last = ci;

    // the CPYs right after the concat that write state tails of it into the cache
    for (int k = ci + 1; k < cgraph->n_nodes && p.n_cpy < CONV_STEP_MAX_CPY; k++) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (ggml_cuda_is_view_or_noop_public(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY) {
            break;
        }
        const ggml_tensor * src = n->src[0];
        const ggml_tensor * dst = n->src[1];
        if (src->view_src != cat || conv_step_root(dst) != cache || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
                src->ne[0] != 3 || src->ne[1] != C || src->ne[2] != ns || src->nb[1] != cat->nb[1] || src->nb[2] != cat->nb[2] ||
                src->view_offs % sizeof(float) != 0 || (int) (src->view_offs / sizeof(float)) + 3 > 3 + T ||
                dst->ne[0] != 3 * C || dst->ne[1] != ns || dst->nb[0] != sizeof(float) ||
                (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            break;
        }
        p.cpy[p.n_cpy].dst     = (char *) dst->data;
        p.cpy[p.n_cpy].dst_nb1 = dst->nb[1];
        p.cpy[p.n_cpy].col0    = (int) (src->view_offs / sizeof(float));
        p.n_cpy++;
        p.last = k;
    }
    if (p.n_cpy == 0) {
        return false; // the step must write the new state tail, else the state never advances
    }

    // the SSM_CONV -> SILU that reads the concat, when only views sit between it and the last CPY
    {
        int k = p.last + 1;
        while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop_public(cgraph->nodes[k])) {
            k++;
        }
        if (k + 1 < cgraph->n_nodes) {
            const ggml_tensor * conv = cgraph->nodes[k];
            const ggml_tensor * silu = cgraph->nodes[k + 1];
            const ggml_tensor * w    = conv->src[1];
            if (conv->op == GGML_OP_SSM_CONV && conv->src[0] == cat && w != nullptr && w->type == GGML_TYPE_F32 &&
                    w->ne[0] == 4 && w->ne[1] == C && w->nb[0] == sizeof(float) && conv->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(conv) && conv->ne[0] == C && conv->ne[1] == T && conv->ne[2] == ns &&
                    silu->op == GGML_OP_UNARY && ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU && silu->src[0] == conv &&
                    ggml_is_contiguous(silu) && silu->type == GGML_TYPE_F32 &&
                    ggml_cuda_node_has_n_uses_public(cgraph, k, 1) &&
                    (conv->flags & GGML_TENSOR_FLAG_COMPUTE) && (silu->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                // other threads still read x, the state and the concat while one writes y: no overlap allowed
                auto ovl = [&](const ggml_tensor * t) {
                    const char * a0 = (const char *) silu->data; const char * a1 = a0 + ggml_nbytes(silu);
                    const char * b0 = (const char *) t->data;    const char * b1 = b0 + ggml_nbytes(t);
                    return a0 < b1 && b0 < a1;
                };
                // y on top of x is fine when the layouts match: each thread then overwrites only elements it has read
                const bool x_alias = silu->data == x->data && x->nb[0] == (size_t) C * sizeof(float) &&
                    x->nb[1] == sizeof(float) && x->nb[2] == (size_t) T * C * sizeof(float);
                bool clash = (ovl(x) && !x_alias) || ovl(cat) || ovl(w) ||
                    ovl(gr->src[0]) || ovl(gr->src[1]) || (gr->src[0]->data == s0->data);
                if (!clash) {
                    p.conv = conv;
                    p.silu = silu;
                    p.last = k + 1;
                }
            }
        }
    }

    p.elide = true;
    return true;
}

bool ggml_cuda_conv_state_gather_elidable(const ggml_cgraph * cgraph, const ggml_tensor * gr) {
    if (conv_step_disabled() || cgraph == nullptr || gr == nullptr || gr->op != GGML_OP_GET_ROWS) {
        return false;
    }
    for (int ci = 0; ci < cgraph->n_nodes; ci++) {
        const ggml_tensor * n = cgraph->nodes[ci];
        if (n->op != GGML_OP_CONCAT || n->src[0] == nullptr || conv_step_root(n->src[0]) != gr) {
            continue;
        }
        conv_step_plan p;
        if (conv_step_plan_concat(cgraph, ci, p) && p.gr == gr && p.elide) {
            return true;
        }
    }
    return false;
}

int ggml_cuda_try_conv_step_fusion(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (conv_step_disabled()) {
        return -1;
    }
    conv_step_plan p;
    if (!conv_step_plan_concat(cgraph, i, p)) {
        return -1;
    }
    const ggml_tensor * s0 = p.cat->src[0];
    const ggml_tensor * x  = p.cat->src[1];
    const int64_t C = s0->ne[1], ns = s0->ne[2];

    conv_step_args a = {};
    a.cache     = (const char *) p.gr->src[0]->data;
    a.cache_nb1 = p.gr->src[0]->nb[1];
    a.ids       = (const int32_t *) p.gr->src[1]->data;
    a.ids_nb0   = p.gr->src[1]->nb[0];
    a.s0        = (const char *) s0->data; a.s0_nb1 = s0->nb[1]; a.s0_nb2 = s0->nb[2];
    a.x = (const char *) x->data; a.x_nb0 = x->nb[0]; a.x_nb1 = x->nb[1]; a.x_nb2 = x->nb[2];
    a.out = (float *) p.cat->data;
    a.C = (int) C; a.T = (int) x->ne[0]; a.n_seqs = (int) ns; a.n_cpy = p.n_cpy;
    for (int j = 0; j < p.n_cpy; j++) {
        a.cpy[j] = p.cpy[j];
    }
    if (p.conv != nullptr) {
        const ggml_tensor * w = p.conv->src[1];
        a.conv_w     = (const float *) w->data;
        a.conv_w_nb1 = w->nb[1];
        a.y          = (float *) p.silu->data;
    }

    static std::atomic<bool> witnessed(false);
    if (!witnessed.exchange(true)) {
        GGML_LOG_WARN("[conv-step] fused conv step engaged (C=%d T=%d n_seqs=%d cpy=%d conv=%d)\n",
                a.C, a.T, a.n_seqs, a.n_cpy, (int) (a.conv_w != nullptr));
    }

    const int nt = 256;
    conv_step_concat_f32<3><<<(a.C + nt - 1) / nt, nt, 0, ctx.stream()>>>(a);
    return p.last - i;
}

// graph_optimize part of the fused conv step: the SSM_CONV -> SILU that reads the step's CONCAT comes a
// few nodes later (the recurrent-state gather and reset sit between). Move it right behind the CONCAT's
// state-tail CPYs so ggml_cuda_try_conv_step_fusion sees it. Runs before allocation; the moved nodes
// need only the CONCAT and the conv weights, both ready at the new position.
void ggml_cuda_conv_step_graph_optimize(ggml_cgraph * cgraph) {
    if (conv_step_disabled() || cgraph == nullptr) {
        return;
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * cat = cgraph->nodes[i];
        if (cat->op != GGML_OP_CONCAT || cat->src[0] == nullptr || cat->src[1] == nullptr ||
                cat->src[0]->type != GGML_TYPE_F32 || cat->src[1]->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(cat) || cat->src[0]->ne[0] != 3) {
            continue;
        }
        int last = i;
        for (int k = i + 1; k < cgraph->n_nodes; k++) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (ggml_cuda_is_view_or_noop_public(n)) {
                continue;
            }
            if (n->op != GGML_OP_CPY || n->src[0]->view_src != cat) {
                break;
            }
            last = k;
        }
        if (last == i) {
            continue; // no state-tail CPYs: this is not a conv step
        }
        int ic = -1;
        for (int k = last + 1; k < cgraph->n_nodes && k <= last + 32; k++) {
            if (cgraph->nodes[k]->op == GGML_OP_SSM_CONV && cgraph->nodes[k]->src[0] == cat) {
                ic = k;
                break;
            }
        }
        if (ic < 0 || ic + 1 >= cgraph->n_nodes) {
            continue;
        }
        ggml_tensor * conv = cgraph->nodes[ic];
        ggml_tensor * silu = cgraph->nodes[ic + 1];
        if (silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != conv) {
            continue;
        }
        for (int k = ic + 1; k > last + 2; k--) {
            cgraph->nodes[k] = cgraph->nodes[k - 2];
        }
        cgraph->nodes[last + 1] = conv;
        cgraph->nodes[last + 2] = silu;
        i = last + 2;
    }
}
