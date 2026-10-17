#include "qsa-sel.cuh"

#include <cstdlib>

// [gfx906 A4] QSA selection tail of qwen4exp (src/models/qwen4exp.cpp, build_qsa_sel).
//
// Per QSA layer the per-token cell index is rebuilt from scratch out of eleven tiny
// launches, of which these seven exist only to produce `sel_idx = (sel - dump) * live + dump`
// for a [n_sel, n_batch] tensor (n_sel = 259 at decode, i.e. 6 elements of real work):
//
//   ones  = fill(live, 1.0f)                       GGML_OP_FILL    fill_kernel
//   ramp  = cumsum(ones)                           GGML_OP_CUMSUM  cumsum_kernel
//   dump  = scale_bias(ramp, 1.0f, n_kv - 1)       GGML_OP_SCALE   scale_f32
//   sub   = cast_f32(sel_idx) - dump               GGML_OP_SUB     k_bin_bcast
//   mul   = sub * live                             GGML_OP_MUL     k_bin_bcast
//   add   = mul + dump                             GGML_OP_ADD     k_bin_bcast
//   sel   = cast_i32(add)                          GGML_OP_CPY     cpy_scalar_contiguous
//
// Everything above is a pure function of (sel_idx, live, n_kv): the ramp is the constant
// n_kv + slot, so the whole tail collapses into one element-wise kernel. The replaced nodes
// are elided in the compute loop, like the conv-step and GDN-gather fusions.
//
// Shape independence: one thread per (slot, token) cell, no width-dependent reduction, so
// prefill (large n_batch) behaves exactly like decode. Unlike R8's qsa-mask rewrite there is
// no n_kv threshold and therefore no prefill cliff; the fused value is bit-identical because
// `live` is exactly 0.0 or 1.0 and `dump`/`sel` are exact small integers in f32.

static __global__ void qsa_sel_tail_kernel(
        const float * __restrict__ live, // [n_sel, n_batch]
        const float * __restrict__ sel,  // [n_sel, n_batch], cast_f32 of the i32 cell index
        int32_t     * __restrict__ dst,  // [n_sel, n_batch]
        const int64_t n_sel, const int64_t nelem, const float bias) {
    const int64_t k = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nelem) {
        return;
    }
    const int64_t slot = k % n_sel;
    const float   dump = bias + (float) (slot + 1); // cumsum(ones)[slot] = slot + 1
    const float   x    = sel[k];
    dst[k] = (int32_t) ((x - dump) * live[k] + dump);
}

static bool qsa_sel_disabled() {
    static const bool disabled = getenv("GGML_CUDA_NO_QSA_SEL_FUSION") != nullptr;
    return disabled;
}

int ggml_cuda_try_qsa_sel_fusion(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    if (qsa_sel_disabled()) {
        return -1;
    }

    static const enum ggml_op ops[7] = {
        GGML_OP_FILL, GGML_OP_CUMSUM, GGML_OP_SCALE, GGML_OP_SUB, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_CPY,
    };
    const int last = i + 6;
    if (!ggml_can_fuse_subgraph(cgraph, i, { ops[0], ops[1], ops[2], ops[3], ops[4], ops[5], ops[6] }, { last })) {
        return -1;
    }

    const ggml_tensor * t_fill  = cgraph->nodes[i + 0];
    const ggml_tensor * t_cum   = cgraph->nodes[i + 1];
    const ggml_tensor * t_dump  = cgraph->nodes[i + 2];
    const ggml_tensor * t_sub   = cgraph->nodes[i + 3];
    const ggml_tensor * t_mul   = cgraph->nodes[i + 4];
    const ggml_tensor * t_add   = cgraph->nodes[i + 5];
    const ggml_tensor * t_cast  = cgraph->nodes[i + 6];

    // edges: the chain must be exactly the expression quoted above
    if (t_cum->src[0] != t_fill || t_dump->src[0] != t_cum || t_mul->src[0] != t_sub ||
            t_add->src[0] != t_mul || t_cast->src[0] != t_add || t_sub->src[1] != t_dump || t_add->src[1] != t_dump) {
        return -1;
    }

    const ggml_tensor * live = t_mul->src[1];
    const ggml_tensor * sel  = t_sub->src[0];
    if (live == nullptr || sel == nullptr || live == t_sub || live->type != GGML_TYPE_F32 ||
            sel->type != GGML_TYPE_F32 || t_cast->type != GGML_TYPE_I32) {
        return -1;
    }

    // the whole tail is F32 [n_sel > 1, n_batch], contiguous, no depth
    if (!ggml_are_same_shape(t_fill, t_cast) || !ggml_are_same_shape(live, t_cast) || !ggml_are_same_shape(sel, t_cast) ||
            t_cast->ne[0] < 2 || t_cast->ne[2] != 1 || t_cast->ne[3] != 1) {
        return -1;
    }
    for (const ggml_tensor * t : { t_fill, t_cum, t_dump, t_sub, t_mul, t_add, live, sel }) {
        if (!ggml_is_contiguous(t)) {
            return -1;
        }
    }
    if (!ggml_is_contiguous(t_cast)) {
        return -1;
    }

    // fill value must be 1.0 (the ramp) and the scale must be a pure bias
    float fill_v = 0.0f, scale_v = 0.0f, bias_v = 0.0f;
    memcpy(&fill_v, t_fill->op_params, sizeof(float));
    memcpy(&scale_v, t_dump->op_params + 0, sizeof(float));
    memcpy(&bias_v, t_dump->op_params + 1, sizeof(float));
    if (fill_v != 1.0f || scale_v != 1.0f) {
        return -1;
    }

    // the fused launch reads live/sel and writes only t_cast: no aliasing with the elided outputs
    {
        const char * lo = (const char *) t_cast->data;
        const char * hi = lo + ggml_nbytes(t_cast);
        for (const ggml_tensor * t : { live, sel }) {
            const char * a = (const char *) t->data;
            const char * b = a + ggml_nbytes(t);
            if (lo < b && a < hi) {
                return -1;
            }
        }
    }

    const int64_t n_sel = t_cast->ne[0];
    const int64_t nelem = ggml_nelements(t_cast);
    const int64_t nblk  = (nelem + 255) / 256;

    qsa_sel_tail_kernel<<<(unsigned) nblk, 256, 0, ctx.stream()>>>(
        (const float *) live->data, (const float *) sel->data, (int32_t *) t_cast->data, n_sel, nelem, bias_v);
    CUDA_CHECK(cudaGetLastError());

    return 6;
}
