#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);

// ===== [gfx906 P6] adjacent elementwise chain =====
// A run of 2..8 adjacent, single-consumer F32 elementwise nodes with one shape
// (ADD/SUB/MUL/DIV, SCALE, CLAMP, SQRT, SQR and the unary family) is evaluated by
// one kernel instead of one launch per node. Per element the ops are applied in the
// same order with the same expressions as the solo kernels, so the result is
// bit-identical for the chains this matcher accepts.
enum gfx906_ec_op {
    GFX906_EC_ADD = 0,
    GFX906_EC_SUB,
    GFX906_EC_MUL,
    GFX906_EC_DIV,
    GFX906_EC_SCALE,
    GFX906_EC_CLAMP,
    GFX906_EC_SQRT,
    GFX906_EC_SQR,
    GFX906_EC_ABS,
    GFX906_EC_NEG,
    GFX906_EC_TANH,
    GFX906_EC_SIGMOID,
    GFX906_EC_SILU,
    GFX906_EC_SOFTPLUS,
    GFX906_EC_EXP,
    GFX906_EC_RELU,
    GFX906_EC_GELU,
    GFX906_EC_GELU_QUICK,
    GFX906_EC_GELU_ERF,
};

#define GFX906_EC_MAX_STEPS 8

struct gfx906_elem_chain_step {
    const float * p;     // second operand (null for unary/scale/clamp/sqrt/sqr)
    int64_t       s0;    // element stride of dim0 (0 when the operand broadcasts dim0)
    int64_t       s1;    // element stride of dim1 (0 when the operand broadcasts dim1)
    int           op;
    float         a, b;  // SCALE: scale/bias, CLAMP: min/max
};

struct gfx906_elem_chain_plan {
    const ggml_tensor *         in = nullptr;   // chain input (tail-1 shape, dim0 not broadcast)
    const ggml_tensor *         nodes[GFX906_EC_MAX_STEPS] = {};
    gfx906_elem_chain_step      steps[GFX906_EC_MAX_STEPS] = {};
    int                         n = 0;
};

// dry = match only (stats/launch-free). Returns the number of nodes covered (>=2) or 0.
int  gfx906_elem_chain_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, bool dry);

// [gfx906 P6.1] one-shot producer-epilogue census, printed per graph evaluation
void gfx906_pe_census(const ggml_cgraph * cgraph);
