#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);

// [gfx906 A5] one step of an adjacent elementwise chain (see ggml_cuda_elem_chain_match)
struct ggml_cuda_elem_chain_step {
    const float * x;                        // operand; null for a unary step
    int64_t       nb1, nb2, nb3;            // operand strides in elements
    int64_t       ne1, ne2, ne3;            // 1 == broadcast along that dimension
    int           op;
};

// [gfx906 A5] 2-4 adjacent elementwise nodes (ADD/SUB/MUL/DIV + softplus/sigmoid/silu) with one
// shape collapse, dispatched as a single launch. n-1 launches saved per chain.
struct ggml_cuda_elem_chain {
    const ggml_tensor *       head_x = nullptr;
    ggml_cuda_elem_chain_step steps[4] = {};
    const ggml_tensor *       nodes[4] = {};
    int                       n = 0;
};

bool ggml_cuda_elem_chain_match(const ggml_cgraph * cgraph, int i, ggml_cuda_elem_chain & plan);
void ggml_cuda_op_elem_chain(ggml_backend_cuda_context & ctx, const ggml_cuda_elem_chain & plan);
