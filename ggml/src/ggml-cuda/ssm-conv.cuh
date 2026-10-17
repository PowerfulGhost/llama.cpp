#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// [gfx906 A1] fused recurrent conv step: GET_ROWS(conv cache) -> CONCAT(state, x^T) -> the state-tail
// CPYs (+ optionally SSM_CONV -> SILU) in one launch. Returns the number of nodes after i the launch
// covered, or -1 when the CONCAT at i does not qualify.
int  ggml_cuda_try_conv_step_fusion(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// true: this conv-state gather is read in place by its fused conv step and must not be computed
bool ggml_cuda_conv_state_gather_elidable(const ggml_cgraph * cgraph, const ggml_tensor * gr);
// graph_optimize: move the step's SSM_CONV -> SILU right behind its CONCAT's state-tail CPYs
void ggml_cuda_conv_step_graph_optimize(ggml_cgraph * cgraph);
// ggml_cuda_is_view_or_noop, exported for the fusion matchers
bool ggml_cuda_is_view_or_noop_public(const ggml_tensor * t);
bool ggml_cuda_node_has_n_uses_public(const ggml_cgraph * cgraph, int node_idx, int n_uses);
