#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// [gfx906-opt P3] SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST -> RMS_NORM -> MUL in one
// launch (qwen4exp hc combine + the next hc mix norm); sma_* fold the shared-expert
// combine in as post's x (b + a*sigmoid(g)); yq optionally emits the result as q8_1.
bool ggml_cuda_op_hc_post_rms_norm_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * inject, ggml_tensor * post,
        const float s_in, const float b_in, const float s_out, const float b_out, const ggml_tensor * norm, ggml_tensor * mul_tensor, void * yq,
        const ggml_tensor * sma_g = nullptr, const ggml_tensor * sma_a = nullptr, const ggml_tensor * sma_b = nullptr,
        const bool dry_run = false);
