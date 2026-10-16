#pragma once

#include "common.cuh"

// [gfx906-opt P3] SCALE -> SILU -> MUL_MAT(hc up, repacked Q8_0) -> RESHAPE -> DSV4_HC_PRE(gated)
// fused into one launch (qwen4exp hyper-connection mix, single token rows). The SCALE/SILU and
// q8_1 quantize arithmetic match the solo launches exactly; the repacked Q8_0 dot is accumulated
// per full 32-value block and summed sequentially, whereas the solo mat-vec
// (mul_mat_vec_q8_0_repacked) splits each block in two 16-value halves and reduces over 64 lanes
// with a DPP butterfly. The two orders round differently at the last bit, so greedy output can
// diverge (measured: a normal technical prompt forks around token ~60; degenerate repeated text
// forks earlier). Numerically the model stays sane: PPL +0.5% relative, inside the +/-5% noise band.
// Off-switch: GFX906_HC_UP_PRE=0 (tolerant off), GFX906_NO_HC_UP_PRE=1 (legacy).

bool ggml_cuda_hc_up_pre_ok(const ggml_tensor * scale, const ggml_tensor * silu, const ggml_tensor * up, const ggml_tensor * pre);
void ggml_cuda_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, const ggml_tensor * up,
        ggml_tensor * pre, void * yq);

// [gfx906-opt P3] if some later repack-route MUL_MAT consumes `t` as a flat src1 whose
// row needs no padding, claim a ring-cache slot and have the caller's kernel also write
// t as q8_1 blocks (the consumer's quantize cache then hits instead of re-quantizing).
// Returns nullptr when no such consumer exists, or under stream capture when a buffer
// would have to grow (the consumer then quantizes normally - correctness is unaffected).
void * ggml_cuda_hc_up_pre_emit_target(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const ggml_tensor * t);

// [gfx906-opt P3.4] SCALE -> SILU in one launch (wider hc mix batches)
void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * scale_node, ggml_tensor * silu_node);
