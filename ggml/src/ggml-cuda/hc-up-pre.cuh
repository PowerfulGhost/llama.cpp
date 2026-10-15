#pragma once

#include "common.cuh"

// [gfx906-opt P3] SCALE -> SILU -> MUL_MAT(hc up, repacked Q8_0) -> RESHAPE -> DSV4_HC_PRE(gated)
// fused into one launch (qwen4exp hyper-connection mix, single token rows). The kernel's
// quantize/dot/sum order matches the four separate launches, so the DSV4_HC_PRE output is
// bitwise identical (off-switch: GFX906_NO_HC_UP_PRE=1).

bool ggml_cuda_hc_up_pre_ok(const ggml_tensor * scale, const ggml_tensor * silu, const ggml_tensor * up, const ggml_tensor * pre);
void ggml_cuda_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, const ggml_tensor * up,
        ggml_tensor * pre, void * yq);

// [gfx906-opt P3] if some later repack-route MUL_MAT consumes `t` as a flat src1 whose
// row needs no padding, claim a ring-cache slot and have the caller's kernel also write
// t as q8_1 blocks (the consumer's quantize cache then hits instead of re-quantizing).
// Returns nullptr when no such consumer exists, or under stream capture when a buffer
// would have to grow (the consumer then quantizes normally - correctness is unaffected).
void * ggml_cuda_hc_up_pre_emit_target(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const ggml_tensor * t);
