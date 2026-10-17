#pragma once

#include "common.cuh"

// [gfx906 A4] fuse the QSA selection tail of qwen4exp into one launch:
//   dump  = scale_bias(cumsum(fill(live, 1.0f)), 1.0f, n_kv - 1)   // constant ramp: n_kv + slot
//   idx_f = (cast_f32(sel_idx) - dump) * live + dump
//   sel   = cast_i32(idx_f)
// Returns the number of extra nodes to skip (>= 0) when the fusion fired, -1 otherwise.
int ggml_cuda_try_qsa_sel_fusion(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
