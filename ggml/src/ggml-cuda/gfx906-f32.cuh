// GCN (gfx906) F32 mat-vec family (ported from gfx906-perf/furnace):
// one-wave-per-row and grouped shared-activation launches for thin F32 mat-vecs.
#pragma once

bool ggml_cuda_f32_gcn_matvec(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, int cc);
bool ggml_cuda_f32_multi_ok(const ggml_tensor * mm);
void ggml_cuda_mul_mat_f_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * mms, int n);

// Tolerant env parser shared by this family and by ggml-cuda.cu.
// Returns `def` when the variable is unset or empty; false when the value is
// 0/off/false/no (case-insensitive); true otherwise.
// For an `_OFF`-style switch use `!ggml_cuda_f32_env_on("..._OFF", false)`
// (unset => feature ON, 0/off/false/no => feature ON, 1/on/true/yes => feature OFF).
bool ggml_cuda_f32_env_on(const char * name, bool def);
