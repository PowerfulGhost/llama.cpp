#include "common.cuh"

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst);

void ggml_cuda_ldxtk_chunk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_ldxtk_chunk_supported(int device, const ggml_tensor * dst);
void ggml_cuda_ldxtk_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_ldxtk_merge_supported(int device, const ggml_tensor * dst);
