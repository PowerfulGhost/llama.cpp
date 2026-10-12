#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8
M=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
C="-m $M -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -p 16 -n 128 -r 3 -o md"
for B in 1 2 4 8 16 32; do
  echo "### PP8        b=$B"; ./llama-bench $C -sm layer  -b $B -ub $B
  if [ "$B" != "1" ]; then
    echo "### PP8+fusion b=$B"; GGML_CUDA_REPACK_MMV_FUSION=1 ./llama-bench $C -sm layer -b $B -ub $B
  fi
  echo "### TP8        b=$B"; ./llama-bench $C -sm tensor -b $B -ub $B
done
echo "=== DONE ==="
