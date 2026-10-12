#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib
export HSA_FORCE_FINE_GRAIN_PCIE=1
M=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
F="-m $M -f /corpus/corpus-perplexity.txt -c 512 --chunks 2 -ngl 99 -fa on -ctk f16 -ctv f16 -sm layer -lm dio"
for B in 1 2 4; do
  for U in 1 0; do
    printf "b=%s fusion=%s : " "$B" "$U"
    GGML_CUDA_REPACK_MMV_FUSION=$U ./llama-perplexity $F -b $B 2>&1 | grep -o "Final estimate: PPL = [0-9.]* +/- [0-9.]*" || echo "(fail)"
  done
done
echo "=== DONE ==="
