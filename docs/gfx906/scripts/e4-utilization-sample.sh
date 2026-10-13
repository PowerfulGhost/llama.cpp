#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib HSA_FORCE_FINE_GRAIN_PCIE=1 GPU_MAX_HW_QUEUES=8
M=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
C="-m $M -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -o md -r 5 -n 1 -p 4096"
for mode in layer tensor; do
  ./llama-bench $C -sm $mode >/prof/u_$mode.bench 2>&1 &
  B=$!
  sleep 8   # 等预热和第一轮
  : > /prof/u_$mode.jsonl
  for i in $(seq 1 60); do rocm-smi --showuse --json 2>/dev/null >> /prof/u_$mode.jsonl; sleep 0.2; done
  wait $B
done
echo "=== DONE ==="
