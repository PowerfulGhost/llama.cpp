#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8
M=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
BASE="-m $M -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -p 512,2048,4096 -n 128 -r 3 -o md -sm layer"
echo "### 用了 repack 的设备 buffer（-v）"
./llama-bench $BASE -p 512 -n 32 -r 1 -v 2>&1 | grep -E "Repacked model buffer size" | head -3
echo "=== ON ==="; ./llama-bench $BASE; echo "exitON=$?"
echo "=== OFF ==="; ./llama-bench $BASE --repack 0; echo "exitOFF=$?"
echo "=== DONE ==="
