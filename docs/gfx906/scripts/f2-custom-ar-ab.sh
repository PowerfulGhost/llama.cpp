#!/usr/bin/env bash
# F2（自定义 AllReduce）A/B：容器内执行路径 /build/bin
#
# docker run --rm --ipc=host -w /build/bin \
#   --device=/dev/kfd --device=/dev/dri --group-add video --group-add 993 \
#   -v $PWD/builds/tp:/build:ro -v $PWD/models:/models:ro \
#   -v $PWD/llama.cpp/docs/gfx906/scripts/f2-custom-ar-ab.sh:/ab.sh:ro \
#   --entrypoint bash qwen-flash-llama-builder:v1 /ab.sh
set -u
export LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib
export HSA_FORCE_FINE_GRAIN_PCIE=1
M=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
C="-m $M -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -sm tensor -p 512,2048 -n 128 -r 3 -o md"

echo "=== A: 自定义 AR 开（默认）==="
./llama-bench $C; echo "exitA=$?"

echo "=== B: 自定义 AR 关（GGML_ENABLE_CUSTOM_AR=0）==="
GGML_ENABLE_CUSTOM_AR=0 ./llama-bench $C; echo "exitB=$?"

echo "=== C: 仅取证（-v 才打印 AR 初始化路径与自动路径选择）==="
./llama-bench -m $M -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -sm tensor -p 512 -n 32 -r 1 -o md -v
echo "exitC=$?"
echo "=== DONE ==="
