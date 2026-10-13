#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib HSA_FORCE_FINE_GRAIN_PCIE=1 GPU_MAX_HW_QUEUES=8
M=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
BODY='{"prompt":"alpha bravo charlie delta echo foxtrot golf hotel india","n_predict":128,"temperature":0,"cache_prompt":false,"ignore_eos":true}'
one() { curl -s -X POST http://127.0.0.1:8080/completion -H 'Content-Type: application/json' -d "$BODY" -o "$1"; }
sweep() {
  local mode=$1
  ./llama-server -m $M -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -sm $mode -c 32768 -np 8 --host 127.0.0.1 --port 8080 >/prof/srvb_$mode.log 2>&1 &
  local SP=$!
  for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"' && break; sleep 5; done
  one /dev/null 2>/dev/null
  for N in 1 2 4 8; do
    S=$(date +%s.%N)
    local pids=()
    for i in $(seq 1 $N); do one /prof/b_${mode}_${N}_$i.json & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    E=$(date +%s.%N)
    TOT=$(cat /prof/b_${mode}_${N}_*.json 2>/dev/null | grep -o '"tokens_predicted":[0-9]*' | grep -o '[0-9]*$' | awk '{s+=$1} END{print s+0}')
    awk -v s=$S -v e=$E -v t=$TOT -v n=$N -v m="$mode" 'BEGIN{printf "--- %-6s 并发 %2d: %.2f s, %4d tok → %6.1f tok/s 总吞吐, 单请求 %.1f tok/s\n", m, n, e-s, t, t/(e-s), (t/n)/(e-s)}'
  done
  kill $SP 2>/dev/null; sleep 6
}
sweep layer
sweep tensor
echo "=== DONE ==="
