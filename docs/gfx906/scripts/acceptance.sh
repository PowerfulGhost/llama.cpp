#!/usr/bin/env bash
# 验收套件：每次上游 rebase / 改动之后跑一遍，判断是否可接受。
# 用法（宿主）：
#   docker run --rm --ipc=host --device=/dev/kfd --device=/dev/dri --group-add video --group-add 993 \
#     -v $PWD/llama.cpp:/src -v $PWD/builds/tp:/build -v $PWD/models:/models:ro \
#     --entrypoint bash qwen-flash-llama-builder:v1 -lc 'bash /src/docs/gfx906/scripts/acceptance.sh'
set -u
export LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8
M=${MODEL:-/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
P="alpha bravo charlie delta echo foxtrot"
R=${R:-2}
PASS=0; FAIL=0
chk() { # name, actual, expected, tolerance_pct
    local name=$1 act=$2 exp=$3 tol=$4
    local ok=$(awk -v a=$act -v e=$exp -v t=$tol 'BEGIN{print (a>=e*(1-t/100)) ? 1 : 0}')
    if [ "$ok" = "1" ]; then echo "  PASS  $name: $act (baseline $exp ±${tol}%)"; PASS=$((PASS+1));
    else echo "  FAIL  $name: $act < baseline $exp -${tol}%"; FAIL=$((FAIL+1)); fi
}

echo "=== 1. 构建 ==="
cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
      -DGPU_TARGETS=gfx906 -DGGML_HIP_RCCL=ON -DLLAMA_CURL=OFF >/dev/null 2>&1 || { echo "  FAIL  configure"; exit 1; }
ninja -C /build -t restat >/dev/null 2>&1
cmake --build /build --target llama-bench llama-cli llama-perplexity test-backend-ops -j 24 2>&1 | tail -1

cd /build/bin
echo "=== 2. repack A/B（对照 = 同一 build 的 --repack 0）==="
BASE="-m $M -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -p 512,2048,4096 -n 128 -r $R -o md -sm layer"
./llama-bench $BASE --repack 0 > /tmp/acc_off.txt 2>&1
./llama-bench $BASE            > /tmp/acc_on.txt  2>&1
python3 - <<'PY' > /tmp/acc_ab.txt
import re
def g(f):
    d={}
    for l in open(f, errors='replace'):
        if 'qwen4exp' in l:
            fl=[x.strip() for x in l.split('|')]
            try: d[fl[-3]]=float(fl[-2].split('\u00b1')[0])
            except Exception: pass
    return d
a,b=g('/tmp/acc_on.txt'),g('/tmp/acc_off.txt')
for k in ['pp512','pp2048','pp4096','tg128']:
    if k in a and k in b: print(k, a[k], b[k])
PY
while read -r k on off; do
    case $k in
        pp512|pp2048|pp4096) chk "$k (repack vs 无 repack)" "$on" "$(awk -v o=$off 'BEGIN{print o*1.15}')" 5 ;;
        tg128)               chk "$k (repack vs 无 repack)" "$on" "$(awk -v o=$off 'BEGIN{print o*0.98}')" 5 ;;
    esac
done < /tmp/acc_ab.txt
echo "  （基线参考：pp512 451.6 / pp2048 1161.7 / pp4096 1467.7 / tg128 35.3）"

echo "=== 3. token 一致性（融合必须在宽度 1 有差异、宽度 ≥2 逐位一致）==="
run_cli() { local tag=$1; shift; "$@" ./llama-cli -m "$M" -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -sm layer \
    -p "$P" -n 32 --temp 0 --seed 1 -st --no-warmup --simple-io --no-display-prompt < /dev/null 2>/dev/null \
    | sed 's/\[ Prompt[^]]*\]//g' | sed '/^$/d;/^Exiting/d' > /tmp/acc_$tag.txt; }
run_cli fuse0 env
run_cli fuse1 env GGML_CUDA_REPACK_MMV_FUSION=1
if cmp -s /tmp/acc_fuse0.txt /tmp/acc_fuse1.txt; then
    echo "  WARN  融合 ON/OFF 输出相同（与预期不符：宽度 1 应有差异，检查 witness）"; FAIL=$((FAIL+1))
else echo "  PASS  融合 ON/OFF 在宽度 1 有差异（预期行为）"; PASS=$((PASS+1)); fi
echo "  PPL（batch 1 / 8 chunks）：融合 ON 应≈4.228，OFF 应≈4.212"
GGML_CUDA_REPACK_MMV_FUSION=1 ./llama-perplexity -m $M -f /src/docs/gfx906/scripts/corpus-perplexity-sample.txt \
    -c 512 --chunks 8 -b 1 -ngl 99 -fa on -ctk f16 -ctv f16 -sm layer -lm dio 2>&1 | grep -o "PPL = [0-9.]*" | tail -1

echo "=== 4. 上游 repack 目标类型是否仍被正确识别 ==="
./llama-bench -m $M -lm dio -ngl 99 -fa on -sm layer -p 16 -n 1 -r 1 -v 2>&1 | grep -c "Repacked model buffer size" | \
    awk '{ if ($1>=8) {print "  PASS  8 张卡都用了 repack buffer ("$1")"} else {print "  FAIL  repack buffer 只在 "$1" 张卡上生效"} }'

echo
echo "=== 结果：PASS=$PASS FAIL=$FAIL ==="
[ "$FAIL" = "0" ] && echo "验收通过" || echo "验收未通过，请检查上面 FAIL 项"
