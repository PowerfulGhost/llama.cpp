#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib
M=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
P="alpha bravo charlie delta echo foxtrot"
common() {
    if [ "$1" = on ]; then ARGS=(); else ARGS=(-nr); fi
    ./llama-cli -m "$M" -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 -sm layer "${ARGS[@]}" \
        -p "$P" -n 32 --temp 0 --seed 1 -st --no-warmup --simple-io --no-display-prompt \
        < /dev/null
}
run() { local tag=$1 mode=$2; common "$mode" > "/tmp/rp.$tag.out" 2> "/tmp/rp.$tag.err"; echo "$tag($mode) exit=$? out_bytes=$(stat -c%s /tmp/rp.$tag.out)"; }
run on1 on
run off off
run on2 on
echo "--- on1 的 stdout（后 200 字节）---"; tail -c 200 /tmp/rp.on1.out
echo; echo "--- off 的 stdout（后 200 字节）---"; tail -c 200 /tmp/rp.off.out
echo; echo "--- on2 的 stdout（后 200 字节）---"; tail -c 200 /tmp/rp.on2.out
echo; echo "=== 比较（只看生成的文本）==="
for a in off on2; do
    if diff -q <(sed 's/\[ Prompt[^]]*\]//g' /tmp/rp.on1.out) <(sed 's/\[ Prompt[^]]*\]//g' "/tmp/rp.$a.out") >/dev/null; then
        echo "$a vs on1: 一致 ✅"
    else
        echo "$a vs on1: 不一致 ❌"
    fi
done
echo "=== DONE ==="
