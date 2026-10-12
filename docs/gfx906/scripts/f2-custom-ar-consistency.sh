#!/usr/bin/env bash
# 并行/数值路径改动（如 F2 自定义 AllReduce）的 token 一致性检查。
#
# 用法同上见 f2-custom-ar-ab.sh 的 docker 头注。要点：
#   - 只在「生成的文本」上比对：性能行 [ Prompt: ... ] 每次必然不同，必须剔除
#   - 贪心解码：--temp 0；固定 --seed 1；-st 单轮 + </dev/null 避免进入交互
#   - prompt 用词表式内容（短的自然语言可能触发首个 token 就 EOS 的随机行为）
#   - 想看到 AR 初始化路径，给 llama-cli 加 -v（默认 verbosity 会吞掉 INFO）
set -u
export LD_LIBRARY_PATH=/build/bin:/opt/rocm/lib
M=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
P="alpha bravo charlie delta echo foxtrot"

text_only() { tr '\r' '\n' < "$1" | grep -vE "^\[ Prompt:" | grep -vE "^[[:space:]]*$"; }

run() {
    local tag=$1; shift
    "$@" ./llama-cli -m "$M" -lm dio -ngl 99 -fa on -ctk f16 -ctv f16 \
        -p "$P" -n 32 --temp 0 --seed 1 -st --no-warmup --simple-io \
        < /dev/null > "/tmp/c2.$tag.out" 2> "/tmp/c2.$tag.err"
    echo "$tag: exit=$? text_md5=$(text_only "/tmp/c2.$tag.out" | md5sum | cut -d' ' -f1)"
}

run layer      env                                                        # 参考：-sm layer（PP8）
run tp_custom  env HSA_FORCE_FINE_GRAIN_PCIE=1                            # TP8 + 自定义 AR
run tp_rcc     env GGML_ENABLE_CUSTOM_AR=0 HSA_FORCE_FINE_GRAIN_PCIE=1    # TP8 + RCCL

echo "=== 是否逐字节一致（只比生成文本）==="
for t in tp_custom tp_rcc; do
    if diff <(text_only /tmp/c2.layer.out) <(text_only "/tmp/c2.$t.out") >/dev/null; then
        echo "layer vs $t: 一致 ✅"
    else
        echo "layer vs $t: 不一致 ❌"; diff <(text_only /tmp/c2.layer.out) <(text_only "/tmp/c2.$t.out")
    fi
done
if diff <(text_only /tmp/c2.tp_custom.out) <(text_only /tmp/c2.tp_rcc.out) >/dev/null; then
    echo "tp_custom vs tp_rcc: 一致 ✅"
else
    echo "tp_custom vs tp_rcc: 不一致 ❌"
    diff <(text_only /tmp/c2.tp_custom.out) <(text_only /tmp/c2.tp_rcc.out)
fi

echo "=== 生成的文本 ==="
text_only /tmp/c2.tp_custom.out | tail -4
echo "=== DONE ==="
