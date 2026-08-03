#!/bin/bash
# 出问题时截取日志片段，输出直接复制粘贴（不需要传文件）
#
#   bash _validation/excerpt.sh
#
# 每份日志各截：最后 40 行、第一处 Traceback 前后各 30 行、所有错误行。

set -u

echo "## 8. 故障片段"
echo

for f in /tmp/rloo-cp1-dp8.log /tmp/rloo-cp2-dp4.log /tmp/rloo-cp4-dp2.log /tmp/rloo-negative.log; do
    [ -f "$f" ] || continue
    echo "### $(basename "$f")"
    echo

    echo '<details><summary>错误行汇总</summary>'
    echo
    echo '```'
    grep -nE "Error|Exception|assert|Traceback|NCCL|can't be cast|shape mismatch|CUDA" "$f" 2>/dev/null \
        | grep -viE "cache_position|not documented|docstring|flash_attn|NVSHMEM|NCCL_|\.\.\.\.+ (True|False|None)" \
        | head -30
    echo '(以上为空说明没有明显错误行)'
    echo '```'
    echo
    echo '</details>'
    echo

    if grep -qn "Traceback" "$f" 2>/dev/null; then
        first=$(grep -n "Traceback" "$f" | head -1 | cut -d: -f1)
        echo '<details><summary>第一处 Traceback 上下文</summary>'
        echo
        echo '```'
        sed -n "$((first > 30 ? first - 30 : 1)),$((first + 30))p" "$f"
        echo '```'
        echo
        echo '</details>'
        echo
    fi

    echo '<details><summary>最后 40 行</summary>'
    echo
    echo '```'
    tail -40 "$f"
    echo '```'
    echo
    echo '</details>'
    echo
done
