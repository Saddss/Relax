#!/bin/bash
# 多卡验证第 2 轮：只跑 B(CP=2) / C(CP=4) 两组。
#
#   MODEL_DIR=/path/to/model DATA_DIR=/path/to/data bash _validation/run_round2.sh
#
# 与第 1 轮的差别：A 组不重跑（单卡已覆盖）。--calculate-per-token-loss 的新 guard
# 在参数校验阶段就会触发、不需要 GPU，已在开发机端到端验证过。
# 报告生成到 ./RESULT2.md

set -u

RECIPE=examples/algorithms/run-qwen3-0.6B-1xgpu-gsm8k-rloo.sh
: "${MODEL_DIR:?请设置 MODEL_DIR（Qwen3-0.6B 的父目录）}"
: "${DATA_DIR:?请设置 DATA_DIR（gsm8k 的父目录）}"
export MODEL_DIR DATA_DIR
export NUM_ROLLOUT="${NUM_ROLLOUT:=3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"

_ngpu=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
export RESOURCE="${RESOURCE:-{\"actor\": [1, $_ngpu], \"rollout\": [1, $_ngpu]\}}"

if [ ! -f "$RECIPE" ]; then
    echo "✗ 找不到 $RECIPE —— 请在 Relax 仓库根目录运行"
    exit 1
fi

cleanup() {
    ray stop --force >/dev/null 2>&1 || true
    pkill -9 sglang >/dev/null 2>&1 || true
    sleep 5
}

run_group() {
    local name=$1 log=$2
    shift 2
    echo ""
    echo "══════ 组 $name → $log ══════"
    cleanup
    if env "$@" bash "$RECIPE" >"$log" 2>&1; then
        echo "EXIT=0" >>"$log"
        echo "  组 $name 完成 (EXIT=0)"
    else
        echo "EXIT=$?" >>"$log"
        echo "  组 $name 结束，非零退出 —— 见 $log"
    fi
}

echo "NUM_ROLLOUT=$NUM_ROLLOUT  GPUS=$CUDA_VISIBLE_DEVICES  RESOURCE=$RESOURCE"

# B / C: 正向重跑，这次工作区干净
run_group B /tmp/rloo-cp2-dp4.log CONTEXT_PARALLEL_SIZE=2
run_group C /tmp/rloo-cp4-dp2.log CONTEXT_PARALLEL_SIZE=4

cleanup

echo ""
echo "══════ 生成 RESULT2.md ══════"
bash _validation/collect.sh >RESULT2.md
echo "完成。请打开 RESULT2.md，补完第 5-7 节后把全文复制发回。"
echo ""
sed -n '/## 4. 跨组一致性/,$p' RESULT2.md | head -20
