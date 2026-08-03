#!/bin/bash
# 多卡验证：一次跑完 A/B/C/D 四组并生成 RESULT.md
#
# 用法（在 Relax 仓库根目录）:
#   MODEL_DIR=/path/to/model DATA_DIR=/path/to/data bash _validation/run_all.sh
#
# MODEL_DIR 下需有 Qwen3-0.6B/，DATA_DIR 下需有 gsm8k/main/train-00000-of-00001.parquet
#
# 跑完后报告在 ./RESULT.md，原始日志在 /tmp/rloo-*.log

set -u

RECIPE=examples/algorithms/run-qwen3-0.6B-1xgpu-gsm8k-rloo.sh
: "${MODEL_DIR:?请设置 MODEL_DIR（Qwen3-0.6B 的父目录）}"
: "${DATA_DIR:?请设置 DATA_DIR（gsm8k 的父目录）}"
export MODEL_DIR DATA_DIR
export NUM_ROLLOUT="${NUM_ROLLOUT:=3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"

# GPUs per role. The recipe defaults to the single-GPU {"actor":[1,1],"rollout":[1,1]},
# which cannot run CP>1: Megatron requires world_size % (tp*pp*cp) == 0, so a CP=2
# group would die in validate_args with "world size (1) is not divisible by
# total_model_size (2)". Every group here therefore needs all visible GPUs.
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
    # 组级环境变量以 KEY=VAL 形式传入，只对这一次调用生效
    if env "$@" bash "$RECIPE" >"$log" 2>&1; then
        echo "EXIT=0" >>"$log"
        echo "  组 $name 完成 (EXIT=0)"
    else
        echo "EXIT=$?" >>"$log"
        echo "  组 $name 结束，非零退出 —— 见 $log"
    fi
}

echo "NUM_ROLLOUT=$NUM_ROLLOUT  GPUS=$CUDA_VISIBLE_DEVICES"

# A: CP=1, DP=8 —— 三个 batch 变量必须一起改（RLOO 要求 rbs*n_samples==gbs）
run_group A /tmp/rloo-cp1-dp8.log ROLLOUT_BATCH_SIZE=8 GLOBAL_BATCH_SIZE=64 MICRO_BATCH_SIZE=8

# B: CP=2, DP=4
run_group B /tmp/rloo-cp2-dp4.log CONTEXT_PARALLEL_SIZE=2

# C: CP=4, DP=2
run_group C /tmp/rloo-cp4-dp2.log CONTEXT_PARALLEL_SIZE=4

# D: 负向，预期启动即失败（4*8=32 != 16，隐含两个 optimizer step）
run_group D /tmp/rloo-negative.log CONTEXT_PARALLEL_SIZE=2 GLOBAL_BATCH_SIZE=16

cleanup

echo ""
echo "══════ 生成 RESULT.md ══════"
bash _validation/collect.sh >RESULT.md
echo "完成。请打开 RESULT.md，补完第 5-7 节后把 全文 复制发回。"
echo ""
echo "自动判定摘要："
sed -n '/## 4. 跨组一致性/,$p' RESULT.md | head -20
