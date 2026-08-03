#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# 教学实验脚本 —— 基于 run-qwen3-0.6B-1xgpu-grpo.sh，把几个参数改成可通过环境变量开关，
# 用来做对照实验。原脚本保持不动。
#
# 可调开关:
#   EXP_NAME          实验名，写进日志/tensorboard 名字里，便于区分
#   GLOBAL_BATCH_SIZE 默认 16。设成 32 则每轮只有 1 个训练 step
#   KL_MODE           off(默认) = 不加 --use-kl-loss；on = 加上并用 KL_COEF
#   KL_COEF           KL_MODE=on 时的系数，默认 0.1
#   SAVE_DIR          设了就保存 checkpoint，不设就不保存
#   SAVE_INTERVAL     每几个 train step 存一次，默认 5

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
EXP_NAME="${EXP_NAME:=exp}"
echo "当前时间: $now / 实验: $EXP_NAME"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/beginner-task}"
MODEL_DIR="${MODEL_DIR:-/your/model}"
DATA_DIR="${DATA_DIR:-/your/data}"

GSM8K_RAW="${DATA_DIR}/gsm8k/main/train-00000-of-00001.parquet"
GSM8K_CLEAN="${DATA_DIR}/gsm8k/main/train_clean.parquet"
if [ ! -f "${GSM8K_CLEAN}" ]; then
    python3 - <<EOF
import pandas as pd
df = pd.read_parquet("${GSM8K_RAW}")
df["answer"] = df["answer"].str.split("####").str[-1].str.strip()
df.to_parquet("${GSM8K_CLEAN}", index=False)
print(f"Wrote {len(df)} rows to ${GSM8K_CLEAN}")
EOF
fi

NUM_ROLLOUT="${NUM_ROLLOUT:=5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:=4}"
N_SAMPLES="${N_SAMPLES:=8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:=16}"

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-0.6B
    --ref-load ${MODEL_DIR}/Qwen3-0.6B
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

# ── 实验开关 3: 保存 checkpoint ──────────────────────────────────────────
SAVE_ARGS=()
if [ -n "${SAVE_DIR:-}" ]; then
    SAVE_ARGS=(
        --save "${SAVE_DIR}"
        --save-interval "${SAVE_INTERVAL:=5}"
    )
fi

ROLLOUT_ARGS=(
    --prompt-data ${GSM8K_CLEAN}
    --input-key question
    --label-key answer
    --apply-chat-template
    --rollout-shuffle

    --rm-type math

    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
    --n-samples-per-prompt ${N_SAMPLES}
    --rollout-max-response-len 2048
    --rollout-temperature 1

    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --balance-data
    --use-fault-tolerance
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --calculate-per-token-loss
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --log-probs-max-tokens-per-gpu 8192
)

# ── 实验开关 2/4: KL loss 开关与系数 ────────────────────────────────────
GRPO_ARGS=(
    --advantage-estimator grpo
    --entropy-coef 0.00
    --eps-clip 0.2

    --use-rollout-logprobs
)
KL_MODE="${KL_MODE:=off}"
if [ "${KL_MODE}" = "on" ]; then
    GRPO_ARGS+=(
        --use-kl-loss
        --kl-loss-coef "${KL_COEF:=0.1}"
        --kl-loss-type low_var_kl
    )
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.45
)

WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name qwen3-0.6b-${EXP_NAME}-${now}
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-0.6b-${EXP_NAME}-${now}.log
