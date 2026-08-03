#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# 任务34 (Deterministic Trajectory Replay) 的探针实验。
# 在单卡 0.6B GRPO 基础上打开两个现有 debug dump,用来回答三个问题:
#   1. dump 里实际有哪些字段
#   2. 能否离线用 dumped 数据重算 loss
#   3. 重算值与日志 pg_loss 的偏差量级 → 定容差阈值
#
# 关键:开 --use-kl-loss 让 ref_log_probs 也进 dump(actor.py:1371 的条件),
# 否则少一个 loss 需要的字段。同时不开 --use-rollout-logprobs,
# 让 log_probs(旧策略)走 actor_fwd 那一路进 data_fields(actor.py:1369)。

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
EXP_NAME="${EXP_NAME:=replay-probe}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/beginner-task}"
MODEL_DIR="${MODEL_DIR:-/your/model}"
DATA_DIR="${DATA_DIR:-/your/data}"
DUMP_DIR="${DUMP_DIR:-/workspace/checkpoints/replay-probe}"
mkdir -p "${DUMP_DIR}"

GSM8K_CLEAN="${DATA_DIR}/gsm8k/main/train_clean.parquet"

NUM_ROLLOUT="${NUM_ROLLOUT:=2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:=4}"
N_SAMPLES="${N_SAMPLES:=8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:=32}"   # 32 → 每轮 1 个 step,advantage 不全零

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-0.6B
    --ref-load ${MODEL_DIR}/Qwen3-0.6B
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

# ── 探针核心:打开两个现有 dump ──────────────────────────────────────────
DUMP_ARGS=(
    --save-debug-rollout-data "${DUMP_DIR}/rollout_{rollout_id}.pt"
    --save-debug-train-data   "${DUMP_DIR}/train_r{rollout_id}_rank{rank}.pt"
)

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

# 开 KL 让 ref_log_probs 进 dump;不开 --use-rollout-logprobs 让 log_probs 也进
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.01
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
)

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
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${DUMP_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-0.6b-${EXP_NAME}-${now}.log
