#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_SERVICE_URL="${MODEL_SERVICE_URL:-http://127.0.0.1:8993}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/recipe/time_series_forecast/base_short.yaml}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/datasets/ETTM1/train.parquet}"
VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/datasets/ETTM1/test.parquet}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3-4B}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-30}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${TENSOR_MODEL_PARALLEL_SIZE:-1}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
PROJECT_NAME="${PROJECT_NAME:-TimeSeriesForecast}"
EXP_NAME="${EXP_NAME:-ETTM1-Qwen3-4B}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"

export PYTHONPATH="$PROJECT_ROOT/verl:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_SERVICE_URL
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

exec env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$PYTHON_BIN" -m arft.main_agent_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_PARALLEL_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$CONFIG_PATH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.device="$DEVICE" \
    "$@"
