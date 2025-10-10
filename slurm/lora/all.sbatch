#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=16
#SBATCH -p a100
#SBATCH -t 30-00:00:00
#SBATCH -o logs/llmft-lora-train/%A_%a.out
#SBATCH -e logs/llmft-lora-train/%A_%a.err
#SBATCH -J Qwen3-8B-LoRA-Train
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
###SBATCH --array=0-0
DISABLE_VERSION_CHECK=1
datasets=()

# TODO: change settings here
SAVE_DIR="saves/qwen3-8b"				# root dir, -> accd. to base model
LORA_RANK=32							# LoRA rank
# model alias = prefix + dataset + suffix
MODEL_PREFIX="lora$LORA_RANK-"
MODEL_SUFFIX=""
# one array item = one training = one model
datasets+=("shandong_alpaca")
datasets+=("shandong_alpaca_rag")
datasets+=("shandong_alpaca_digitonly")
datasets+=("shandong_alpaca_digitonly_rag")
datasets+=("shandong_pk_alpaca")
datasets+=("shandong_pk_alpaca_rag")
datasets+=("shandong_pk_alpaca_digitonly")
datasets+=("shandong_pk_alpaca_digitonly_rag")

TRAIN_CONFIG=examples/train_lora/qwen3_lora16_sft.yaml
INFER_CONFIG=examples/inference/qwen3_lora_eval.yaml
echo "GPUs: $CUDA_VISIBLE_DEVICES"
set -x

DATASET=${datasets[$SLURM_ARRAY_TASK_ID]}
MODEL_ALIAS="${MODEL_PREFIX}${DATASET}${MODEL_SUFFIX}"
MODEL="$SAVE_DIR/$MODEL_ALIAS/${MODEL_ALIAS}"
llamafactory-cli train $TRAIN_CONFIG \
	lora_rank=$LORA_RANK \
	dataset="${DATASET}_train" \
	eval_dataset="${DATASET}_val" \
	output_dir=$MODEL

WANDB_DISABLED="true" llamafactory-cli train $INFER_CONFIG \
    adapter_name_or_path=$MODEL \
    eval_dataset=${DATASET}_test \
    output_dir=$SAVE_DIR/$MODEL_ALIAS/${DATASET}_test