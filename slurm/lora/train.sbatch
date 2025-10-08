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
datasets=()

# TODO: change settings here
SAVE_DIR="saves/qwen3-8b"				# root dir, -> accd. to base model
LORA_RANK=16							# LoRA rank
# model alias = prefix + dataset + suffix
MODEL_PREFIX="lora$LORA_RANK-"
MODEL_SUFFIX=""
MODEL_ALIAS="${MODEL_PREFIX}${DATASET}${MODEL_SUFFIX}"
MODEL="$SAVE_DIR/$MODEL_ALIAS/${MODEL_ALIAS}"
# one array item = one training = one model
datasets+=("sentencing_0926")

TRAIN_CONFIG=examples/train_lora/qwen3_lora16_sft.yaml
echo "GPUs: $CUDA_VISIBLE_DEVICES"
set -x

DATASET=${datasets[$SLURM_ARRAY_TASK_ID]}
llamafactory-cli train $TRAIN_CONFIG \
	adapter_name_or_path=$SAVE_DIR/lora-16-pk/sft \
	lora_rank=$LORA_RANK \
	dataset="${DATASET}_train" \
	eval_dataset="${DATASET}_val" \
	output_dir=$MODEL