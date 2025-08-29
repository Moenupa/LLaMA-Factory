#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=16
#SBATCH -p a100
#SBATCH -t 30-00:00:00
#SBATCH -o logs/legal-lora-train/%j.out
#SBATCH -e logs/legal-lora-train/%j.err
#SBATCH -J Legal-Qwen3-8B-LoRA-Train
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0

# major settings
SAVE_PATH="saves/qwen3-8b"
LORA_RANK=16
MODEL_PREFIX="lora$LORA_RANK-"
MODEL_SUFFIX=""
datasets=("sentencing_gsm" "sentencing_pk_math12k")

# get training param template
TRAIN_CONFIG=examples/train_lora/qwen3_lora16_sft.yaml
echo "GPUs: $CUDA_VISIBLE_DEVICES"

set -x

DATASET=${datasets[$SLURM_ARRAY_TASK_ID]}
MODEL_ALIAS="$MODEL_PREFIX$DATASET$MODEL_SUFFIX"
llamafactory-cli train $TRAIN_CONFIG \
    lora_rank=$LORA_RANK \
    dataset="${DATASET}_train" \
    eval_dataset="${DATASET}_val" \
    output_dir=$SAVE_PATH/$MODEL_ALIAS/$MODEL_ALIAS