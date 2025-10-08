#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=16
#SBATCH -p a100
#SBATCH -t 30-00:00:00
#SBATCH -o logs/vllm-infer/%A_%a.out
#SBATCH -e logs/vllm-infer/%A_%a.err
#SBATCH -J vllm-infer
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
###SBATCH --array=0-0
datasets=()

# TODO: change settings here
SAVE_DIR="saves/qwen25vl-7b"				# root dir, -> accd. to base model
# model path = root dir / model alias / model alias
MODEL_ALIAS="Qwen2.5-VL-7B-grpo-geo3k-hf"
MODEL="$SAVE_DIR/$MODEL_ALIAS/${MODEL_ALIAS}"
#MODEL="/public/models/Qwen2.5-VL-3B-Instruct"
# one array item = one inference = one dataset
datasets+=("mmhalbench")
datasets+=("mmvet")
datasets+=("ocrbench")
datasets+=("vizwiz")

set -x

MODEL_BASENAME=$(basename $MODEL)
DATASET=${datasets[$SLURM_ARRAY_TASK_ID]}
mkdir -p $SAVE_DIR/$MODEL_BASENAME/$DATASET
WANDB_DISABLED="true" python scripts/vllm_infer.py \
    --model_name_or_path $MODEL \
    --dataset $DATASET \
    --template "qwen2_vl" \
    --cutoff_len 24576 \
    --max_new_tokens 8192 \
    --save_name $SAVE_DIR/$MODEL_BASENAME/$DATASET/generated_predictions.jsonl