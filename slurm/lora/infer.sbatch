#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=16
#SBATCH -p a100
#SBATCH -t 30-00:00:00
#SBATCH -o logs/llmft-lora-infer/%A_%a.out
#SBATCH -e logs/llmft-lora-infer/%A_%a.err
#SBATCH -J Qwen3-8B-LoRA-Infer
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
###SBATCH --array=0-0
DISABLE_VERSION_CHECK=1
models=()
datasets=()

# TODO: change settings here
SAVE_DIR="saves/qwen3-8b"				# root dir, -> accd. to base model
# model path = root dir / model alias / model alias
models+=("lora32-shandong_alpaca")		# model alias
# one array item = one inference = one dataset
datasets+=("lora32-shandong_alpaca_test")

INFER_CONFIG=examples/inference/qwen3_lora_eval.yaml
echo "GPUs: $CUDA_VISIBLE_DEVICES"

set -x

MODEL_ALIAS=${models[$SLURM_ARRAY_TASK_ID]}
DATASET=${datasets[$SLURM_ARRAY_TASK_ID]}
WANDB_DISABLED="true" llamafactory-cli train $INFER_CONFIG \
    adapter_name_or_path=$SAVE_DIR/$MODEL_ALIAS/${MODEL_ALIAS} \
    eval_dataset=$DATASET \
    output_dir=$SAVE_DIR/$MODEL_ALIAS/${DATASET}