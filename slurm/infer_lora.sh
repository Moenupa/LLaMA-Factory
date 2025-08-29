#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=16
#SBATCH -p a100
#SBATCH -t 30-00:00:00
#SBATCH -o logs/legal-lora-infer/%j.out
#SBATCH -e logs/legal-lora-infer/%j.err
#SBATCH -J Legal-Qwen3-8B-LoRA-Infer
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0

# major settings
SAVE_PATH="saves/qwen3-8b"
MODEL="$SAVE_PATH/lora-16-pk/sft"
datasets=("sentencing_math12k_test" "sentencing_pk_math12k_test")

# get proper params
INFER_CONFIG=examples/inference/qwen3_lora_eval.yaml
echo "GPUs assigned by Slurm: $CUDA_VISIBLE_DEVICES, $INFER_CONFIG"

set -x

MODEL_BASENAME=$(basename $MODEL)
MODEL_ALIAS=$(basename $(dirname $MODEL))
DATASET=${datasets[$SLURM_ARRAY_TASK_ID]}
WANDB_DISABLED="true" llamafactory-cli train $INFER_CONFIG \
    adapter_name_or_path=$MODEL \
    eval_dataset=$DATASET \
    output_dir=$SAVE_PATH/$MODEL_ALIAS/$DATASET