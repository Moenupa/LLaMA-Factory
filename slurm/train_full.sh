#!/bin/bash
#SBATCH -o logs/full-sft/%j.out
#SBATCH -e logs/full-sft/%j.err
#SBATCH -J Qwen3-full-8b-sft-sentencing
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=7-00:00:00

TRAINING_CONFIG=examples/train_full/qwen3_full_sft_8b.yaml

echo "GPUs $CUDA_VISIBLE_DEVICES, $TRAINING_CONFIG"
cat $TRAINING_CONFIG
echo "------------------------------------------------------------------------------"

# this does not work
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 llamafactory-cli train $TRAINING_CONFIG

# this works
accelerate launch --config_file examples/accelerate/fsdp_config.yaml src/train.py $TRAINING_CONFIG

# not really working
# accelerate launch --config_file sh/default_config.yaml src/train.py $TRAINING_CONFIG
