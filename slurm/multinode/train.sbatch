#!/bin/bash
#SBATCH --nodes=4
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=16
#SBATCH -p a100
#SBATCH -t 30-00:00:00
#SBATCH -o logs/llmft-vltest-train/%A_%a.out
#SBATCH -e logs/llmft-vltest-train/%A_%a.err
#SBATCH -J Qwen25-7B-VL-Train
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
DISABLE_VERSION_CHECK=1

# TODO: change settings here
SAVE_DIR="saves/qwen25vl-7b"				# root dir, -> accd. to base model
# model alias = prefix + dataset + suffix
MODEL_ALIAS="deepspeed-test"
MODEL="$SAVE_DIR/$MODEL_ALIAS/${MODEL_ALIAS}"
# one array item = one training = one model

TRAIN_CONFIG=examples/train_full/qwen2_5vl_full_sft.yaml
echo "GPUs: $CUDA_VISIBLE_DEVICES"
set -x

FORCE_TORCHRUN=1 llamafactory-cli train $TRAIN_CONFIG \
	output_dir=$MODEL
