#!/bin/bash
#SBATCH -o logs/lora/%j.out
#SBATCH -e logs/lora/%j.err
#SBATCH -J Qwen3-LoRA-sft-sentencing
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=7-00:00:00

TRAIN_CONFIG=examples/train_lora/qwen3_lora16_rag_sft.yaml
TRAIN_CONFIG=examples/train_lora/qwen3_lora16_rag_gsm_sft.yaml

echo "GPUs: $CUDA_VISIBLE_DEVICES $TRAIN_CONFIG"
cat $TRAIN_CONFIG
echo "------------------------------------------------------------------------------"

llamafactory-cli train $TRAIN_CONFIG