#!/bin/bash
#SBATCH -o logs/lora-infer/%j.out
#SBATCH -e logs/lora-infer/%j.err
#SBATCH -J Qwen3-lora-8b-infer-sentencing
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=7-00:00:00

INFER_CONFIG=examples/inference/qwen3_rag_eval.yaml

echo "GPUs assigned by Slurm: $CUDA_VISIBLE_DEVICES, $INFER_CONFIG"
cat $INFER_CONFIG

echo "------------------------------------------------------------------------------"

llamafactory-cli train $INFER_CONFIG