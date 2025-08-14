#!/bin/bash
#SBATCH -o logs/full-infer/%j.out
#SBATCH -e logs/full-infer/%j.err
#SBATCH -J Qwen3-full-8b-infer-sentencing
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=7-00:00:00

INFER_CONFIG=examples/inference/qwen3_eval.yaml

echo "GPUs assigned by Slurm: $CUDA_VISIBLE_DEVICES"
echo "starting inference with config: $INFER_CONFIG"
cat $INFER_CONFIG

echo "------------------------------------------------------------------------------"

# python scripts/vllm_infer.py --model_name_or_path saves/qwen3-8b/full/sft --dataset sentencing_test 
llamafactory-cli train $INFER_CONFIG