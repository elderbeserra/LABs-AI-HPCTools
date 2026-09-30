#!/bin/bash
#SBATCH --job-name=bert_baseline
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00

source "$STORE/mypython/bin/activate"

python BASELINE/train_baseline.py
