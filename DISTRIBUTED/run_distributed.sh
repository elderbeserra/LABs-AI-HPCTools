#!/bin/bash
#SBATCH --job-name=bert_ddp
#SBATCH --output=distributed_%j.out 
#SBATCH --error=distributed_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2 
#SBATCH --mem=64G
#SBATCH --time=00:30:00

source $STORE/mypython/bin/activate

srun python DISTRIBUTED/train_distributed.py