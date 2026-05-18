#!/bin/bash
#SBATCH --job-name=phishingsecurity
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=DistilBERTforphishing.out
#SBATCH --error=DistilBERTforphishing.err
module load cuda/12.1
conda activate phishenv
python main.py