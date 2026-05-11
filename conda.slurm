#!/bin/bash
#SBATCH --job-name=unidcase
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH--cpus-per-task=8
#SBATCH --gres=gpu:1

source $(conda info --base)/etc/profile.d/conda.sh
conda create --name phishenv python=3.9 -y
conda activate phishenv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets