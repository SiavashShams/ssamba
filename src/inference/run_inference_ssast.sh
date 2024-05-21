#!/bin/bash

#SBATCH --job-name=inference
#SBATCH --gres=gpu:l40:1         # Request an L40 GPU
#SBATCH --cpus-per-task=8        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate new_env_name


python inference_ssast.py --model_size $1
