#!/bin/bash

# Array of pre-trained models
declare -a models=("amba_base_400" "amba_tiny_400" "amba_small_400" "amba_base_300" "amba_tiny_300" "amba_small_300" "amba_base_250" "amba_tiny_250" "amba_small_250")

# Submit a job for each model
for model in "${models[@]}"; do
    sbatch run_sc_amba.sh $model
done

