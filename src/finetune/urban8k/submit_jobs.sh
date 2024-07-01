#!/bin/bash

# Array of pre-trained models
declare -a models=("amba_tiny_400" "amba_base_400" "amba_small_400")

# Submit a job for each model
for model in "${models[@]}"; do
    sbatch urban_amba.sh $model
done

