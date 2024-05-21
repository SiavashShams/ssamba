#!/bin/bash

# Array of pre-trained models
declare -a models=("ssast_tiny_400")

# Submit a job for each model
for model in "${models[@]}"; do
    sbatch run_as.sh $model
done

