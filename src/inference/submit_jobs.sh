#!/bin/bash

# List of model sizes to run
MODEL_SIZES=("tiny" "base" "small")

# Submit a Slurm job for each model size
for MODEL_SIZE in "${MODEL_SIZES[@]}"
do
    echo "Submitting job for model size: $MODEL_SIZE"
    sbatch run_inference_amba.sh $MODEL_SIZE
done

echo "All jobs submitted."
