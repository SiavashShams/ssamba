#!/bin/bash

for model in ssast_patch400_tiny; do
        sbatch run_sid.sh $model 1e-4
done
