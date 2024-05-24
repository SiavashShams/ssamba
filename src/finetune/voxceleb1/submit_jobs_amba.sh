#!/bin/bash

for model in amba_patch250_tiny amba_patch300_tiny amba_patch400_tiny amba_patch250_small amba_patch300_small amba_patch400_small amba_patch250_base amba_patch300_base amba_patch400_base ; do
        sbatch run_sid.sh $model 1e-4
done
