#!/bin/bash

#SBATCH --job-name=amba_er
#SBATCH --gres=gpu:a40:1         # Request an A40 GPU
#SBATCH --cpus-per-task=12        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log

set -x
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate superv
export TORCH_HOME=../../pretrained_models

# Default parameters
model_function=${1:-ssast_patch400_base}  # default to 'ssast_patch400_base'
lr=${2:-1e-5}

expname=emotion_${model_function}_${lr}
expdir=./exp/$expname
mkdir -p $expdir

#for test_fold in fold1 fold2 fold3 fold4 fold5;
for test_fold in fold1;
do
  echo "running cross-validation on $test_fold"
  mkdir -p $expdir/unfreeze_cross-valid-on-${test_fold}; mkdir -p ./log/emotion/unfreeze_cross-valid-on-${test_fold}
  python3 ~/courses/adv_dl/final/s3prl/s3prl/run_downstream.py --expdir $expdir/unfreeze_cross-valid-on-${test_fold} -m train -u $mdl -d emotion -c ~/courses/adv_dl/final/s3prl/s3prl/downstream/emotion/config.yaml -s hidden_states -o "config.downstream_expert.datarc.test_fold='$test_fold'" -f
done
