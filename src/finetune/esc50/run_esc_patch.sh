#!/bin/bash
#SBATCH --job-name=esc
#SBATCH --gres=gpu:l40:1         # Request an L40 GPU
#SBATCH --cpus-per-task=8        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log

set -x
export TORCH_HOME=../../pretrained_models
mkdir exp

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate new_env_name

pretrain_exp="ssast"
pretrain_model=$1
pretrain_path="/engram/naplab/shared/ssamba/models/${pretrain_model}.pth"


dataset=esc50
dataset_mean=-6.6268077
dataset_std=5.358466
target_length=512
noise=True

bal=none
lr=1e-4
freqm=24
timem=96
mixup=0
epoch=50
batch_size=48
fshape=16
tshape=16
fstride=10
tstride=10

task=ft_avgtok
if [[ $pretrain_model == *"tiny"* ]]; then
  model_size="tiny"
elif [[ $pretrain_model == *"small"* ]]; then
  model_size="small"
else
  model_size="base"
fi

head_lr=1

base_exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}

for((fold=1;fold<=5;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=/engram/naplab/shared/datafiles/esc_train_data_${fold}.json
  te_data=/engram/naplab/shared/datafiles/esc_eval_data_${fold}.json
  
  CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --use_wandb --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/esc_class_labels_indices.csv --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
  --model_size ${model_size} --adaptschedule False \
  --pretrained_mdl_path ${pretrain_path} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
  --num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
  --lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics acc
done

python ./get_esc_result.py --exp_path ${base_exp_dir}
