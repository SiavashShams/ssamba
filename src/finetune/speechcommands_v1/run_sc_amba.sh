#!/bin/bash
#SBATCH --job-name=esc
#SBATCH --gres=gpu:l40:1         # Request an L40 GPU
#SBATCH --cpus-per-task=8        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log

set -x
export TORCH_HOME=../../pretrained_models
mkdir exp

# prep speechcommands dataset and download the pretrained model
if [ -e data/datafiles ]
then
    echo "speechcommands already downloaded and processed."
else
    python prep_sc.py
fi

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate new_env_name



pretrain_exp="amba"
pretrain_model=$1
pretrain_path="/engram/naplab/shared/ssamba/models/${pretrain_model}.pth"

dataset=speechcommands
dataset_mean=-6.845978
dataset_std=5.5654526
target_length=128
noise=True
tr_data=./data/datafiles/speechcommand_train_data.json
val_data=./data/datafiles/speechcommand_valid_data.json
eval_data=./data/datafiles/speechcommand_eval_data.json

bal=none
lr=2.5e-4
freqm=48
timem=48
mixup=0.6
epoch=30
batch_size=128
fshape=16
tshape=16
fstride=10
tstride=10

task=ft_avgtok
model_size=base
if [[ $pretrain_model == *"tiny"* ]]; then
  model_size="tiny"
  embed_dim=192
elif [[ $pretrain_model == *"small"* ]]; then
  model_size="small"
  embed_dim=384
else
  model_size="base"
  embed_dim=768
fi

head_lr=1

base_exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}


pretrain_path=./${pretrain_exp}/${pretrain_model}.pth
exp_dir=./exp/test01-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${task}-${model_size}-$pretrain_exp-${pretrain_model}-${head_lr}x-noise${noise}

CUDA_CACHE_DISABLE=1 python -W ignore ../../run_amba.py --use_wandb --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/speechcommands_class_labels_indices.csv --n_class 30 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --embed_dim ${embed_dim} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} --if_cls_token 'true' --final_pool_type 'mean' --if_abs_pos_embed 'true' --if_devide_out 'true' --use_middle_cls_token 'true' \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss BCE --metrics acc --embed_dim ${embed_dim} --depth ${depth} \
--rms_norm ${rms_norm} --residual_in_fp32 ${residual_in_fp32} \
--fused_add_norm ${fused_add_norm} --if_rope ${if_rope} --if_rope_residual ${if_rope_residual} \
--bimamba_type ${bimamba_type} --use_middle_cls_token ${use_middle_cls_token} --drop_path_rate ${drop_path_rate} --stride ${stride} --channels ${channels} --num_classes ${num_classes} --drop_rate ${drop_rate} --norm_epsilon ${norm_epsilon} --if_bidirectional ${if_bidirectional} --final_pool_type ${final_pool_type} --if_abs_pos_embed ${if_abs_pos_embed} --if_bimamba ${if_bimamba} --if_cls_token ${if_cls_token} --if_devide_out ${if_devide_out} --use_double_cls_token ${use_double_cls_token} 
