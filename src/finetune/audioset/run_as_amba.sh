#!/bin/bash
#SBATCH --job-name=as
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log


set -x
export TORCH_HOME=../../pretrained_models
mkdir exp

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate new_env_name



pretrain_exp="amba"
pretrain_model=$1
pretrain_path="/engram/naplab/shared/ssamba/models/${pretrain_model}.pth"

dataset=audioset
set=balanced
dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=1024
noise=False

task=ft_avgtok
model_size=base
head_lr=3
warmup=True

if [ $set == balanced ]
then
  bal=none
  lr=1e-4
  epoch=50
  tr_data=/engram/naplab/shared/ssamba/datafiles/bal_train_data.json
elif [ $set == full ]
then
  bal=bal
  lr=1e-5
  epoch=5
  tr_data=/engram/naplab/shared/ssamba/datafiles/bal_train_data.json
fi




patch_size=16
embed_dim=768
depth=24
rms_norm='false'
residual_in_fp32='false'
fused_add_norm='false'
if_rope='false'
if_rope_residual='false'
bimamba_type="v2"
drop_path_rate=0.1
stride=16
channels=1
num_classes=1000
drop_rate=0.
norm_epsilon=1e-5
if_bidirectional='true'
final_pool_type='none'
if_abs_pos_embed='true'
if_bimamba='false'
if_cls_token='true'
if_devide_out='true'
use_double_cls_token='false'
use_middle_cls_token='false'

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


te_data=/engram/naplab/shared/ssamba/datafiles/eval_modified_labeled_with_data_key.json
freqm=48
timem=192
mixup=0.5
fstride=10
tstride=10
fshape=16
tshape=16
batch_size=24
exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}-3

CUDA_CACHE_DISABLE=1 python -W ignore ../../run_amba.py --use_wandb --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 15 --lrscheduler_step 10 --lrscheduler_decay 0.75 --wa True --wa_start 12 --wa_end 40 \
--loss BCE --metrics mAP --embed_dim ${embed_dim} --depth ${depth} \
--rms_norm ${rms_norm} --residual_in_fp32 ${residual_in_fp32} \
--fused_add_norm ${fused_add_norm} --if_rope ${if_rope} --if_rope_residual ${if_rope_residual} \
--bimamba_type ${bimamba_type} --use_middle_cls_token ${use_middle_cls_token} --drop_path_rate ${drop_path_rate} --stride ${stride} --channels ${channels} --num_classes ${num_classes} --drop_rate ${drop_rate} --norm_epsilon ${norm_epsilon} --if_bidirectional ${if_bidirectional} --final_pool_type ${final_pool_type} --if_abs_pos_embed ${if_abs_pos_embed} --if_bimamba ${if_bimamba} --if_cls_token ${if_cls_token} --if_devide_out ${if_devide_out} --use_double_cls_token ${use_double_cls_token} 
