#!/bin/bash


set -x
export TORCH_HOME=../../pretrained_models
mkdir exp
mkdir slurm_log
#export CUDA_VISIBLE_DEVICES=1  # Use only the first GPU


task=pretrain_joint
mask_patch=300

# audioset and librispeech
dataset=asli
tr_data=/engram/naplab/shared/ssamba/datafiles/audioset_librispeech.json
te_data=/engram/naplab/shared/ssamba/datafiles/eval_data.json
dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=1024
num_mel_bins=128

model_size=base
# no patch split overlap
fshape=16
tshape=16
fstride=${fshape}
tstride=${tshape}


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

# no class balancing as it implicitly uses label information
bal=none
batch_size=64
lr=1e-4
# learning rate decreases if the pretext task performance does not improve on the validation set
lr_patience=2
epoch=10
# no spectrogram masking
freqm=0
timem=0
# no mixup training
mixup=0

exp_dir=./exp/amba-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}
# Check for existing model and optimizer state
model_path="$exp_dir/models/audio_model.lastmodel.pth"
optimizer_path="$exp_dir/models/optim_state.pth"
resume_args=""

if [[ -f $model_path && -f $optimizer_path ]]; then
    resume_args="--resume-model $model_path --resume-optimizer $optimizer_path"
fi

CUDA_CACHE_DISABLE=1 python -W ignore ../resume_amba.py --use_wandb $resume_args --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv /home/ss6928/ssamba/src/finetune/audioset/data/class_labels_indices.csv \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
--task ${task} --lr_patience ${lr_patience} --epoch_iter 4000 --patch_size ${patch_size} --embed_dim ${embed_dim} --depth ${depth} \
--rms_norm ${rms_norm} --residual_in_fp32 ${residual_in_fp32} \
--fused_add_norm ${fused_add_norm} --if_rope ${if_rope} --if_rope_residual ${if_rope_residual} \
--bimamba_type ${bimamba_type} --use_middle_cls_token ${use_middle_cls_token} --drop_path_rate ${drop_path_rate} --stride ${stride} --channels ${channels} --num_classes ${num_classes} --drop_rate ${drop_rate} --norm_epsilon ${norm_epsilon} --if_bidirectional ${if_bidirectional} --final_pool_type ${final_pool_type} --if_abs_pos_embed ${if_abs_pos_embed} --if_bimamba ${if_bimamba} --if_cls_token ${if_cls_token} --if_devide_out ${if_devide_out} --use_double_cls_token ${use_double_cls_token} --use_middle_cls_token ${use_middle_cls_token}
