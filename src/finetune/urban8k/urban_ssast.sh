#!/bin/bash
#SBATCH --job-name=urban8k
#SBATCH --gres=gpu:a40:1
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
pretrain_path="/engram/naplab/shared/ssast/models/${pretrain_model}.pth"

dataset=urban8k
set=balanced
dataset_mean=-5.138443946838379
dataset_std=4.229961395263672
target_length=6144
noise=False

task=ft_avgtok_1sec
model_size=base
head_lr=1
warmup=True
lr=1e-4
bal=none
epoch=70

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

freqm=48
timem=192
mixup=0.0
fstride=16
tstride=16
fshape=16
tshape=16
batch_size=5

base_exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}

# Initialize arrays to hold the results
all_map=()
all_acc=()

for val_fold in {1..10}; do
  echo 'Now validating on fold'${val_fold}

  # Initialize arrays to hold the training data paths
  train_metadata_csv_paths=()
  train_audio_base_paths=()

  # Aggregate training data from all folds except the validation fold
  for fold in {1..10}; do
    if [ $fold -ne $val_fold ]; then
      train_metadata_csv_paths+=("/engram/naplab/shared/UrbanSound8K/audio/concatenated_fold${fold}/concatenated_metadata.csv")
      train_audio_base_paths+=("/engram/naplab/shared/UrbanSound8K/audio/concatenated_fold${fold}")
    fi
  done

  # Set validation fold paths
  val_metadata_csv_path="/engram/naplab/shared/UrbanSound8K/audio/concatenated_fold${val_fold}/concatenated_metadata.csv"
  val_audio_base_path="/engram/naplab/shared/UrbanSound8K/audio/concatenated_fold${val_fold}"

  # Convert arrays to colon-separated strings for passing as arguments
  train_metadata_csv_paths_str=$(IFS=:; echo "${train_metadata_csv_paths[*]}")
  train_audio_base_paths_str=$(IFS=:; echo "${train_audio_base_paths[*]}")

  exp_dir=${base_exp_dir}/fold${val_fold}

  # Run the training script for the current fold
  CUDA_CACHE_DISABLE=1 python -W ignore ../../run_ssast_1sec.py --use_wandb --dataset ${dataset} \
    --metadata_csv ${train_metadata_csv_paths_str} --audio_base_path ${train_audio_base_paths_str} --exp-dir $exp_dir \
    --label-csv ./data/urban8k_class_labels_indices.csv --n_class 10 \
    --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
    --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
    --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
    --model_size ${model_size} --adaptschedule False \
    --pretrained_mdl_path ${pretrain_path} \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
    --num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
    --lrscheduler_start 20 --lrscheduler_step 10 --lrscheduler_decay 0.85 --wa False --loss CE --metrics acc --embed_dim ${embed_dim} --depth ${depth} \
    --rms_norm ${rms_norm} --residual_in_fp32 ${residual_in_fp32} \
    --fused_add_norm ${fused_add_norm} --if_rope ${if_rope} --if_rope_residual ${if_rope_residual} \
    --bimamba_type ${bimamba_type} --use_middle_cls_token ${use_middle_cls_token} --drop_path_rate ${drop_path_rate} --stride ${stride} --channels ${channels} --num_classes ${num_classes} --drop_rate ${drop_rate} --norm_epsilon ${norm_epsilon} --if_bidirectional ${if_bidirectional} --final_pool_type ${final_pool_type} --if_abs_pos_embed ${if_abs_pos_embed} --if_bimamba ${if_bimamba} --if_cls_token ${if_cls_token} --if_devide_out ${if_devide_out} --use_double_cls_token ${use_double_cls_token} \
    --val_metadata_csv ${val_metadata_csv_path} --val_audio_base_path ${val_audio_base_path}

  # Extract the validation results
  map=$(tail -n 1 $exp_dir/metrics.csv | cut -d',' -f2) 
  acc=$(tail -n 1 $exp_dir/metrics.csv | cut -d',' -f3) 

  all_map+=($map)
  all_acc+=($acc)
done

# Compute the average mAP and accuracy
avg_map=$(echo "${all_map[@]}" | awk '{for(i=1;i<=NF;i++)sum+=$i; print sum/NF}')
avg_acc=$(echo "${all_acc[@]}" | awk '{for(i=1;i<=NF;i++)sum+=$i; print sum/NF}')

echo "Average mAP: $avg_map"
echo "Average Accuracy: $avg_acc"
