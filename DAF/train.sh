#!/bin/bash

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=AD-VTAB
CONFIG=$1
GPUS=1
CKPT=$2
# WEIGHT_DECAY=0.0001

GPUS_PER_NODE=1
CPUS_PER_TASK=5

mkdir -p logs
mkdir -p csvs

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

low_rank_dim=8
SEED=0
dynamic_update_interval=10
param_budget=0.2
sensitivity_exp_name="dynamic_spt_lora_a${alpha}_b${beta}"

for LR in 0.0003
do
    for DATASET in cifar caltech101 dtd oxford_flowers102 svhn sun397
    do
        for WEIGHT_DECAY in 0.0001
        do
            exp_name=vtab_dynamic_spt_lora_a${alpha}_b${beta}_i${dynamic_update_interval}_p${param_budget}
            export MASTER_PORT=$((12000 + $RANDOM % 20000))
            python train_dynamic_spt.py --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET} --model_name=vit_base_patch16_224_in21k_spt --resume=checkpoints/ViT-B_16.npz --output_dir=./saves/${DATASET}_dynamic_spt_lr-${LR}_wd-${WEIGHT_DECAY} --batch-size=64 --lr=${LR} --epochs=200 --weight-decay=${WEIGHT_DECAY} --no_aug --mixup=0 --cutmix=0 --direct_resize --smoothing=0 --launcher="none" --seed=${SEED} --val_interval=5  --opt=adamw --low_rank_dim=${low_rank_dim} --initial_sensitivity_path=sensitivity_${sensitivity_exp_name}/${DATASET}/param_req_${param_budget}.pth --exp_name=${currenttime}-${exp_name} --seed=0 --test --block=BlockSPTParallel  --structured_type=lora --structured_vector --freeze_stage --dynamic_update_interval=${dynamic_update_interval} --param_budget=${param_budget} --alpha=${alpha} --beta=${beta} | tee -a logs/${currenttime}-${exp_name}.log
        done
    done
done 
