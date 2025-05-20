#!/bin/bash

MakeDir(){
    if ! [ -e "$1" ]
    then
        mkdir -p "$1"
    fi
}

cd /yuxintian-20phd/projects/fednoisy/FedNoisy

SEED=1
# DATA_DIR='../rawdata/cifar10'
# DATASET='cifar10'
# OUT_DIR='../centrNLLdata/cifar10'
LR=0.01
MOMENTUM=0.5
WEIGHT_DECAY=0.0005
MODEL='ResNet18'
EXP_NAME='observation'
WD_P_NAME='NLFL-ob-sym'
WD_J_TYPE='baseline'
WD_MODE='online'
NUM_EPOCHS=200
# NOISE_RATIO=0.1866
N_SETUP='n06ls24'
NOISE_MODE='sym'
# FL_DATA_SETUP='iid'

python fednoisy/algorithms/centr/main.py \
--dataset $DATASET \
--noise_mode $NOISE_MODE \
--noise_ratio $NOISE_RATIO \
--raw_data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--model $MODEL \
--num_epochs $NUM_EPOCHS \
--batch_size 320 \
--lr $LR \
--seed $SEED \
--momentum $MOMENTUM \
--use_wandb \
--weight_decay $WEIGHT_DECAY \
--wandb_project_name $WD_P_NAME \
--wandb_run_name $DATASET-central-sym-eq-$N_SETUP-bs320 \
--wandb_tags $DATASET sym eq$N_SETUP lr-$LR e$NUM_EPOCHS bs320 sgd central for$FL_DATA_SETUP \
--wandb_job_type $WD_J_TYPE >> /dev/null 2>&1 &

python fednoisy/algorithms/centr/main.py \
--dataset $DATASET \
--noise_mode $NOISE_MODE \
--noise_ratio $NOISE_RATIO \
--raw_data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--model $MODEL \
--num_epochs $NUM_EPOCHS \
--batch_size 160 \
--lr $LR \
--seed $SEED \
--momentum $MOMENTUM \
--use_wandb \
--weight_decay $WEIGHT_DECAY \
--wandb_project_name $WD_P_NAME \
--wandb_run_name $DATASET-central-sym-eq-$N_SETUP-bs160 \
--wandb_tags $DATASET sym eq$N_SETUP lr-$LR e$NUM_EPOCHS bs160 sgd central for$FL_DATA_SETUP \
--wandb_job_type $WD_J_TYPE >> /dev/null 2>&1 &

python fednoisy/algorithms/centr/main.py \
--dataset $DATASET \
--noise_mode $NOISE_MODE \
--noise_ratio $NOISE_RATIO \
--raw_data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--model $MODEL \
--num_epochs $NUM_EPOCHS \
--batch_size 64 \
--lr $LR \
--seed $SEED \
--momentum $MOMENTUM \
--use_wandb \
--weight_decay $WEIGHT_DECAY \
--wandb_project_name $WD_P_NAME \
--wandb_run_name $DATASET-central-sym-eq-$N_SETUP-bs64 \
--wandb_tags $DATASET sym eq$N_SETUP lr-$LR e$NUM_EPOCHS bs64 sgd central for$FL_DATA_SETUP \
--wandb_job_type $WD_J_TYPE >> /dev/null 2>&1 &

wait
echo " Done all jobs"
exit 0