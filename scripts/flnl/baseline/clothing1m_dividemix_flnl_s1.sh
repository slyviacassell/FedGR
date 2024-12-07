#!/bin/bash

MakeDir(){
    if ! [ -e "$1" ]
    then
        mkdir -p "$1"
    fi
}

cd /yuxintian-20phd/projects/fednoisy/FedNoisy

SEED=1
DATA_DIR='../flnldata/clothing1m'
DATASET='clothing1m'
N_CLIENTS=500
OUT_DIR='../flnl-baseline/clothing1m/'
EPOCHS=10
LR=0.003
MOMENTUM=0.5
WEIGHT_DECAY=0.0005
BSZ=32
COM_ROUND=200
SR=0.02
PARTITION='iid'
MODEL='ResNet50'
EXP_NAME='fedavg-dividemix'
WD_P_NAME=FLNL
WD_J_TYPE='baseline'
WD_MODE='offline'

LOG_DIR=${OUT_DIR}/logs/
MakeDir "${LOG_DIR}"

# clothing1m-c${N_CLIENTS}-sr${SR}-iid-real-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r50-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/fedavg/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--num_clients $N_CLIENTS \
--noise_mode real \
--data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--com_round $COM_ROUND \
--epochs $EPOCHS \
--sample_ratio $SR \
--lr $LR \
--momentum $MOMENTUM \
--weight_decay $WEIGHT_DECAY \
--seed $SEED \
--preload \
--batch_size $BSZ \
--exp_name $EXP_NAME \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_group clothing1m-c${N_CLIENTS}-sr${SR}-iid-real-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r50-s$SEED \
--wandb_tags ${DATASET} clients-${N_CLIENTS} real lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed$SEED \
--wandb_job_type $WD_J_TYPE \
--num_samples 1000000 \
--dividemix \
--dividemix_warmup_round 100 \
--dividemix_lambda_u 25 \
>> "${LOG_DIR}/clothing1m-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-iid-real-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r50-s${SEED}.log" 2>&1 &

PARTITION='noniid-labeldir'
DIR_ALPHA=0.3

# clothing1m-c${N_CLIENTS}-sr${SR}-dir03-real-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r50-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/fedavg/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--dir_alpha $DIR_ALPHA \
--num_clients $N_CLIENTS \
--noise_mode real \
--data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--com_round $COM_ROUND \
--epochs $EPOCHS \
--sample_ratio $SR \
--lr $LR \
--momentum $MOMENTUM \
--weight_decay $WEIGHT_DECAY \
--seed $SEED \
--preload \
--batch_size $BSZ \
--exp_name $EXP_NAME \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_group clothing1m-c${N_CLIENTS}-sr${SR}-dir03-real-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r50-s$SEED \
--wandb_tags ${DATASET} clients-${N_CLIENTS} real lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed$SEED \
--wandb_job_type $WD_J_TYPE \
--num_samples 1000000 \
--dividemix \
--dividemix_warmup_round 100 \
--dividemix_lambda_u 25 \
>> "${LOG_DIR}/clothing1m-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-dir03-real-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r50-s${SEED}.log" 2>&1 &

wait
echo "All runs are done!"
exit 0