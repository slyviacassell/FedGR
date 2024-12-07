#!/bin/bash

MakeDir(){
    if ! [ -e "$1" ]
    then
        mkdir "$1"
    fi
}

cd /yuxintian-20phd/projects/fednoisy/FedNoisy

SEED=1
DATA_DIR='../flnldata/cifar10'
DATASET='cifar10'
N_CLIENTS=100
OUT_DIR='../flnl-baseline/cifar10/'
EPOCHS=10
LR=0.01
MOMENTUM=0.5
WEIGHT_DECAY=0.0005
BSZ=32
COM_ROUND=500
SR=0.1
PARTITION='iid'
MODEL='ResNet18'
EXP_NAME='fedavg-cot'
WD_P_NAME=FLNL
WD_J_TYPE='baseline'
WD_MODE='offline'

LOG_DIR=${OUT_DIR}/logs/
MakeDir "${LOG_DIR}"

# # cf10-c${N_CLIENTS}-sr${SR}-iid-clean-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
# WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/fedavg/main.py \
# --dataset $DATASET \
# --model $MODEL \
# --partition $PARTITION \
# --num_clients $N_CLIENTS \
# --noise_mode clean \
# --globalize \
# --data_dir $DATA_DIR \
# --out_dir $OUT_DIR \
# --com_round $COM_ROUND \
# --epochs $EPOCHS \
# --sample_ratio $SR \
# --lr $LR \
# --momentum $MOMENTUM \
# --weight_decay $WEIGHT_DECAY \
# --seed $SEED \
# --preload \
# --batch_size $BSZ \
# --exp_name $EXP_NAME \
# --use_wandb \
# --wandb_project_name $WD_P_NAME \
# --wandb_group cf10-c${N_CLIENTS}-sr${SR}-iid-clean-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r18-s${SEED} \
# --wandb_tags ${DATASET} clients-${N_CLIENTS} clean lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed${SEED} \
#  --wandb_job_type $WD_J_TYPE

# cf10-c${N_CLIENTS}-sr${SR}-iid-n06ls510-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/fedavg/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--num_clients $N_CLIENTS \
--noisy_client_ratio 0.6 \
--min_noise_ratio 0.50 \
--max_noise_ratio 1.0 \
--noise_mode sym \
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
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-iid-n06ls510-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n06ls510 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed${SEED} \
--wandb_job_type $WD_J_TYPE \
--coteaching \
--coteaching_forget_rate 0.45 \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-iid-n06ls510-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-iid-n10ls510-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/fedavg/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--num_clients $N_CLIENTS \
--min_noise_ratio 0.50 \
--max_noise_ratio 1.0 \
--noise_mode sym \
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
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-iid-n10ls510-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n10ls510 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed${SEED} \
--wandb_job_type $WD_J_TYPE \
--coteaching \
--coteaching_forget_rate 0.75 \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-iid-n10ls510-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-iid-n06las24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/fedavg/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--num_clients $N_CLIENTS \
--noisy_client_ratio 0.6 \
--min_noise_ratio 0.20 \
--max_noise_ratio 0.40 \
--noise_mode asym \
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
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-iid-n06las24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n06las24 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed${SEED} \
--wandb_job_type $WD_J_TYPE \
--coteaching \
--coteaching_forget_rate 0.18 \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-iid-n06las24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-iid-n10las24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/fedavg/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--num_clients $N_CLIENTS \
--min_noise_ratio 0.20 \
--max_noise_ratio 0.40 \
--noise_mode asym \
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
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-iid-n10las24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n10las24 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed${SEED} \
--wandb_job_type $WD_J_TYPE \
--coteaching \
--coteaching_forget_rate 0.3 \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-iid-n10las24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-iid-n06lm24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/fedavg/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--num_clients $N_CLIENTS \
--noisy_client_ratio 0.6 \
--min_noise_ratio 0.20 \
--max_noise_ratio 0.40 \
--noise_mode mixed \
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
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-iid-n06lm24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n06lm24 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed${SEED} \
--wandb_job_type $WD_J_TYPE \
--coteaching \
--coteaching_forget_rate 0.18 \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-iid-n06lm24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-iid-n10lm24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/fedavg/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--num_clients $N_CLIENTS \
--min_noise_ratio 0.20 \
--max_noise_ratio 0.40 \
--noise_mode mixed \
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
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-iid-n10lm24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n10lm24 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed${SEED} \
--wandb_job_type $WD_J_TYPE \
--coteaching \
--coteaching_forget_rate 0.3 \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-iid-n10lm24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

wait
echo "All runs are done!"
exit 0
