#!/bin/bash

MakeDir(){
    if ! [ -e "$1" ]
    then
        mkdir -p "$1"
    fi
}

cd /yuxintian-20phd/projects/fednoisy/FedNoisy

# SEED=1
DATA_DIR='../flnldata/clothing1m'
DATASET='clothing1m'
N_CLIENTS=500
OUT_DIR='../flnl-fednll/clothing1m/'
EPOCHS=10
LR=0.003
MOMENTUM=0.9
WEIGHT_DECAY=0.0005
BSZ=32
COM_ROUND=200
SR=0.02
PARTITION='iid'
MODEL='ResNet50'
EXP_NAME='fednll'
WD_P_NAME=FLNL-debug
WD_J_TYPE='explore'
WD_MODE='online'

CS_METRIC='loss_mean'
SSL_METHOD='fedprox_like'
SSL2SL_REG_WEIGHT=0.0
SSL_WEIGHT=0.2
ORCHESTRA_TEMPERATURE=0.5
WARUM_ROUND=0
METRIC_MODEL='global'
N_SYS_SNIFFING_PER_CLIENT=1

CUR_DATE=$(date +%Y%m%d%H)
LOG_DIR=${OUT_DIR}/logs/${CUR_DATE}
MakeDir "${LOG_DIR}"


# clothing1m-c${N_CLIENTS}-sr${SR}-iid-real-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r50-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/flnl/standalone/main.py \
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
--wandb_group clothing1m-c${N_CLIENTS}-sr${SR}-iid-real-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r50-s${SEED} \
--wandb_tags ${DATASET} clients-${N_CLIENTS} real lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed$SEED \
--wandb_job_type $WD_J_TYPE \
--num_samples 1000000 \
--use_fednll \
--n_sys_sniffing_per_client $N_SYS_SNIFFING_PER_CLIENT \
--cs_metric $CS_METRIC \
--metric_model $METRIC_MODEL \
--warmup_round $WARUM_ROUND \
--ssl_method $SSL_METHOD \
--ssl2sl_reg_weight $SSL2SL_REG_WEIGHT \
--ssl_weight $SSL_WEIGHT \
--orchestra_temperature $ORCHESTRA_TEMPERATURE \
--local_ema_plus_global_decay 0.5 \
--local_ema_plus_global \
--pse_method fixmatch \
--local_ema \
--fixmatch_threshold 0.9 \
--use_strong_aug \
--local_ema_reinit \
--pse_size_threshold 0.5 \
--gmm_selection inter \
--anchor_model global \
--upper_rate_threshold 0.8 \
--num_workers 8 \
--soft_silent_round 0 \
>> "${LOG_DIR}/clothing1m-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-iid-real-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r50-s$SEED.log" 2>&1 &

PARTITION='noniid-labeldir'
DIR_ALPHA=0.3

wait
echo "All runs are done!"