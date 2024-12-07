#!/bin/bash

MakeDir(){
    if ! [ -e "$1" ]
    then
        mkdir -p "$1"
    fi
}

cd /yuxintian-20phd/projects/fednoisy/FedNoisy

# SEED=1
DATA_DIR='../flnldata/cifar10'
DATASET='cifar10'
N_CLIENTS=100
OUT_DIR='../flnl-fednll/cifar10/'
EPOCHS=10
LR=0.01
MOMENTUM=0.5
WEIGHT_DECAY=0.0005
BSZ=32
COM_ROUND=500
SR=0.1
PARTITION='noniid-labeldir'
DIR_ALPHA=0.3
MODEL='ResNet18'
EXP_NAME='fednll'
WD_P_NAME=FLNL-debug
WD_J_TYPE='explore'
WD_MODE='online'

CS_METRIC='loss_mean'
SSL_METHOD='fedprox_like'
SSL2SL_REG_WEIGHT=0.0
SSL_WEIGHT=0.1
ORCHESTRA_TEMPERATURE=0.5
WARUM_ROUND=0
METRIC_MODEL='global'
N_SYS_SNIFFING_PER_CLIENT=10

LOCAL_EMA_PLUS_GLOBAL_DECAY=0.9
FIXMATCH_THRESHOLD=0.9
PSE_SIZE_THRESHOLD=0.5
GMM_SELECTION='inter'
ANCHOR_MODEL='global'
UPPER_RATE_THRESHOLD=0.8
SOFT_SILENT_ROUND=100

CUR_DATE=$(date +%Y%m%d%H)
LOG_DIR=${OUT_DIR}/logs/${CUR_DATE}
MakeDir "${LOG_DIR}"

# cf10-c${N_CLIENTS}-sr${SR}-dir03-clean-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/flnl/standalone/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--dir_alpha $DIR_ALPHA \
--num_clients $N_CLIENTS \
--noise_mode clean \
--globalize \
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
--use_fednll \
--exp_name $EXP_NAME \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-dir03-clean-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r18-s${SEED} \
--wandb_tags ${DATASET} clients-${N_CLIENTS} clean lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed$SEED \
--wandb_job_type $WD_J_TYPE \
--n_sys_sniffing_per_client $N_SYS_SNIFFING_PER_CLIENT \
--cs_metric $CS_METRIC \
--metric_model $METRIC_MODEL \
--warmup_round $WARUM_ROUND \
--ssl_method $SSL_METHOD \
--ssl2sl_reg_weight $SSL2SL_REG_WEIGHT \
--ssl_weight $SSL_WEIGHT \
--orchestra_temperature $ORCHESTRA_TEMPERATURE \
--local_ema_plus_global_decay $LOCAL_EMA_PLUS_GLOBAL_DECAY \
--local_ema_plus_global \
--pse_method fixmatch \
--use_strong_aug \
--local_ema \
--local_ema_plus_global \
--fixmatch_threshold $FIXMATCH_THRESHOLD \
--local_ema_reinit \
--pse_size_threshold $PSE_SIZE_THRESHOLD \
--gmm_selection $GMM_SELECTION \
--anchor_model $ANCHOR_MODEL \
--upper_rate_threshold $UPPER_RATE_THRESHOLD \
--soft_silent_round $SOFT_SILENT_ROUND \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-dir03-clean-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r18-s${SEED}.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-dir03-n06ls510-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/flnl/standalone/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--dir_alpha $DIR_ALPHA \
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
--use_fednll \
--exp_name $EXP_NAME \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-dir03-n06ls510-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r18-s${SEED} \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n06ls510 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed$SEED \
--wandb_job_type $WD_J_TYPE \
--n_sys_sniffing_per_client $N_SYS_SNIFFING_PER_CLIENT \
--cs_metric $CS_METRIC \
--metric_model $METRIC_MODEL \
--warmup_round $WARUM_ROUND \
--ssl_method $SSL_METHOD \
--ssl2sl_reg_weight $SSL2SL_REG_WEIGHT \
--ssl_weight $SSL_WEIGHT \
--orchestra_temperature $ORCHESTRA_TEMPERATURE \
--local_ema_plus_global_decay $LOCAL_EMA_PLUS_GLOBAL_DECAY \
--local_ema_plus_global \
--pse_method fixmatch \
--use_strong_aug \
--local_ema \
--local_ema_plus_global \
--fixmatch_threshold $FIXMATCH_THRESHOLD \
--local_ema_reinit \
--pse_size_threshold $PSE_SIZE_THRESHOLD \
--gmm_selection $GMM_SELECTION \
--anchor_model $ANCHOR_MODEL \
--upper_rate_threshold $UPPER_RATE_THRESHOLD \
--soft_silent_round $SOFT_SILENT_ROUND \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-dir03-n06ls510-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-dir03-n10ls510-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/flnl/standalone/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--dir_alpha $DIR_ALPHA \
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
--use_fednll \
--exp_name $EXP_NAME \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-dir03-n10ls510-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r18-s${SEED} \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n10ls510 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed$SEED \
--wandb_job_type $WD_J_TYPE \
--n_sys_sniffing_per_client $N_SYS_SNIFFING_PER_CLIENT \
--cs_metric $CS_METRIC \
--metric_model $METRIC_MODEL \
--warmup_round $WARUM_ROUND \
--ssl_method $SSL_METHOD \
--ssl2sl_reg_weight $SSL2SL_REG_WEIGHT \
--ssl_weight $SSL_WEIGHT \
--orchestra_temperature $ORCHESTRA_TEMPERATURE \
--local_ema_plus_global_decay $LOCAL_EMA_PLUS_GLOBAL_DECAY \
--local_ema_plus_global \
--pse_method fixmatch \
--use_strong_aug \
--local_ema \
--local_ema_plus_global \
--fixmatch_threshold $FIXMATCH_THRESHOLD \
--local_ema_reinit \
--pse_size_threshold $PSE_SIZE_THRESHOLD \
--gmm_selection $GMM_SELECTION \
--anchor_model $ANCHOR_MODEL \
--upper_rate_threshold $UPPER_RATE_THRESHOLD \
--soft_silent_round $SOFT_SILENT_ROUND \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-dir03-n10ls510-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-dir03-n06las24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/flnl/standalone/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--dir_alpha $DIR_ALPHA \
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
--use_fednll \
--exp_name $EXP_NAME \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-dir03-n06las24-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r18-s${SEED} \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n06las24 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed$SEED \
--wandb_job_type $WD_J_TYPE \
--n_sys_sniffing_per_client $N_SYS_SNIFFING_PER_CLIENT \
--cs_metric $CS_METRIC \
--metric_model $METRIC_MODEL \
--warmup_round $WARUM_ROUND \
--ssl_method $SSL_METHOD \
--ssl2sl_reg_weight $SSL2SL_REG_WEIGHT \
--ssl_weight $SSL_WEIGHT \
--orchestra_temperature $ORCHESTRA_TEMPERATURE \
--local_ema_plus_global_decay $LOCAL_EMA_PLUS_GLOBAL_DECAY \
--local_ema_plus_global \
--pse_method fixmatch \
--use_strong_aug \
--local_ema \
--local_ema_plus_global \
--fixmatch_threshold $FIXMATCH_THRESHOLD \
--local_ema_reinit \
--pse_size_threshold $PSE_SIZE_THRESHOLD \
--gmm_selection $GMM_SELECTION \
--anchor_model $ANCHOR_MODEL \
--upper_rate_threshold $UPPER_RATE_THRESHOLD \
--soft_silent_round $SOFT_SILENT_ROUND \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-dir03-n06las24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-dir03-n10las24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/flnl/standalone/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--dir_alpha $DIR_ALPHA \
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
--use_fednll \
--exp_name $EXP_NAME \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-dir03-n10las24-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r18-s${SEED} \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n10las24 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed$SEED \
--wandb_job_type $WD_J_TYPE \
--n_sys_sniffing_per_client $N_SYS_SNIFFING_PER_CLIENT \
--cs_metric $CS_METRIC \
--metric_model $METRIC_MODEL \
--warmup_round $WARUM_ROUND \
--ssl_method $SSL_METHOD \
--ssl2sl_reg_weight $SSL2SL_REG_WEIGHT \
--ssl_weight $SSL_WEIGHT \
--orchestra_temperature $ORCHESTRA_TEMPERATURE \
--local_ema_plus_global_decay $LOCAL_EMA_PLUS_GLOBAL_DECAY \
--local_ema_plus_global \
--pse_method fixmatch \
--use_strong_aug \
--local_ema \
--local_ema_plus_global \
--fixmatch_threshold $FIXMATCH_THRESHOLD \
--local_ema_reinit \
--pse_size_threshold $PSE_SIZE_THRESHOLD \
--gmm_selection $GMM_SELECTION \
--anchor_model $ANCHOR_MODEL \
--upper_rate_threshold $UPPER_RATE_THRESHOLD \
--soft_silent_round $SOFT_SILENT_ROUND \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-dir03-n10las24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-dir03-n06lm24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/flnl/standalone/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--dir_alpha $DIR_ALPHA \
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
--use_fednll \
--exp_name $EXP_NAME \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-dir03-n06lm24-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r18-s${SEED} \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n06lm24 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed$SEED \
--wandb_job_type $WD_J_TYPE \
--n_sys_sniffing_per_client $N_SYS_SNIFFING_PER_CLIENT \
--cs_metric $CS_METRIC \
--metric_model $METRIC_MODEL \
--warmup_round $WARUM_ROUND \
--ssl_method $SSL_METHOD \
--ssl2sl_reg_weight $SSL2SL_REG_WEIGHT \
--ssl_weight $SSL_WEIGHT \
--orchestra_temperature $ORCHESTRA_TEMPERATURE \
--local_ema_plus_global_decay $LOCAL_EMA_PLUS_GLOBAL_DECAY \
--local_ema_plus_global \
--pse_method fixmatch \
--use_strong_aug \
--local_ema \
--local_ema_plus_global \
--fixmatch_threshold $FIXMATCH_THRESHOLD \
--local_ema_reinit \
--pse_size_threshold $PSE_SIZE_THRESHOLD \
--gmm_selection $GMM_SELECTION \
--anchor_model $ANCHOR_MODEL \
--upper_rate_threshold $UPPER_RATE_THRESHOLD \
--soft_silent_round $SOFT_SILENT_ROUND \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-dir03-n06lm24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

# cf10-c${N_CLIENTS}-sr${SR}-dir03-n10lm24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED
WANDB_MODE=$WD_MODE CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/flnl/standalone/main.py \
--dataset $DATASET \
--model $MODEL \
--partition $PARTITION \
--dir_alpha $DIR_ALPHA \
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
--use_fednll \
--exp_name $EXP_NAME \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_group cf10-c${N_CLIENTS}-sr${SR}-dir03-n10lm24-lr${LR}-l${EPOCHS}r${COM_ROUND}-bs${BSZ}-sgd-r18-s${SEED} \
--wandb_tags ${DATASET} clients-${N_CLIENTS} n10lm24 lr-${LR} wd${WEIGHT_DECAY} momentum${MOMENTUM} l${EPOCHS}r${COM_ROUND} bs${BSZ} sgd ${PARTITION} ${MODEL} sr${SR} seed$SEED \
--wandb_job_type $WD_J_TYPE \
--n_sys_sniffing_per_client $N_SYS_SNIFFING_PER_CLIENT \
--cs_metric $CS_METRIC \
--metric_model $METRIC_MODEL \
--warmup_round $WARUM_ROUND \
--ssl_method $SSL_METHOD \
--ssl2sl_reg_weight $SSL2SL_REG_WEIGHT \
--ssl_weight $SSL_WEIGHT \
--orchestra_temperature $ORCHESTRA_TEMPERATURE \
--local_ema_plus_global_decay $LOCAL_EMA_PLUS_GLOBAL_DECAY \
--local_ema_plus_global \
--pse_method fixmatch \
--use_strong_aug \
--local_ema \
--local_ema_plus_global \
--fixmatch_threshold $FIXMATCH_THRESHOLD \
--local_ema_reinit \
--pse_size_threshold $PSE_SIZE_THRESHOLD \
--gmm_selection $GMM_SELECTION \
--anchor_model $ANCHOR_MODEL \
--upper_rate_threshold $UPPER_RATE_THRESHOLD \
--soft_silent_round $SOFT_SILENT_ROUND \
>> "${LOG_DIR}/cf10-${EXP_NAME}-c${N_CLIENTS}-sr${SR}-dir03-n10lm24-lr$LR-l$EPOCHSr$COM_ROUND-bs$BSZ-sgd-r18-s$SEED.log" 2>&1 &

wait
echo "All runs are done!"