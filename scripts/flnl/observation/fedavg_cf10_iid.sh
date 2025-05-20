#!/bin/bash
# NOISE_MODE='mixed'
# WD_P_NAME='NLFL'
DATA_DIR='../flnldata/cifar10'
OUT_DIR='../nlfl-ob-sym/cifar10/'
N_SETUP='n06ls24'

python fednoisy/algorithms/nlfl/main.py \
--dataset cifar10 \
--model ResNet18 \
--partition iid \
--num_clients 10 \
--min_noise_ratio 0.2 \
--max_noise_ratio 0.4 \
--noise_mode $NOISE_MODE \
--data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--com_round 200 \
--batch_size 32 \
--epochs 10 \
--sample_ratio 1.0 \
--lr 0.01 \
--momentum 0.5 \
--weight_decay 0.0005 \
--seed 1 \
--preload \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_run_name cf10-c10-iid-$N_SETUP \
--wandb_tags cifar10 $N_SETUP lr-0.01 l10r200 bs32 sgd iid c10 fedavg \
--wandb_job_type baseline \
--noisy_client_ratio 0.6 >> /dev/null 2>&1 &

sleep 10m

python fednoisy/algorithms/nlfl/main.py \
--dataset cifar10 \
--model ResNet18 \
--partition iid \
--num_clients 10 \
--min_noise_ratio 0.2 \
--max_noise_ratio 0.4 \
--noise_mode $NOISE_MODE \
--data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--com_round 200 \
--batch_size 32 \
--epochs 10 \
--sample_ratio 0.5 \
--lr 0.01 \
--momentum 0.5 \
--weight_decay 0.0005 \
--seed 1 \
--preload \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_run_name cf10-c10-iid-$N_SETUP \
--wandb_tags cifar10 $N_SETUP lr-0.01 l10r200 bs32 sgd iid c10 fedavg \
--wandb_job_type baseline \
--noisy_client_ratio 0.6 >> /dev/null 2>&1 &

sleep 10m

python fednoisy/algorithms/nlfl/main.py \
--dataset cifar10 \
--model ResNet18 \
--partition iid \
--num_clients 10 \
--min_noise_ratio 0.2 \
--max_noise_ratio 0.4 \
--noise_mode $NOISE_MODE \
--data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--com_round 200 \
--batch_size 32 \
--epochs 10 \
--sample_ratio 0.2 \
--lr 0.01 \
--momentum 0.5 \
--weight_decay 0.0005 \
--seed 1 \
--preload \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_run_name cf10-c10-iid-$N_SETUP \
--wandb_tags cifar10 $N_SETUP lr-0.01 l10r200 bs32 sgd iid c10 fedavg \
--wandb_job_type baseline \
--noisy_client_ratio 0.6 >> /dev/null 2>&1 &

sleep 10m

python fednoisy/algorithms/nlfl/main.py \
--dataset cifar10 \
--model ResNet18 \
--partition iid \
--num_clients 10 \
--min_noise_ratio 0.2 \
--max_noise_ratio 0.4 \
--noise_mode $NOISE_MODE \
--data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--com_round 200 \
--batch_size 32 \
--epochs 1 \
--sample_ratio 1.0 \
--lr 0.01 \
--momentum 0.5 \
--weight_decay 0.0005 \
--seed 1 \
--preload \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_run_name cf10-c10-iid-$N_SETUP \
--wandb_tags cifar10 $N_SETUP lr-0.01 l10r200 bs32 sgd iid c10 fedavg \
--wandb_job_type baseline \
--noisy_client_ratio 0.6 >> /dev/null 2>&1 &

sleep 10m

python fednoisy/algorithms/nlfl/main.py \
--dataset cifar10 \
--model ResNet18 \
--partition iid \
--num_clients 10 \
--min_noise_ratio 0.2 \
--max_noise_ratio 0.4 \
--noise_mode $NOISE_MODE \
--data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--com_round 200 \
--batch_size 32 \
--epochs 1 \
--sample_ratio 0.5 \
--lr 0.01 \
--momentum 0.5 \
--weight_decay 0.0005 \
--seed 1 \
--preload \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_run_name cf10-c10-iid-$N_SETUP \
--wandb_tags cifar10 $N_SETUP lr-0.01 l10r200 bs32 sgd iid c10 fedavg \
--wandb_job_type baseline \
--noisy_client_ratio 0.6 >> /dev/null 2>&1 &

sleep 10m

python fednoisy/algorithms/nlfl/main.py \
--dataset cifar10 \
--model ResNet18 \
--partition iid \
--num_clients 10 \
--min_noise_ratio 0.2 \
--max_noise_ratio 0.4 \
--noise_mode $NOISE_MODE \
--data_dir $DATA_DIR \
--out_dir $OUT_DIR \
--com_round 200 \
--batch_size 32 \
--epochs 1 \
--sample_ratio 0.2 \
--lr 0.01 \
--momentum 0.5 \
--weight_decay 0.0005 \
--seed 1 \
--preload \
--use_wandb \
--wandb_project_name $WD_P_NAME \
--wandb_run_name cf10-c10-iid-$N_SETUP \
--wandb_tags cifar10 $N_SETUP lr-0.01 l10r200 bs32 sgd iid c10 fedavg \
--wandb_job_type baseline \
--noisy_client_ratio 0.6 >> /dev/null 2>&1 &

sleep 10m

wait
echo " Done all jobs"
exit 0