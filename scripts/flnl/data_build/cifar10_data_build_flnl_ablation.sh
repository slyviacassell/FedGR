#!/bin/bash
# SEED=1
RAW_DATA_DIR='../rawdata/cifar10'
DATA_DIR='../flnldata/cifar10'
DATASET='cifar10'
# echo "python build_dataset_fed.py --dataset $DATASET --partition iid --num_clients 10 --globalize --noise_mode clean --seed $SEED --raw_data_dir $RAW_DATA_DIR --data_dir $DATA_DIR"

python build_dataset_fed.py --dataset $DATASET --partition iid --num_clients 10 --noise_mode sym --noisy_client_ratio 0.6 --max_noise_ratio 1.00 --min_noise_ratio 0.50 --seed $SEED --raw_data_dir $RAW_DATA_DIR --data_dir $DATA_DIR

python build_dataset_fed.py --dataset $DATASET --partition iid --num_clients 10 --noise_mode asym --noisy_client_ratio 0.6 --max_noise_ratio 0.40 --min_noise_ratio 0.20 --seed $SEED --raw_data_dir $RAW_DATA_DIR --data_dir $DATA_DIR

python build_dataset_fed.py --dataset $DATASET --partition iid --num_clients 10 --noise_mode mixed --noisy_client_ratio 0.6 --max_noise_ratio 0.40 --min_noise_ratio 0.20 --seed $SEED --raw_data_dir $RAW_DATA_DIR --data_dir $DATA_DIR


python build_dataset_fed.py --dataset $DATASET --partition noniid-labeldir --num_clients 10 --dir_alpha 0.3 --noise_mode sym --noisy_client_ratio 0.6 --max_noise_ratio 1.00 --min_noise_ratio 0.50 --seed $SEED --raw_data_dir $RAW_DATA_DIR --data_dir $DATA_DIR

python build_dataset_fed.py --dataset $DATASET --partition noniid-labeldir --num_clients 10 --dir_alpha 0.3 --noise_mode asym --noisy_client_ratio 0.6 --max_noise_ratio 0.40 --min_noise_ratio 0.20 --seed $SEED --raw_data_dir $RAW_DATA_DIR --data_dir $DATA_DIR

python build_dataset_fed.py --dataset $DATASET --partition noniid-labeldir --num_clients 10 --dir_alpha 0.3 --noise_mode mixed --noisy_client_ratio 0.6 --max_noise_ratio 0.40 --min_noise_ratio 0.20 --seed $SEED --raw_data_dir $RAW_DATA_DIR --data_dir $DATA_DIR