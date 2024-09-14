#!/bin/bash
python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode clean --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist

python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode sym --noise_ratio 0.4 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode sym --noise_ratio 0.5 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode sym --noise_ratio 0.6 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode sym --noise_ratio 0.7 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist

python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --noise_mode sym --max_noise_ratio 0.40 --min_noise_ratio 0.00 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --noise_mode sym --max_noise_ratio 0.70 --min_noise_ratio 0.00 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist

python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode asym --noise_ratio 0.2 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode asym --noise_ratio 0.3 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode asym --noise_ratio 0.4 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist

python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --noise_mode asym --max_noise_ratio 0.20 --min_noise_ratio 0.0 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition iid --num_clients 10 --noise_mode asym --max_noise_ratio 0.40 --min_noise_ratio 0.0 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist

python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode clean --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist

python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode sym --noise_ratio 0.4 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode sym --noise_ratio 0.5 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode sym --noise_ratio 0.6 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode sym --noise_ratio 0.7 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist

python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --noise_mode sym --max_noise_ratio 0.40 --min_noise_ratio 0.0 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --noise_mode sym --max_noise_ratio 0.70 --min_noise_ratio 0.0 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist

python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode asym --noise_ratio 0.2 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode asym --noise_ratio 0.3 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode asym --noise_ratio 0.4 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist

python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --noise_mode asym --max_noise_ratio 0.20 --min_noise_ratio 0.0 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist
python build_dataset_fed.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --noise_mode asym --max_noise_ratio 0.40 --min_noise_ratio 0.0 --seed 1 --raw_data_dir ../rawdata/mnist/MNIST --data_dir ../flnldata/mnist