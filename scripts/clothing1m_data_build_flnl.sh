#!/bin/bash
python build_dataset_fed.py --dataset clothing1m \
    --partition iid \
    --num_clients 10 \
    --globalize \
    --noise_mode real \
    --raw_data_dir ../rawdata/clothing1M/ \
    --data_dir ../flnldata/clothing1m \
    --seed 1 \
    --num_samples 265664

python build_dataset_fed.py --dataset clothing1m \
    --partition noniid-labeldir \
    --dir_alpha 0.1 \
    --num_clients 10 \
    --globalize \
    --noise_mode real \
    --raw_data_dir ../rawdata/clothing1M/ \
    --data_dir ../flnldata/clothing1m \
    --seed 1 \
    --num_samples 265664
