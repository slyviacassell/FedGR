# Command
- Dataset Download
```bash
python scripts/torchvision_dataset_download.py
```

- Dataset
```bash
bash scripts/cifar10_data_build_flnl.sh
```

- FedAvg with multiple GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 python fednoisy/algorithms/flnl/scale/main.py --dataset cifar10 --model ResNet18 --partition iid --num_clients 10 --min_noise_ratio 0.40 --noise_mode sym --max_noise_ratio 0.70 --data_dir ../flnldata/cifar10 --out_dir ../Fed-Noisy-checkpoint-hold/cifar10/ --com_round 200 --epochs 5 --sample_ratio 1.0 --lr 0.01 --momentum 0.9 --weight_decay 0.0005 --seed 1 --preload --use_cs --num_clients_per_gpu 5 --world_size 3 --metric_model local --cs_metric loss_mean
```

- FedAvg with single GPUs
```bash
CUDA_VISIBLE_DEVICES=0 python fednoisy/algorithms/flnl/standalone/main.py --dataset cifar10 --model ResNet18 --partition iid --num_clients 10 --min_noise_ratio 0.40 --noise_mode sym --max_noise_ratio 0.70 --data_dir ../flnldata/cifar10 --out_dir ../Fed-Noisy-checkpoint-hold/cifar10/ --com_round 200 --epochs 5 --sample_ratio 1.0 --lr 0.01 --momentum 0.9 --weight_decay 0.0005 --seed 1 --preload --use_cs --metric_model local --cs_metric loss_mean
```