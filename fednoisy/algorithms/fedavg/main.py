from json import load
import os
import sys
import argparse
import random
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators

sys.path.append(os.getcwd())
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)
from fednoisy.data.NLLData import functional as nllF
from fednoisy.algorithms.fedavg.client import (
    FedNLLFedAvgClientTrainer,
    FedNLLFedAvgMixupClientTrainer,
    FedNLLFedAvgCoteachingClientTrainer,
    FedNLLFedAvgDynamicBootstrappingClientTrainer,
    FedNLLFedAvgDivideMixClientTrainer,
)
from fednoisy.algorithms.fedavg.server import FedAvgServerHandler

from fednoisy.algorithms.fedavg.standalone import FedAvgStandalone
from fednoisy.algorithms.fedavg.misc import read_fednll_args
from fednoisy.data.dataset import FedNLLDataset
from fednoisy.utils.misc import (
    setup_seed,
    make_dirs,
    make_exp_name,
    result_parser,
    make_alg_name,
    now,
)
from fednoisy.models.build_model import build_model, build_multi_model
from fednoisy.utils.wandb_logger import WandbLogger

args = read_fednll_args()
if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

setup_seed(args.seed)
if args.dataset == "clothing1m":
    args.noise_mode = "real"
    args.globalize = True
    args.noise_ratio = 0.39

nll_name = nllF.FedNLL_name(**vars(args))
exp_name = args.exp_name
alg_name = "FedAvg-standalone"
# exp_name = make_exp_name("fedavg", args)
# alg_name = make_alg_name(args)
time_stamp=now()
cmp_out_dir = os.path.join(args.out_dir, nll_name, alg_name, exp_name,time_stamp)
args.time_stamp = time_stamp
make_dirs(cmp_out_dir)

if args.coteaching is True or args.dividemix is True:
    model = build_multi_model(
        args.model, CLASS_NUM[args.dataset], dataset=args.dataset, num_models=2
    )
else:
    model = build_model(args.model, CLASS_NUM[args.dataset], dataset=args.dataset)

# ==== prepare logger ====
server_logger = Logger(
    log_name="ServerHandler",
    log_file=os.path.join(cmp_out_dir, "server.log"),
)

client_logger = Logger(
    log_name="ClientTrainer",
    log_file=os.path.join(cmp_out_dir, "client.log"),
)

if args.use_wandb:
    wandb_logger = WandbLogger(
            args.wandb_project_name, 
            exp_cfg=vars(args), 
            group=args.time_stamp+"-"+args.wandb_group,
            tags=args.wandb_tags,
            job_type=args.wandb_job_type,
            name=exp_name + "-standalone",
        )
else:
    wandb_logger = None

# ==== choose server handler and client trainer ====
handler = FedAvgServerHandler(
    model, args.com_round, args.sample_ratio, logger=server_logger, wandb_logger=wandb_logger, args=args
)  # server

if args.mixup is True:
    # ---- FedAvg-Mixup ----
    trainer = FedNLLFedAvgMixupClientTrainer(
        model, args.num_clients, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args
    )  # client
elif args.coteaching is True:
    # ---- FedAvg-Coteaching ----
    trainer = FedNLLFedAvgCoteachingClientTrainer(
        model, args.num_clients, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args
    )  # client
elif args.dividemix is True:
    # ---- FedAvg-DivideMix ----
    trainer = FedNLLFedAvgDivideMixClientTrainer(
        model, args.num_clients, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args
    )  # client
elif args.dynboot is True:
    trainer = FedNLLFedAvgDynamicBootstrappingClientTrainer(
        model, args.num_clients, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args
    )
else:
    # ---- FedAvg & FedAvg-RobustLoss ----
    trainer = FedNLLFedAvgClientTrainer(
        model, args.num_clients, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args
    )  # client

# ==== server dataset ====
handler_dataset = FedNLLDataset(args, test_preload=args.preload)
handler.setup_dataset(handler_dataset)

# ==== client trainer dataset ====
trainer_dataset = FedNLLDataset(
    args, train_preload=args.preload, test_preload=args.preload
)
trainer.setup_dataset(trainer_dataset)
trainer.setup_optim(
    args.epochs, args.batch_size, args.lr, args.weight_decay, args.momentum
)

# ====  launch pipeline ====
print(f"FedNLL scene: {nll_name}")
pipeline = FedAvgStandalone(handler, trainer, args=args, save_best=args.save_best, out_path=cmp_out_dir, wandb_logger=wandb_logger)

pipeline.main()
