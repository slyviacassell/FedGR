import torch
import argparse
import sys
from time import sleep

from json import load
import os
import argparse
import random
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp

from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.core.network import DistNetwork
from fedlab.core.client import PassiveClientManager
import wandb

sys.path.append(os.getcwd())
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)
from fednoisy.data.NLLData import functional as nllF
from fednoisy.algorithms.nlfl.misc import read_fednll_args
from fednoisy.data.dataset import FedNLLDataset
from fednoisy.utils.misc import (
    setup_seed,
    make_dirs,
    make_exp_name,
    result_parser,
    make_alg_name,
    now,
)
from fednoisy.models.nlfl_models.build_model import build_model
from fednoisy.utils.wandb_logger import WandbLogger
from fednoisy.algorithms.nlfl.standalone import NLLFedAvgStandalone
from fednoisy.algorithms.nlfl.server import NLLFedAvgServerHandler
from fednoisy.algorithms.nlfl.client import NLLFedAvgClientTrainer

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
# exp_name = make_exp_name("fedavg", args)
exp_name = args.exp_name
alg_name = "FedAvg"
time_stamp=now()
cmp_out_dir = os.path.join(args.out_dir, nll_name, alg_name, exp_name,time_stamp)
args.time_stamp = time_stamp
make_dirs(cmp_out_dir)

backbone, head = build_model(args.model, CLASS_NUM[args.dataset], dataset=args.dataset)

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
        tags=args.wandb_tags,
        job_type=args.wandb_job_type,
        name=args.time_stamp + '-' + args.wandb_run_name,
    )
else:
    wandb_logger = None

# ==== choose server handler and client trainer ====
handler = NLLFedAvgServerHandler(
    head, args.com_round, args.sample_ratio, logger=server_logger, wandb_logger=wandb_logger, args=args
)  # server


# ---- FedAvg & FedAvg-RobustLoss ----
trainer = NLLFedAvgClientTrainer(
    backbone, head, args.num_clients, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args
)

# ==== server dataset ====
handler_dataset = FedNLLDataset(args, test_preload=args.preload, train_preload=args.preload)
handler.setup_dataset(handler_dataset)

# ==== client trainer dataset ====
trainer_dataset = FedNLLDataset(
    args, train_preload=args.preload, test_preload=args.preload
)
trainer.setup_dataset(trainer_dataset)
trainer.init_optim(
    args.epochs, args.batch_size, args.lr, args.weight_decay, args.momentum
)

# ====  launch pipeline ====
print(f"FedNLL scene: {nll_name}")
pipeline = NLLFedAvgStandalone(handler, trainer, args=args, save_best=args.save_best, out_path=cmp_out_dir, wandb_logger=wandb_logger)

pipeline.main()