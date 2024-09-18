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
import numpy as np

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

from fednoisy.algorithms.flnl.standalone.standalone import FedAPStandalone
from fednoisy.algorithms.flnl.standalone.fedap import (
    FedAPClientTrainer,
    FedAPServerHandler,
)
from fednoisy.algorithms.flnl.standalone.fedap_cs import (
    FedAPCSClientTrainer,
    FedAPCSServerHandler,
)
from fednoisy.algorithms.flnl.standalone.fedap_orchestra import (
    FedAPOrchestraClientTrainer,
    FedAPOrchestraServerHandler,
)
from fednoisy.algorithms.flnl.standalone.fednll import (
    FedAPNLLClientTrainer,
    FedAPNLLServerHandler,
)

from fednoisy.algorithms.flnl.misc import read_fednll_args
from fednoisy.data.dataset import FedNLLDataset
from fednoisy.utils.misc import (
    setup_seed,
    make_dirs,
    result_parser,
    now,
    load_obj,
)
from fednoisy.models.build_model import build_model, build_multi_model
from fednoisy.utils.wandb_logger import WandbLogger

from fednoisy.models.orchestra_models.build_model import build_model as build_orchestra_model


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
alg_name = "FedAP-standalone"
time_stamp=now()
cmp_out_dir = os.path.join(args.out_dir, nll_name, alg_name, exp_name, time_stamp)
args.time_stamp = time_stamp
make_dirs(cmp_out_dir)
args.cmp_out_dir = cmp_out_dir

if args.n_sys_sniffing_per_client != 0:
    args.sniffing_round = int(np.ceil(1. / args.sample_ratio) * args.n_sys_sniffing_per_client)
else:
    args.sniffing_round = 0

model = build_model(args.model, CLASS_NUM[args.dataset], dataset=args.dataset)

if args.use_orchestra or args.use_fednll:
    model = build_orchestra_model(args.model, CLASS_NUM[args.dataset], dataset=args.dataset, orchestra_dim=args.feat_dim)

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
if args.use_cs:
    handler = FedAPCSServerHandler(
        model, args.com_round, args.sample_ratio, logger=server_logger, wandb_logger=wandb_logger, args=args
    ) # server
elif args.use_orchestra:
    handler = FedAPOrchestraServerHandler(
        model, args.com_round, args.sample_ratio, logger=server_logger, wandb_logger=wandb_logger, args=args
    ) # server
elif args.use_fednll:
    handler = FedAPNLLServerHandler(
        model, args.com_round, args.sample_ratio, logger=server_logger, wandb_logger=wandb_logger, args=args
    )
else:
    handler = FedAPServerHandler(
        model, args.com_round, args.sample_ratio, logger=server_logger, wandb_logger=wandb_logger, args=args
    ) # server

# ---- FedAvg & FedAvg-RobustLoss ----
if args.use_cs:
    trainer = FedAPCSClientTrainer(
        model, args.num_clients, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args
    ) # client
elif args.use_orchestra:
    trainer = FedAPOrchestraClientTrainer(
        model, args.num_clients, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args
    ) # client
elif args.use_fednll:
    trainer = FedAPNLLClientTrainer(
        model, args.num_clients, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args
    )
else:
    trainer = FedAPClientTrainer(
        model, args.num_clients, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args
    ) # client

if args.restore:
    assert args.ckpt_dir is not None
    handler_state = load_obj(os.path.join(args.ckpt_dir, "server_handler"))
    print("loaded server handler")
    trainer_state = load_obj(os.path.join(args.ckpt_dir, "client_trainer"))
    print("loaded client trainer")
    
    handler.__setstate__(handler_state.__getstate__())
    trainer.__setstate__(trainer_state.__getstate__())

# ==== server dataset ====
handler_dataset = FedNLLDataset(args, test_preload=args.preload, train_preload=args.preload)
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
pipeline = FedAPStandalone(handler, trainer, args=args, save_best=args.save_best, out_path=cmp_out_dir, wandb_logger=wandb_logger)

pipeline.main()

if args.use_wandb:
    wandb_logger.run.finish()