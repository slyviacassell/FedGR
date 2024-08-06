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
from fednoisy.algorithms.flnl.scale import (
    FedAPPassiveClientManager, 
    FedAPSynchronousServerManager,
)
from fednoisy.algorithms.flnl.scale.fedap_cs import (
    FedAPCSClientTrainer,
    FedAPCSServerHandler,
)
from fednoisy.algorithms.flnl.scale.fedap import (
    FedAPClientTrainer,
    FedAPServerHandler
)
from fednoisy.algorithms.flnl.misc import read_fednll_args
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


def main():
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

    args.num_clients = args.num_clients_per_gpu * (args.world_size - 1)

    nll_name = nllF.FedNLL_name(**vars(args))
    # exp_name = make_exp_name("fedavg", args)
    exp_name = args.exp_name
    alg_name = "FedAP-scale"
    time_stamp=now()
    cmp_out_dir = os.path.join(args.out_dir, nll_name, alg_name, exp_name,time_stamp)
    args.time_stamp = time_stamp
    make_dirs(cmp_out_dir)

    assert args.num_clients == (args.world_size - 1) * args.num_clients_per_gpu

    model = build_model(args.model, CLASS_NUM[args.dataset], dataset=args.dataset)
    
    mp.spawn(scale, args=(args, cmp_out_dir, model), nprocs=args.world_size, join=True)

def scale(rank, args, cmp_out_dir, model):

    # ==== prepare wandb logger ====
    if args.use_wandb: # todo wandb for multiprocessing
        wandb_logger = WandbLogger(
            args.wandb_project_name, 
            exp_cfg=vars(args), 
            group=args.time_stamp+"-"+args.wandb_group,
            tags=args.wandb_tags,
            job_type=args.wandb_job_type,
            name="server" if rank == 0 else f"client-gpu-{rank-1}",
        )
    else:
        wandb_logger = None

    if rank == 0:
        print(f"Start server with ip: {args.ip}, port: {args.port}")
        server_logger = Logger(
            log_name="ServerHandler",
            log_file=os.path.join(cmp_out_dir, "server.log"),
        )
        
        server_network = DistNetwork(
            address=(args.ip, args.port),
            world_size=args.world_size,
            rank=rank,
            ethernet=args.ethernet,
        ) # default timeout is 30 minutes
        
        if args.use_cs:
            handler = FedAPCSServerHandler(
                model, args.com_round, args.sample_ratio, logger=server_logger, wandb_logger=wandb_logger, args=args, device=f"cuda:{rank}",
            )
        else:
            handler = FedAPServerHandler(
                model, args.com_round, args.sample_ratio, logger=server_logger, wandb_logger=wandb_logger, args=args, device=f"cuda:{rank}",
            )  # server

        # ==== server dataset ====
        handler_dataset = FedNLLDataset(args, test_preload=args.preload, train_preload=args.preload)
        handler.setup_dataset(handler_dataset)

        server_manager = FedAPSynchronousServerManager(server_network, handler, mode="LOCAL")
        sleep(2)
        server_manager.run()
    else:
        print(f"Start gpu {rank-1}")    
        client_logger = Logger( # todo integrate multiprocessing log files
            log_name=f"ClientTrainer-gpu-{rank-1}",
            log_file=os.path.join(cmp_out_dir, f"client-gpu-{rank-1}.log"),
        )

        network = DistNetwork(
            address=(args.ip, args.port),
            world_size=args.world_size,
            rank=rank,
            ethernet=args.ethernet,
        )

        if args.use_cs:
            trainer = FedAPCSClientTrainer(
                model, args.num_clients_per_gpu, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args, device=f"cuda:{rank-1}",
            )
        else:
            # ---- FedAvg & FedAvg-RobustLoss ----
            trainer = FedAPClientTrainer(
                model, args.num_clients_per_gpu, cuda=True, logger=client_logger, wandb_logger=wandb_logger, args=args, device=f"cuda:{rank-1}",
            )  # client

        # ==== client trainer dataset ====
        trainer_dataset = FedNLLDataset(
            args, train_preload=args.preload, test_preload=args.preload
        )
        trainer.setup_dataset(trainer_dataset)
        trainer.setup_optim(
            args.epochs, args.batch_size, args.lr, args.weight_decay, args.momentum
        )

        client_manager = FedAPPassiveClientManager(trainer=trainer, network=network)
        sleep(5)
        client_manager.run()

if __name__ == "__main__":
    main()