from json import load
import os
import sys
import argparse
import random
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.core.standalone import StandalonePipeline

sys.path.append(os.getcwd())
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)
from fednoisy.data.NLLData import functional as nllF
from fednoisy.utils.misc import (
    setup_seed,
    make_dirs,
    result_parser,
    now,
)
from fednoisy.models.build_model import build_model

from fednoisy.utils.wandb_logger import WandbLogger


class NLLFedAvgStandalone(StandalonePipeline):
    def __init__(
        self, handler, trainer, args, out_path, logger=None, wandb_logger: WandbLogger=None, save_best=False, save_last=True,
    ):
        super().__init__(handler, trainer)
        self._LOGGER = Logger() if logger is None else logger
        self.wandb_logger = wandb_logger
        self.save_best = save_best
        self.save_last = save_last
        self.args = args
        self.exp_name = args.exp_name
        self.nll_name = nllF.FedNLL_name(**vars(args))
        self.out_path = out_path
        self.record_file = os.path.join(self.out_path, "result_record.txt")
        self.best_model_path = os.path.join(self.out_path, "best_model.pth")
        self.last_model_path = os.path.join(self.out_path, "last_model.pth")

        self.loss_hist = []
        self.acc_hist = []
        self.max_acc = 0

    def main(self):
        # check existence of record file
        if os.path.exists(self.record_file):
            accs, _, _ = result_parser(self.record_file)
            if len(accs) >= self.args.com_round:
                self.handler._LOGGER.info(
                    f"Experiment done! Result saved in {self.record_file}!"
                )
                return

        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            # broadcast 
            broadcast = self.handler.downlink_package

            # evaluate
            if self.handler.round > 0:
                if self.args.noise_mode != "clean":
                    self.trainer.nll_monitor(broadcast, self.handler._LOGGER)
                self.evaluate(broadcast)

            # client side
            random.shuffle(sampled_clients)
            self.trainer.local_process(broadcast, sampled_clients, self.handler.round)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

        if self.save_last:            
            torch.save(
                {
                    "backbone": self.trainer.backbone.state_dict(),
                    "head": self.handler.model.state_dict(),
                    "rounds": self.args.com_round,
                },
                self.last_model_path,
            )

    def evaluate(self, broadcast):
        # trainer eval
        loss_, acc_ = self.trainer.evaluate(broadcast)
        if self.wandb_logger is not None:
            self.wandb_logger.run.log(
                    {
                        "test/test_loss": loss_,
                        "test/test_acc": acc_,
                        "other/comm_round": self.handler.round - 1,
                        "other/overall_noise_ratio": self.handler.dataset.overall_noisy_ratio,
                    },
                    step=self.handler.round - 1,
                    commit=True,
                )
        self.handler._LOGGER.info(
            f"Round [{self.handler.round - 1}/{self.handler.global_round}] test after avg: \t Loss: {loss_:.5f} \t Acc: {100*acc_:.3f}%"
        )
        self.loss_hist.append(loss_)
        self.acc_hist.append(acc_ * 100)
        record = open(self.record_file, "w")
        record.write(f"{vars(self.args)}\n")
        record.write("acc:" + str(self.acc_hist) + "\n")
        record.write("loss:" + str(self.loss_hist) + "\n")
        record.close()

        if self.save_best:
            if acc_ > self.max_acc:
                self.max_acc = acc_
                torch.save(
                    {
                        "backbone": self.trainer.backbone.state_dict(),
                        "head": self.handler.model.state_dict(),
                        "rounds": self.handler.round - 1,
                    },
                    self.best_model_path
                )
                self.handler._LOGGER.info(f"Best global model saved.")
