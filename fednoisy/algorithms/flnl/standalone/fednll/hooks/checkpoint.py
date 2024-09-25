import torch
import torch.nn as nn
import math
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, auc, precision_score, f1_score
from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os.path as osp

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
    SerialClientLocalEMAHook,
    SyncServerEMAHook,
)
from fednoisy.utils.misc import (
    AverageMeter,
)
from fednoisy.algorithms.flnl.hooks import (
    ClientCheckPointHook,
    ServerCheckPointHook,
)
from fednoisy.algorithms.flnl.standalone.fednll.hooks import (
    LocalOrchestra,
)
from fednoisy.utils.misc import (
    save_obj,
    load_obj,
)

# only check point client and server since both of them are not statefuless


class FedNLLClientCheckPointHook(ClientCheckPointHook):
    def __init__(self, ckpt_interval=1) -> None:
        super().__init__()
        self.ckpt_interval = ckpt_interval

    def checkpoint(self, client_trainer, *args, **kwargs):
        save_obj(client_trainer, osp.join(client_trainer.args.cmp_out_dir, f"client_trainer_{client_trainer.round}"))
        client_trainer._LOGGER.info(
            f"checkpointed client trainer at round {client_trainer.round}"
        )

        for hook_name, hook in client_trainer.hooks_dict.items():
            if isinstance(hook, SerialClientLocalEMAHook):
                for m in client_trainer.local_ema_models:
                    m.ema_model.to(client_trainer.device)
            elif isinstance(hook, LocalOrchestra):
                client_trainer.global_centroids.to(client_trainer.device)

        client_trainer.model.to(client_trainer.device)
    
    def on_local_process_end(self, client_trainer, *args, **kwargs):
        if self.every_n_round(client_trainer, self.ckpt_interval):
            self.checkpoint(client_trainer)


class FedNLLServerCheckPointHook(ServerCheckPointHook):
    def __init__(self, ckpt_interval=1) -> None:
        super().__init__()
        self.ckpt_interval = ckpt_interval

    def checkpoint(self, handler, *args, **kwargs):
        save_obj(handler, osp.join(handler.args.cmp_out_dir, f"server_handler_{handler.round}"))
        handler._LOGGER.info(
            f"checkpointed server handler at round {handler.round}"
        )

        for hook_name, hook in handler.hooks_dict.items():
            if isinstance(hook, SyncServerEMAHook):
                handler.global_ema_model.ema_model.to(handler.device)

        handler.model.to(handler.device)
    
    def on_global_update_end(self, handler, *args, **kwargs):
        if self.every_n_round(handler, self.ckpt_interval):
            self.checkpoint(handler)