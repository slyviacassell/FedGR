import torch
import torch.nn as nn

from .hook import Hook, SerialClientTrainerHook, SyncServerHook


class ClientGradClipHook(SerialClientTrainerHook):
    def __init__(self, clip_grad_norm):
        self.clip_grad_norm = clip_grad_norm

    def on_optimize_step_start(self, client_trainer, *args, **kwargs):
        nn.utils.clip_grad_norm_(client_trainer.model.parameters(), self.clip_grad_norm) # if ‖g‖ ≥ threshold then g = threshold * g/‖g‖
        # nn.utils.clip_grad_value_(client_trainer.parameters(), 0.1) # if ‖g‖ ≥ max_threshold or ‖g‖ ≤ min_threshold then g = threshold, this would modify the direction of the grad
