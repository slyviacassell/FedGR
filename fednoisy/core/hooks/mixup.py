import torch
import numpy as np
import torch.nn.functional as TF

from .hook import SerialClientTrainerHook

class LocalMixupHook(SerialClientTrainerHook):
    def __init__(self, mixup_alpha) -> None:
        super().__init__()
        if mixup_alpha > 0.:
            self.mixup_beta = mixup_alpha
        else:
            self.mixup_beta = 1.

    def mixup(self, client_trainer, inputs: torch.Tensor, targets: torch.Tensor, *args, **kwargs):
        beta = np.random.beta(self.mixup_beta, self.mixup_beta)        
        # beta = max(beta, 1-beta) # dividemix 尽量保留原始的label
        bsz = inputs.size(0)
        idx = torch.randperm(bsz)
        mixed_inputs = beta * inputs + (1 - beta) * inputs[idx]
        assert targets.shape != (bsz,)
        mixed_targets = beta * targets + (1 - beta) * targets[idx]
        return mixed_inputs, mixed_targets

