import torch
import torch.nn.functional as TF
import numpy as np

from .hook import SerialClientTrainerHook

class DistillationHooK(SerialClientTrainerHook):
    def __init__(self, teacher_model="local_ema") -> None:
        super().__init__()
        self.teacher_model = teacher_model

    def distill_kl_loss(self, client_trainer, inputs, online_outputs, *args, **kwargs):
        if self.teacher_model == "local_ema":
            teacher_model = client_trainer.local_ema_models[client_trainer.l_cid]
        elif self.teacher_model == "global_ema":
            teacher_model = client_trainer.global_ema_model
        elif self.teacher_model == "global":
            teacher_model = client_trainer.cur_global_model
        else:
            raise ValueError(f"Unknown teacher model: {self.teacher_model} for kl penalty loss.")
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        kl_penalty_loss = TF.kl_div(
            TF.log_softmax(online_outputs/client_trainer.args.kl_distill_temperature, dim=-1),
            TF.softmax(teacher_outputs/client_trainer.args.kl_distill_temperature,dim=-1),
            reduction='batchmean'
        ) * client_trainer.args.kl_distill_temperature * client_trainer.args.kl_distill_temperature
        
        return kl_penalty_loss