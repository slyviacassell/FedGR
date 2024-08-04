import re
import torch
import torch.nn.functional as TF
import numpy as np

from .hook import SerialClientTrainerHook

class LocalELRHooK(SerialClientTrainerHook):
    def __init__(self,teacher_model) -> None:
        super().__init__()
        self.teacher_model = teacher_model

    def elr_regularize(self, client_trainer, inputs, online_outputs, *args, **kwargs):
        if self.teacher_model == "local_ema":
            teacher_model = client_trainer.local_ema_models[client_trainer.l_cid]
        elif self.teacher_model == "global_ema":
            teacher_model = client_trainer.global_ema_model
        elif self.teacher_model == "global":
            teacher_model = client_trainer.cur_global_model
        else:
            raise ValueError(f"Unknown teacher model: {self.teacher_model} for elr penalty loss.")
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        elr_penalty_loss = (1. - TF.cosine_similarity(
            torch.softmax(online_outputs,dim=-1), 
            # torch.softmax(teacher_outputs,dim=-1),
            TF.normalize(teacher_outputs, p=1, dim=-1), 
            dim=-1
        )).log().mean()
        return elr_penalty_loss
