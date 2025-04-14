import torch
import torch.nn.functional as TF

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)

class RepresentaionRegHook(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.temperature = client_trainer.args.orchestra_temperature  

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        ssl_loss = TF.kl_div(torch.log_softmax(outputs["cls_head"]["cls_embedding"] / self.temperature+1e-10, dim=1), torch.softmax(targets["simplessl_head"] / self.temperature, dim=1), reduction="batchmean")
        
        return ssl_loss