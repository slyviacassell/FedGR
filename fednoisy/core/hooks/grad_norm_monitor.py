import copy
from collections import OrderedDict

from numpy import s_

from fednoisy.utils.ema import EMA

from .hook import SyncServerHook, SerialClientTrainerHook

class GlobalGradNormMonitorHook(SyncServerHook):
    def __init__(self) -> None:
        super().__init__()
        self.prev_global_params = None

    def on_global_update_start(self, server_handler, *args, **kwargs):
        self.prev_global_params = server_handler.model_parameters
    
    def on_global_update_end(self, server_handler, *args, **kwargs):
        model = server_handler.model
        g_params = server_handler.model_parameters
        grad_norm = 0.
        cur_idx = 0
        for s_name, s_p in model.state_dict().items():
            numel = s_p.numel()
            for name, p in model.named_parameters():
                if s_name != name:
                    continue
                assert numel == p.numel()
                if p.requires_grad:
                    tmp = (g_params[cur_idx: cur_idx + numel] - self.prev_global_params[cur_idx: cur_idx + numel]).norm(2).item()
                    grad_norm += tmp
            cur_idx += numel
        assert cur_idx == g_params.numel()
        
        server_handler._LOGGER.info(
            f"Round [{server_handler.round}/{server_handler.global_round}] Global grad norm: {grad_norm:.4f}"
        )
        if server_handler.wandb_logger is not None:
            server_handler.wandb_logger.run.log({"server/grad_norm": grad_norm}, commit=False)


class LocalGradNormMonitorHook(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()
        self.prev_client_params = None

    def on_client_training_start(self, client_trainer, *args, **kwargs):
        self.prev_client_params = client_trainer.model_parameters
    
    def on_client_training_end(self, client_trainer, *args, **kwargs):
        model = client_trainer.model
        g_params = client_trainer.model_parameters
        grad_norm = 0.
        cur_idx = 0
        for s_name, s_p in model.state_dict().items():
            numel = s_p.numel()
            for name, p in model.named_parameters():
                if s_name != name:
                    continue
                assert numel == p.numel()
                if p.requires_grad:
                    tmp = (g_params[cur_idx: cur_idx + numel] - self.prev_client_params[cur_idx: cur_idx + numel]).norm(2).item()
                    grad_norm += tmp
            cur_idx += numel
        assert cur_idx == g_params.numel()
        
        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} client-{client_trainer.g_cid} Local grad norm: {grad_norm:.4f}"
        )
        if client_trainer.wandb_logger is not None:
            client_trainer.wandb_logger.run.log(
                {
                    f"client-{client_trainer.g_cid}/grad_norm": grad_norm
                }, 
                commit=False, 
                # step=client_trainer.round,
            )