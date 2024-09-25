import copy
import torch

from fednoisy.utils.ema import EMA

from .hook import Hook, SerialClientTrainerHook, SyncServerHook


class SerialClientLocalEMAHook(SerialClientTrainerHook):
    def __init__(self):
        super(SerialClientLocalEMAHook, self).__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        client_trainer.local_ema_models = [
            EMA(
                client_trainer.model,
                beta=client_trainer.args.local_ema_beta,
                update_after_step=0,
                update_every=1,
                inv_gamma=1.0,
                power=1.0
            ) for _ in range(client_trainer.num_clients)
        ]
    
    def on_client_training_start(self, client_trainer, *args, **kwargs):
        local_ema_model = client_trainer.local_ema_models[client_trainer.l_cid]
        if client_trainer.args.local_ema_plus_global:
            local_ema_model.update_moving_average(
                local_ema_model.ema_model, 
                local_ema_model.model, # global model, use after setup global model
                decay=client_trainer.args.local_ema_plus_global_decay
            )

    def on_client_training_end(self, client_trainer, *args, **kwargs):
        local_ema_model = client_trainer.local_ema_models[client_trainer.l_cid]
        if client_trainer.round < 1 and client_trainer.args.local_ema: # init local ema model after first local training round
            local_ema_model.update()

    def on_training_epoch_start(self, client_trainer, *args, **kwargs):
        pass
    
    def on_training_batch_start(self, client_trainer, *args, **kwargs):
        pass

    def on_training_batch_end(self, client_trainer, *args, **kwargs):
        local_ema_model = client_trainer.local_ema_models[client_trainer.l_cid]
        if client_trainer.round >= 1 and client_trainer.args.local_ema:
            local_ema_model.update()

    def on_training_step_start(self, client_trainer, *args, **kwargs):
        self.on_training_batch_start(client_trainer, *args, **kwargs)

    def on_training_step_end(self, client_trainer, *args, **kwargs):
        self.on_training_batch_end(client_trainer, *args, **kwargs)

    def on_final_epoch_start(self, client_trainer, *args, **kwargs):
        pass

    def on_final_epoch_end(self, client_trainer, *args, **kwargs):
        pass

    def on_final_iter_start(self, client_trainer, *args, **kwargs):
        pass

    def on_final_iter_end(self, client_trainer, *args, **kwargs):
        pass

    def update(self, client_trainer, *args, **kwargs):
        local_ema_model = client_trainer.local_ema_models[client_trainer.l_cid]
        local_ema_model.update()
        
    @torch.no_grad()
    def ema_outputs(self, client_trainer, inputs, *args, **kwargs):
        local_ema_model = client_trainer.local_ema_models[client_trainer.l_cid]
        return local_ema_model(inputs, *args, **kwargs)


class SyncServerEMAHook(SyncServerHook):

    def __init__(self):
        super(SyncServerEMAHook, self).__init__()

    def on_init(self, server_handler, *args, **kwargs):
        server_handler.global_ema_model = EMA(
            server_handler.model,
            beta=server_handler.args.global_ema_beta,
            update_after_step=0,
            update_every=1,
            inv_gamma=1.0,
            power=1.0
        )

    def on_global_update_end(self, server_handler, *args, **kwargs):
        server_handler.global_ema_model.update()


class SerialClientGlobalEMAHook(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()
        
    def on_init(self, client_trainer, *args, **kwargs):
        client_trainer.global_ema_model = EMA(
            client_trainer.model,
            beta=client_trainer.args.global_ema_beta,
            update_after_step=0,
            update_every=1,
            inv_gamma=1.0,
            power=1.0
        )

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        global_ema_model_params = client_trainer.cur_payload[1]
        client_trainer.set_global_ema_model(global_ema_model_params)
