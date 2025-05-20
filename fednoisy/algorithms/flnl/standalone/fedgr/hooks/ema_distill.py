import torch
import torch.nn.functional as TF

from fednoisy.utils.misc import AverageMeter
from fednoisy.core.hooks import (
    SerialClientLocalEMAHook,
)


class EMADistillClientHook(SerialClientLocalEMAHook):
    def on_init(self, client_trainer, *args, **kwargs):
        super().on_init(client_trainer, *args, **kwargs)
        client_trainer.cid_soft_targets = [{}] * client_trainer.num_clients
        client_trainer.cid_ema_pse_labels = [{}] * client_trainer.num_clients

    def on_client_training_start(self, client_trainer, *args, **kwargs):
        cid = client_trainer.l_cid
        local_ema_model = client_trainer.local_ema_models[cid]
        local_ema_model.ema_model.to(client_trainer.device)
        if client_trainer.round < client_trainer.args.soft_silent_round:
            client_trainer._LOGGER.info(f"Round {client_trainer.round} client-{cid} local ema is totoal revised.")
            local_ema_model.copy_params_from_model_to_ema()
        else:
            pse_size = client_trainer.cid_hard_label_size[cid]
            d_size = len(client_trainer.dataset.get_dataset(client_trainer.g_cid))
            client_trainer._LOGGER.info(
                f"noise {client_trainer.est_cid_noise[cid] > client_trainer.args.upper_rate_threshold} and "
                f"small pse size {pse_size < client_trainer.args.pse_size_threshold * d_size}"
            )
            if client_trainer.est_cid_noise[cid] > client_trainer.args.upper_rate_threshold and pse_size < client_trainer.args.pse_size_threshold * d_size:
                client_trainer._LOGGER.info(
                    f"Round {client_trainer.round} client-{cid} local ema is totoal revised"
                )
                local_ema_model.copy_params_from_model_to_ema()
            else:
                if client_trainer.args.local_ema_plus_global and client_trainer.args.local_ema:
                    local_ema_plus_global_decay = client_trainer.local_ema_plus_global_decay[cid]
                    client_trainer._LOGGER.info(
                        f"Round {client_trainer.round} client-{cid} local ema is revised by global with decay {local_ema_plus_global_decay}"
                    )
                                        
                    local_ema_model.update_moving_average(
                        local_ema_model.ema_model, 
                        local_ema_model.model, # global model, use this after setup global model
                        decay=local_ema_plus_global_decay
                    )
                
        # soft_targets, ema_pse_labels = self.get_soft_targets(client_trainer)
        # client_trainer.cid_soft_targets[cid].update(soft_targets)
        # if client_trainer.round >= client_trainer.args.sniffing_round:
        #     client_trainer.cid_ema_pse_labels[cid].update(ema_pse_labels)
        
    def get_soft_targets(self, client_trainer, *args, **kwargs):
        cid = client_trainer.l_cid
        model = client_trainer.local_ema_models[cid]
        dataloader = client_trainer.dataset.get_semiws_dataloader(cid=client_trainer.g_cid, train=True, batch_size=64, drop_last=False) 
        device = client_trainer.device
        fixmatch_threshold = client_trainer.args.fixmatch_threshold
        ignore_index = -1

        model.eval()

        sample_outputs = {}
        ema_pse_labels = {}
        pse_hard_meter = AverageMeter()

        with torch.no_grad():
            for batch in dataloader:
                img_w, noisy_labels, labels, guids = batch["img_w"], batch["noisy_label"], batch["label"], batch["guid"]
                labels = labels.to(device, non_blocking=True)
                img_w = img_w.to(device, non_blocking=True)

                logits = model(img_w)
                max_confi, preds = torch.max(torch.softmax(logits, dim=-1), dim=1)

                fixmatch_mask = (max_confi > fixmatch_threshold)
                fixmatch_labels = torch.where(fixmatch_mask, preds, ignore_index)

                pse_hard_meter.update(fixmatch_mask.sum().item()/len(fixmatch_mask), len(fixmatch_mask))

                for guid,logit,fixmatch_label in zip(
                    guids,
                    logits,
                    fixmatch_labels,
                ):
                    sample_outputs[guid.item()] = logit.to("cpu")
                    if fixmatch_label != ignore_index:
                        ema_pse_labels[guid.item()] = fixmatch_label.to("cpu")

        client_trainer._LOGGER.info(f"Round {client_trainer.round} client-{client_trainer.l_cid} ema fixmatch {fixmatch_threshold} size {pse_hard_meter.sum}/{pse_hard_meter.count}")
 
        return sample_outputs, ema_pse_labels

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        sharpen_temp = 0.5
        logits = outputs["cls_head"]["cls_logits"]
        ema_distill_targets = targets["ema_distill_targets"]
        soft_loss = TF.kl_div(torch.log_softmax(logits/sharpen_temp+1e-10, dim=1), torch.softmax(ema_distill_targets/sharpen_temp, dim=1), reduction="batchmean")
        return soft_loss