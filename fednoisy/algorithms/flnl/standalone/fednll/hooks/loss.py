import torch
import torch.nn as nn
import torch.nn.functional as TF
import math
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, auc, precision_score, f1_score
from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
import scipy
from sklearn.utils.extmath import weighted_mode

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)
from fednoisy.utils.misc import (
    AverageMeter,
)
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
)

class SLSSLLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.ssl_weight = client_trainer.args.ssl_weight
        self.sl_weight = client_trainer.args.sl_weight
        self.ssl2sl_reg_weight = client_trainer.args.ssl2sl_reg_weight

    def sl_loss(self, client_trainer, outputs, targets, *args, **kwargs):
        return TF.cross_entropy(outputs["linear_head"]["cls_head"], targets["linear_head"]) * self.sl_weight
    
    def ssl_loss(self, client_trainer, outputs, targets, *args, **kwargs):
        raise NotImplementedError
    
    def ssl2sl_rep_reg(self, outputs):
        raise NotImplementedError
    
    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        sl_loss = self.sl_loss(client_trainer, outputs, targets, *args, **kwargs)
        ssl_loss = self.ssl_loss(client_trainer, outputs, targets, *args, **kwargs)
        loss = sl_loss + ssl_loss + self.ssl2sl_rep_reg(outputs) 
        return loss


class SemiSLSSLLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        # client_trainer.max_confis_ema = {cid: torch.ones(1, device=client_trainer.device)/CLASS_NUM[client_trainer.args.dataset] for cid in range(client_trainer.num_clients)}
        # client_trainer.confis_ema = {cid: (torch.ones(CLASS_NUM[client_trainer.args.dataset], device=client_trainer.device)/CLASS_NUM[client_trainer.args.dataset]) for cid in range(client_trainer.num_clients)}
        client_trainer.max_confis_ema = {cid: None for cid in range(client_trainer.num_clients)}
        client_trainer.confis_ema = {cid: None for cid in range(client_trainer.num_clients)}
        self.confi_gamma = client_trainer.args.confi_gamma
   
        self.local_epoch_cnt = 0

        self.relabels = {cid: {} for cid in range(client_trainer.num_clients)}
        self.sample_probs = {}
        self.est_cid_noise = [0] * client_trainer.args.num_clients
        self.cid_n_samples = [0] * client_trainer.args.num_clients

        self.pred_history = {cid: {} for cid in range(client_trainer.num_clients)}

    def on_client_training_end(self, client_trainer, *args, **kwargs):
        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} "
            f"Client-{client_trainer.l_cid} "
            f"max confi: {client_trainer.max_confis_ema[client_trainer.l_cid].item():.4f} "
            f"confi: {client_trainer.confis_ema[client_trainer.l_cid].cpu().numpy()}"
        )

        self.local_epoch_cnt = 0

    def on_training_epoch_start(self, client_trainer, *args, **kwargs):
        if client_trainer.round >= client_trainer.args.sniffing_round:
            if self.local_epoch_cnt == 0: # global model
                cid = client_trainer.l_cid
                data_loader = client_trainer.dataset.get_semiws_dataloader(cid=cid, train=True, batch_size=128)
                guids, pseudo_labels = self.online_pseudo_labeling(client_trainer, data_loader, cid)
                # todo: maintain the pseudo labels
                self.relabels[cid].update({g: l for g, l in zip(guids, pseudo_labels)})
                for g, p in zip(guids, pseudo_labels):
                    if g not in self.pred_history[cid]:
                        self.pred_history[cid][g] = [p]
                    else:
                        self.pred_history[cid][g].append(p)

    def on_training_epoch_end(self, client_trainer, *args, **kwargs):
        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} client-{client_trainer.g_cid} p_acc: {self.p_meter.avg*100:.2f}%"
        )
        self.p_meter.reset()

        self.local_epoch_cnt += 1
        if client_trainer.round >= client_trainer.args.sniffing_round:
            # if self.local_epoch_cnt == 1:
            cid = client_trainer.l_cid
            # data_loader = client_trainer.dataset.get_eval_train_dataloader(client_trainer.args.dataset,cid=cid, batch_size=128)
            data_loader = client_trainer.dataset.get_semiws_dataloader(cid=cid, train=True, batch_size=128)
            # data_loader = client_trainer.dataset.get_dividemix_dataloader(cid=cid, train=True, batch_size=128)
            guids, pseudo_labels = self.online_pseudo_labeling(client_trainer, data_loader, cid)
            # self.relabels.update({g: l for g, l in zip(guids, pseudo_labels)})

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        self.get_est_cid_noise(client_trainer)

        if client_trainer.round >= client_trainer.args.sniffing_round:
            self.sample_probs.update({g: c for g,c in zip(client_trainer.overall_guids.tolist(), client_trainer.overall_probs.tolist())})

    @torch.no_grad()
    def online_pseudo_labeling(self, client_trainer, dataloader, cid, *args, **kwargs):
        gmm_p_meter = AverageMeter()
        p_meter = AverageMeter()

        model = client_trainer.model
        model.eval()

        guids_list = []
        p_targets_list = []

        for batch in dataloader:
            # imgs_w, guids, noisy_targets, targets = batch["img"], batch["guid"], batch["noisy_label"], batch["label"]
            imgs_w, guids, noisy_targets, targets, imgs_s = batch["img_w"], batch["guid"], batch["noisy_label"], batch["label"], batch["img_s"]
            # imgs_w, imgs_s, guids, noisy_targets, targets = batch["img_0"], batch["img_1"], batch["guid"], batch["noisy_label"], batch["label"], batch["img_s"]

            if client_trainer.cuda:
                imgs_w = imgs_w.to(client_trainer.device)
                # imgs_s = imgs_s.to(client_trainer.device)
                noisy_targets = noisy_targets.to(client_trainer.device)
                targets = targets.to(client_trainer.device)

            outputs_w = model(imgs_w, return_dict=True, full_heads=True)
            # outputs_s = model(imgs_s, return_dict=True, full_heads=True)
            logits = outputs_w["linear_head"]["cls_head"]
            confis = torch.softmax(logits, dim=1)
            probs, preds = torch.max(confis, dim=1)

            max_confi_ema = torch.clamp(client_trainer.max_confis_ema[cid], 0.5, 0.9)
            confi_ema = client_trainer.confis_ema[cid]
            # max norm
            confi_threshold = confi_ema/confi_ema.max() * max_confi_ema
            # mask = (probs > torch.clamp(confi_threshold[preds],0.5, 0.9))
            mask = (probs > 0.9)

            noisy_mask = torch.tensor([True if g in client_trainer.overall_noisy_guids else False for g in guids.numpy()]).to(client_trainer.device)
            mask = (mask & noisy_mask)

            noisy_targets[mask] = preds[mask]
            gmm_p_meter.update(torch.mean((targets[noisy_mask] == noisy_targets[noisy_mask]).float()).item(), len(targets[noisy_mask])+1e-8)
            p_meter.update(torch.mean((targets[mask] == noisy_targets[mask]).float()).item(), len(targets[mask])+1e-8)

            # hard label
            mask = (mask & noisy_mask) | (~noisy_mask)
            p_targets_list += noisy_targets[mask].cpu().tolist() 
            guids_list += guids[mask].tolist()

            # global model confi ema
            confi_gamma = 0.9
            if client_trainer.max_confis_ema[client_trainer.l_cid] is None:
                client_trainer.max_confis_ema[client_trainer.l_cid] = torch.max(confis, dim=1)[0].mean()
            else:
                client_trainer.max_confis_ema[client_trainer.l_cid] = confi_gamma * client_trainer.max_confis_ema[client_trainer.l_cid] + (1 - confi_gamma) * torch.max(confis, dim=1)[0].mean()
        
            if client_trainer.confis_ema[client_trainer.l_cid] is None:
                client_trainer.confis_ema[client_trainer.l_cid] = torch.mean(confis, dim=0)
            else:
                client_trainer.confis_ema[client_trainer.l_cid] = confi_gamma * client_trainer.confis_ema[client_trainer.l_cid] + (1 - confi_gamma) * torch.mean(confis, dim=0)

        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} client-{cid} gmm pseudo acc: {gmm_p_meter.avg*100:.2f}%, cnt: {gmm_p_meter.sum:.0f}/{gmm_p_meter.count:.0f}, "
            f"pseudo acc: {p_meter.avg*100:.2f}%, cnt: {p_meter.sum:.0f}/{p_meter.count:.0f}"
        )

        model.train()

        return guids_list, p_targets_list

    def get_est_cid_noise(self, client_trainer, *args, **kwargs):
        for cid in client_trainer.id_list:
            dataset = client_trainer.dataset.get_dataset(cid=cid, train=True)
            n_mask = [True if g in client_trainer.overall_noisy_guids else False for g in dataset.guids]
            self.est_cid_noise[cid] = sum(n_mask)/len(n_mask)

            self.cid_n_samples[cid] = len(n_mask)

        sup_weights = [len(self.relabels[cid])/self.cid_n_samples[cid] for cid in client_trainer.id_list]
        sup_weights = np.array(sup_weights)
        mask = sup_weights < 0.2
        sup_weights[mask] = 0.0
        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} "
            f"sup weights: {sup_weights}"
        )
    
    @torch.no_grad()
    def confidence_ema(self, client_trainer, confis: torch.Tensor):
        if client_trainer.max_confis_ema[client_trainer.l_cid] is None:
            client_trainer.max_confis_ema[client_trainer.l_cid] = torch.max(confis, dim=1)[0].mean()
        else:
            client_trainer.max_confis_ema[client_trainer.l_cid] = self.confi_gamma * client_trainer.max_confis_ema[client_trainer.l_cid] + (1 - self.confi_gamma) * torch.max(confis, dim=1)[0].mean()
    
        if client_trainer.confis_ema[client_trainer.l_cid] is None:
            client_trainer.confis_ema[client_trainer.l_cid] = torch.mean(confis, dim=0)
        else:
            client_trainer.confis_ema[client_trainer.l_cid] = self.confi_gamma * client_trainer.confis_ema[client_trainer.l_cid] + (1 - self.confi_gamma) * torch.mean(confis, dim=0)


class SupOrchestraLoss(SLSSLLoss):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.orchestra_temperature = client_trainer.args.orchestra_temperature
        SLSSLLoss.on_init(self, client_trainer, *args, **kwargs)

    def loss(self, client_trainer, outputs_s, targets, *args, **kwargs):
        sl_loss = self.sl_loss(client_trainer, outputs_s, targets, *args, **kwargs)
        ssl_loss = self.ssl_loss(client_trainer, outputs_s, targets, *args, **kwargs)
        loss = sl_loss + ssl_loss + self.ssl2sl_rep_reg(outputs_s) 
        return loss
    
    def ssl_loss(self, client_trainer, outputs, targets, *args, **kwargs):
        return -torch.sum(targets["orchestra_head"] * torch.log_softmax(outputs["orchestra_head"]+1e-10, dim=1), dim=1).mean() * self.ssl_weight
    
    def ssl2sl_rep_reg(self, outputs):
        sl_normlized = TF.normalize(outputs["linear_head"]["cls_embedding"], p=2, dim=1)
        ssl_normlized = TF.normalize(outputs["orchestra_head"], p=2, dim=1)
        sl_gram = torch.mm(sl_normlized, sl_normlized.t())
        ssl_gram = torch.mm(ssl_normlized, ssl_normlized.t())
        regulizer = torch.sum((sl_gram - ssl_gram) ** 2,dim=1).mean()
        return regulizer * self.ssl2sl_reg_weight
    

class SemiOrchestraLoss(SupOrchestraLoss, SemiSLSSLLoss):
    def __init__(self) -> None:
        super().__init__()
        self.p_meter = AverageMeter()

    def on_init(self, client_trainer, *args, **kwargs):
        SupOrchestraLoss.on_init(self, client_trainer, *args, **kwargs)
        SemiSLSSLLoss.on_init(self, client_trainer, *args, **kwargs)

    def loss(self, client_trainer, outputs_w, outputs_s, targets, *args, **kwargs):
        with torch.no_grad():
            confis = torch.softmax(outputs_w["linear_head"]["cls_head"], dim=1)
            self.confidence_ema(client_trainer, confis)
        
        if client_trainer.round < client_trainer.args.sniffing_round:
            loss = super().loss(client_trainer, outputs_s, targets, *args, **kwargs)
            return loss
        elif client_trainer.round < client_trainer.args.warmup_round + client_trainer.args.sniffing_round:
            # linear decay according to the noise level and the round
            sl_weight = (1. - self.est_cid_noise[client_trainer.l_cid]) * min(1., (client_trainer.round - client_trainer.args.sniffing_round) / min(client_trainer.args.warmup_round, 50))
            sl_loss = self.sl_loss(client_trainer, outputs_s, targets, *args, **kwargs)
            sl_loss = sl_loss * sl_weight
        
            ssl_loss = self.ssl_loss(client_trainer, outputs_s, targets, *args, **kwargs)

            loss = sl_loss + ssl_loss + self.ssl2sl_rep_reg(outputs_s)
            return loss
        else:
            guids = kwargs["guids"].numpy().tolist()
            labels = kwargs["labels"]
            
            sup_targets = targets["linear_head"]

            self.p_meter.update(torch.mean((labels == sup_targets.cpu()).float()).item(), len(labels))

            noisy_mask = torch.tensor([True if g in client_trainer.overall_noisy_guids else False for g in guids]).to(client_trainer.device)
            noisy_targets = TF.one_hot(sup_targets, num_classes=CLASS_NUM[client_trainer.args.dataset]).float()
            n_probs = torch.tensor([self.sample_probs.get(g, 1) for g in guids], device=client_trainer.device, dtype=torch.float32).unsqueeze(1)
            
            # hard label
            cid = client_trainer.l_cid
            p_mask = torch.tensor([True if g in self.relabels[cid] else False for g in guids]).to(client_trainer.device)
            pseudo_targets = [self.relabels[cid].get(g, sup_targets[i]) for i, g in enumerate(guids)] 
            pseudo_targets = TF.one_hot(torch.tensor(pseudo_targets).to(client_trainer.device), num_classes=CLASS_NUM[client_trainer.args.dataset]).float()

            pseudo_targets[~noisy_mask] = noisy_targets[~noisy_mask] * n_probs[~noisy_mask] + (1. - n_probs[~noisy_mask]) * pseudo_targets[~noisy_mask]

            logits = outputs_s["linear_head"]["cls_head"]
            probs, preds = torch.max(torch.softmax(logits, dim=1), dim=1)
            
            # max_confi_ema = torch.clamp(client_trainer.max_confis_ema[cid], 0.5, 0.9)
            # confi_ema = client_trainer.confis_ema[client_trainer.l_cid]
            # # max norm
            # confi_threshold = confi_ema/confi_ema.max() * max_confi_ema
            # mask = (probs > confi_threshold[preds])
            # mask = (mask & noisy_mask) | (~noisy_mask)
            mask = p_mask

            sl_loss = -torch.sum(pseudo_targets * torch.log_softmax(logits+1e-10, dim=1), dim=1)
            if mask.sum() > 0:
                sl_loss = sl_loss[mask].mean()
            else:
                sl_loss = 0.
            sup_weights = len(self.relabels[cid])/self.cid_n_samples[cid]
            if sup_weights < 0.2:
                sup_weights = 0.0

            ssl_loss = self.ssl_loss(client_trainer, outputs_s, targets, *args, **kwargs)
            
            loss = sl_loss * sup_weights + ssl_loss + self.ssl2sl_rep_reg(outputs_s)
            return loss


# === SSL Losses ===
class SimpleSSLLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.temperature = client_trainer.args.orchestra_temperature  

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        # ssl_loss = -torch.sum(torch.softmax(targets["simplessl_head"] / self.temperature, dim=-1) * torch.log_softmax(outputs["simplessl_head"] / self.temperature + 1e-10, dim=1), dim=1).mean()
        # ssl_loss = -torch.sum(torch.softmax(targets["simplessl_head"] / self.temperature, dim=-1) * torch.log_softmax(outputs["cls_head"]["cls_embedding"] / self.temperature + 1e-10, dim=1), dim=1).mean()
        ssl_loss = TF.kl_div(torch.log_softmax(outputs["cls_head"]["cls_embedding"] / self.temperature+1e-10, dim=1), torch.softmax(targets["simplessl_head"] / self.temperature, dim=1), reduction="batchmean")
        # ssl_loss = TF.mse_loss(outputs["cls_head"]["cls_embedding"], targets["simplessl_head"], reduction="mean")
        ssl2sl_reg = self.ssl2sl_rep_reg(outputs)

        # # tmp test
        # t = 2
        # lam = 0.1
        # uniformalty = torch.pdist(outputs["simplessl_head"], p=2).pow(2)
        # ssl_loss = ssl_loss + lam * uniformalty.mul(-t).exp().mean().log()
        
        return ssl_loss, ssl2sl_reg

    def ssl2sl_rep_reg(self, outputs):
        sl_normlized = TF.normalize(outputs["cls_head"]["cls_embedding"], p=2, dim=1)
        ssl_normlized = TF.normalize(outputs["simplessl_head"], p=2, dim=1)
        sl_gram = torch.mm(sl_normlized, sl_normlized.t())
        ssl_gram = torch.mm(ssl_normlized, ssl_normlized.t())
        regulizer = torch.sum((sl_gram - ssl_gram) ** 2,dim=1).mean()
        return regulizer    


class OrchestraLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()
    
    def on_init(self, client_trainer, *args, **kwargs):
        self.orchestra_temperature = client_trainer.args.orchestra_temperature

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        ssl_loss = -torch.sum(targets["orchestra_head"] * torch.log_softmax(outputs["orchestra_head"] / self.orchestra_temperature + 1e-10, dim=1), dim=1).mean()
        ssl2sl_reg = self.ssl2sl_rep_reg(outputs)
        return ssl_loss, ssl2sl_reg
    
    def ssl2sl_rep_reg(self, outputs):
        sl_normlized = TF.normalize(outputs["cls_head"]["cls_embedding"], p=2, dim=1)
        ssl_normlized = TF.normalize(outputs["orchestra_head"], p=2, dim=1)
        sl_gram = torch.mm(sl_normlized, sl_normlized.t())
        ssl_gram = torch.mm(ssl_normlized, ssl_normlized.t())
        regulizer = torch.sum((sl_gram - ssl_gram) ** 2,dim=1).mean()
        return regulizer
    

class SimSiamLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, client_trainer, outputs, targets=None, *args, **kwargs):
        z1, z2, p1, p2 = outputs["projector_predictor"]["z"][0], outputs["projector_predictor"]["z"][1], outputs["projector_predictor"]["p"][0], outputs["projector_predictor"]["p"][1]
        ssl_loss = self.D(p1, z2) / 2 + self.D(p2, z1) / 2
        ssl2sl_reg = self.ssl2sl_rep_reg(outputs["cls_head"]["cls_embedding"][0], outputs["projector_predictor"]["z"][0]) / 2 + self.ssl2sl_rep_reg(outputs["cls_head"]["cls_embedding"][1], outputs["projector_predictor"]["z"][1]) / 2
        return ssl_loss, ssl2sl_reg
    
    def ssl2sl_rep_reg(self, sl_embeddings, ssl_embeddings):
        sl_normlized = TF.normalize(sl_embeddings, p=2, dim=1)
        ssl_normlized = TF.normalize(ssl_embeddings, p=2, dim=1)
        sl_gram = torch.mm(sl_normlized, sl_normlized.t())
        ssl_gram = torch.mm(ssl_normlized, ssl_normlized.t())
        regulizer = torch.sum((sl_gram - ssl_gram) ** 2,dim=1).mean()
        return regulizer

    def D(self, p, z):  # negative cosine similarity
        # return - TF.cosine_similarity(p, z.detach(), dim=-1).mean()
        return (1. - TF.cosine_similarity(p, z.detach(), dim=-1)).mean()


class BYOLLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()
    
    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        z1, z2, p1, p2 = outputs["projector_predictor"]["z"][0], outputs["projector_predictor"]["z"][1], outputs["projector_predictor"]["p"][0], outputs["projector_predictor"]["p"][1]
        ema_z1, ema_z2 = targets["projector_predictor"]["z"][0], targets["projector_predictor"]["z"][1]
        
        ssl_loss = self.byol_loss_fn(p1, ema_z2) / 2 + self.byol_loss_fn(p2, ema_z1) / 2
        ssl_loss = ssl_loss.mean()
        ssl2sl_reg = self.ssl2sl_rep_reg(outputs["cls_head"]["cls_embedding"][0], outputs["projector_predictor"]["z"][0]) / 2 + self.ssl2sl_rep_reg(outputs["cls_head"]["cls_embedding"][1], outputs["projector_predictor"]["z"][1]) / 2
        return ssl_loss, ssl2sl_reg
        
    def byol_loss_fn(self, x, y):
        # x = TF.normalize(x, dim=-1, p=2)
        # y = TF.normalize(y, dim=-1, p=2)
        # return 2 - 2 * (x * y).sum(dim=-1)
        return (2. - 2 * TF.cosine_similarity(x, y.detach(), dim=-1)).mean()

    
    def ssl2sl_rep_reg(self, sl_embeddings, ssl_embeddings):
        sl_normlized = TF.normalize(sl_embeddings, p=2, dim=1)
        ssl_normlized = TF.normalize(ssl_embeddings, p=2, dim=1)
        sl_gram = torch.mm(sl_normlized, sl_normlized.t())
        ssl_gram = torch.mm(ssl_normlized, ssl_normlized.t())
        regulizer = torch.sum((sl_gram - ssl_gram) ** 2,dim=1).mean()
        return regulizer
# === SSL Losses ===
  
    
class SemiSupLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.local_epoch_cnt = 0
        self.p_meter = AverageMeter()

        self.est_cid_noise = [0] * client_trainer.args.num_clients

        self.relabels = {cid: {} for cid in range(client_trainer.num_clients)}
        self.soft_pse_labels = {cid: {} for cid in range(client_trainer.num_clients)}
        self.sample_probs = {}

        client_trainer.max_confis_ema = {cid: 1./ CLASS_NUM[client_trainer.args.dataset] for cid in range(client_trainer.num_clients)}
        # client_trainer.max_confis_ema = {cid: None for cid in range(client_trainer.num_clients)}
        client_trainer.confis_ema = {cid: torch.ones(CLASS_NUM[client_trainer.args.dataset], device=client_trainer.device) / CLASS_NUM[client_trainer.args.dataset] for cid in range(client_trainer.num_clients)}
        # client_trainer.confis_ema = {cid: None for cid in range(client_trainer.num_clients)}
        self.confi_gamma = client_trainer.args.confi_gamma

        client_trainer.global_max_confi = torch.tensor(1. / CLASS_NUM[client_trainer.args.dataset], device=client_trainer.device)
        client_trainer.global_cls_confi = torch.ones(CLASS_NUM[client_trainer.args.dataset], device=client_trainer.device) / CLASS_NUM[client_trainer.args.dataset]

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        self.get_est_cid_noise(client_trainer)

        if client_trainer.round >= client_trainer.args.sniffing_round:
            self.sample_probs.update({g: c for g,c in zip(client_trainer.overall_guids.tolist(), client_trainer.overall_probs.tolist())})

    def on_training_epoch_start(self, client_trainer, *args, **kwargs):
        # self.p_meter.reset()
        
        if self.local_epoch_cnt == 0: 
            pse_size = len(self.relabels[client_trainer.g_cid])
            d_size = len(client_trainer.dataset.get_dataset(client_trainer.g_cid))

            if client_trainer.round < client_trainer.args.soft_silent_round:
                local_ema_model = client_trainer.local_ema_models[client_trainer.g_cid]
                local_ema_model.copy_params_from_model_to_ema()
            else:
                if self.est_cid_noise[client_trainer.g_cid] > client_trainer.args.upper_rate_threshold and pse_size < client_trainer.args.pse_size_threshold * d_size:
                    if client_trainer.args.local_ema and client_trainer.args.local_ema_reinit:
                        local_ema_model = client_trainer.local_ema_models[client_trainer.g_cid]
                        local_ema_model.copy_params_from_model_to_ema()

            cid = client_trainer.g_cid
            data_loader = client_trainer.dataset.get_semiws_dataloader(cid=cid, train=True, batch_size=32, drop_last=False)
            # self.global_update_confi_ema(client_trsainer, data_loader, cid)
            hard_guids, hard_pse_labels, soft_guids, soft_pse_labels = self.online_pseudo_labeling(client_trainer, data_loader, cid)
            self.soft_pse_labels[cid].update({g: l for g, l in zip(soft_guids, soft_pse_labels)})
            if client_trainer.round >= client_trainer.args.sniffing_round:
                self.relabels[cid].update({g: l for g, l in zip(hard_guids, hard_pse_labels)})

    def on_training_epoch_end(self, client_trainer, *args, **kwargs):
        # if client_trainer.round >= client_trainer.args.sniffing_round:
        #     client_trainer._LOGGER.info(
        #         f"Round {client_trainer.round} client-{client_trainer.g_cid} p_acc: {self.p_meter.avg*100:.2f}%"
        #     ) # will be different due to drop_last

        self.local_epoch_cnt += 1

    def on_client_training_end(self, client_trainer, *args, **kwargs):
        self.local_epoch_cnt = 0

    def get_est_cid_noise(self, client_trainer, *args, **kwargs):
        for cid in client_trainer.id_list:
            dataset = client_trainer.dataset.get_dataset(cid=cid, train=True)
            n_mask = [True if g in client_trainer.overall_noisy_guids else False for g in dataset.guids]
            self.est_cid_noise[cid] = sum(n_mask)/len(n_mask)

    @torch.no_grad()
    def global_update_confi_ema(self, client_trainer, dataloader, cid, *args, **kwargs):
        model = client_trainer.model
        model.eval()

        for batch in dataloader:
            imgs_w, guids, noisy_targets, targets, imgs_s = batch["img_w"], batch["guid"], batch["noisy_label"], batch["label"], batch["img_s"]

            if client_trainer.cuda:
                imgs_w = imgs_w.to(client_trainer.device)

            outputs = model(imgs_w)
            confis = torch.softmax(outputs, dim=1)

            self.local_confidence_ema(client_trainer, confis, cid, 0.9)
            self.global_confidence_ema(client_trainer, confis, self.confi_gamma)

    @torch.no_grad()
    def online_pseudo_labeling(self, client_trainer, dataloader, cid, *args, **kwargs):
        n_set_p_meter = AverageMeter()
        p_meter = AverageMeter()    

        if client_trainer.args.local_ema:
            local_ema_model = client_trainer.local_ema_models[cid]
        else:
            local_ema_model = client_trainer.model
        global_model = client_trainer.model
        global_model.eval()
        local_ema_model.eval()

        hard_guids_list = []
        soft_guids_list = []
        hard_pse_labels_list = []
        soft_pse_labels_list = []
        noisy_clean_gt = []
        noisy_clean_pred = []

        for batch in dataloader:
            # imgs_w, guids, noisy_targets, targets = batch["img"], batch["guid"], batch["noisy_label"], batch["label"]
            imgs_w, guids, noisy_targets, targets, imgs_s = batch["img_w"], batch["guid"], batch["noisy_label"], batch["label"], batch["img_s"]

            clean_mask = noisy_targets == targets
            noisy_clean_gt += clean_mask.cpu().tolist()

            if client_trainer.cuda:
                imgs_w = imgs_w.to(client_trainer.device)
                # imgs_s = imgs_s.to(client_trainer.device)
                noisy_targets = noisy_targets.to(client_trainer.device)
                targets = targets.to(client_trainer.device)

            outputs_w = global_model(imgs_w)
            ema_outputs_w = local_ema_model(imgs_w)
            # outputs_s = model(imgs_s)
            logits = outputs_w
            ema_logits = ema_outputs_w
            confis = torch.softmax(logits, dim=1)
            probs, preds = torch.max(confis, dim=1)

            if client_trainer.args.pse_method == "freematch":
                lower_bound = 1. / CLASS_NUM[client_trainer.args.dataset]
                max_confi_ema = client_trainer.global_max_confi
                # confi_ema = client_trainer.global_cls_confi
                # max_confi_ema = client_trainer.max_confis_ema[cid]
                confi_ema = client_trainer.confis_ema[cid]
                confi_threshold = confi_ema/confi_ema.max() * max_confi_ema
                confi_threshold = torch.clamp(confi_threshold, lower_bound, 0.95)
                confi_mask = (probs > confi_threshold[preds])
            elif client_trainer.args.pse_method == "fixmatch":
                confi_mask = (probs > client_trainer.args.fixmatch_threshold)
                confi_mask = (torch.max(torch.softmax(ema_logits, dim=-1),dim=-1)[0] > client_trainer.args.fixmatch_threshold) | confi_mask
            
            if self.est_cid_noise[cid] < client_trainer.args.upper_rate_threshold:
                global_noisy_mask = torch.tensor([True if g in client_trainer.overall_noisy_guids else False for g in guids.numpy()]).to(client_trainer.device)
            else:
                global_noisy_mask = torch.ones_like(guids, dtype=torch.bool).to(client_trainer.device)

            mask = (confi_mask & global_noisy_mask)

            noisy_targets[mask] = preds[mask]
            n_set_p_meter.update(torch.mean((targets[global_noisy_mask] == preds[global_noisy_mask]).float()).item(), len(targets[global_noisy_mask])+1e-8)
            p_meter.update(torch.mean((targets[mask] == preds[mask]).float()).item(), len(targets[mask])+1e-8)

            # hard label
            mask = (mask & global_noisy_mask) | (~global_noisy_mask)
            hard_pse_labels_list += noisy_targets[mask].cpu().tolist() 
            hard_guids_list += guids[mask].tolist()
            soft_pse_labels_list.append(ema_logits.cpu().numpy())
            soft_guids_list += guids.tolist()
            noisy_clean_pred += ((~global_noisy_mask)).cpu().tolist()
        
        soft_pse_labels_list = np.concatenate(soft_pse_labels_list, axis=0)

        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} client-{cid} "
            f"n_set acc: {n_set_p_meter.avg*100:.2f}%, cnt: {n_set_p_meter.sum:.0f}/{n_set_p_meter.count:.0f}, "
            f"l acc: {p_meter.avg*100:.2f}%, cnt: {p_meter.sum:.0f}/{p_meter.count:.0f}, "
            f"cs f1: {f1_score(noisy_clean_gt, noisy_clean_pred)*100:.2f}%, P: {precision_score(noisy_clean_gt, noisy_clean_pred)*100:.2f}%, R: {recall_score(noisy_clean_gt, noisy_clean_pred)*100:.2f}%, "
            f"n_rate: {(1. - sum(noisy_clean_gt)/len(noisy_clean_gt))*100:.2f}%, e {self.est_cid_noise[cid]*100:.2f}%"
        )

        return hard_guids_list, hard_pse_labels_list, soft_guids_list, soft_pse_labels_list
    
    @torch.no_grad()
    def local_confidence_ema(self, client_trainer, confis: torch.Tensor, cid, gamma):
        if client_trainer.max_confis_ema[cid] is None:
            client_trainer.max_confis_ema[cid] = torch.max(confis, dim=1)[0].mean()
        else:
            client_trainer.max_confis_ema[cid] = gamma * client_trainer.max_confis_ema[cid] + (1 - gamma) * torch.max(confis, dim=1)[0].mean()
    
        if client_trainer.confis_ema[cid] is None:
            client_trainer.confis_ema[cid] = torch.mean(confis, dim=0)
        else:
            client_trainer.confis_ema[cid] = gamma * client_trainer.confis_ema[cid] + (1 - gamma) * torch.mean(confis, dim=0)
    
    def global_confidence_ema(self, client_trainer, confis: torch.Tensor, gamma):
        if client_trainer.global_max_confi is None:
            client_trainer.global_max_confi = torch.max(confis, dim=1)[0].mean()
        else:
            client_trainer.global_max_confi = gamma * client_trainer.global_max_confi + (1 - gamma) * torch.max(confis, dim=1)[0].mean()
    
        if client_trainer.global_cls_confi is None:
            client_trainer.global_cls_confi = torch.mean(confis, dim=0)
        else:
            client_trainer.global_cls_confi = gamma * client_trainer.global_cls_confi + (1 - gamma) * torch.mean(confis, dim=0)

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        sharpen_temp = 0.5
        guids = kwargs["guids"].numpy().tolist()
        if client_trainer.round < client_trainer.args.sniffing_round:
            loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], targets["cls_head"])

            soft_pse_targets = np.array([self.soft_pse_labels[client_trainer.g_cid].get(g, np.zeros(CLASS_NUM[client_trainer.args.dataset])) for i, g in enumerate(guids)])
            soft_pse_targets = torch.tensor(soft_pse_targets, device=client_trainer.device)
            # soft_loss = -torch.sum(torch.softmax(soft_pse_targets/sharpen_temp, dim=-1) * torch.log_softmax(outputs["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), dim=1)
            # loss += soft_loss.mean() * self.soft_loss_weight(client_trainer)
            soft_loss = TF.kl_div(torch.log_softmax(outputs["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), torch.softmax(soft_pse_targets/sharpen_temp, dim=1), reduction="batchmean")
            loss += soft_loss * self.soft_loss_weight(client_trainer)
        elif client_trainer.round < client_trainer.args.warmup_round + client_trainer.args.sniffing_round:
            loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], targets["cls_head"])

            soft_pse_targets = np.array([self.soft_pse_labels[client_trainer.g_cid].get(g, np.zeros(CLASS_NUM[client_trainer.args.dataset])) for i, g in enumerate(guids)])
            soft_pse_targets = torch.tensor(soft_pse_targets, device=client_trainer.device)
            # soft_loss = -torch.sum(torch.softmax(soft_pse_targets/sharpen_temp, dim=-1) * torch.log_softmax(outputs["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), dim=1)
            # loss += soft_loss.mean() * self.soft_loss_weight(client_trainer)
            soft_loss = TF.kl_div(torch.log_softmax(outputs["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), torch.softmax(soft_pse_targets/sharpen_temp, dim=1), reduction="batchmean")
            loss += soft_loss * self.soft_loss_weight(client_trainer)
        else:
            cid = client_trainer.g_cid
            c_probs = torch.tensor([self.sample_probs.get(g, 1) for g in guids], device=client_trainer.device, dtype=torch.float32).unsqueeze(1)
            noisy_mask = torch.tensor([True if g in client_trainer.overall_noisy_guids else False for g in guids]).to(client_trainer.device)
            if self.est_cid_noise[cid] > client_trainer.args.upper_rate_threshold:
                sup_targets = torch.tensor([self.relabels[cid].get(g, -100) for i, g in enumerate(guids)],device=client_trainer.device)
                soft_pse_targets = np.array([self.soft_pse_labels[client_trainer.g_cid].get(g, np.zeros(CLASS_NUM[client_trainer.args.dataset])) for i, g in enumerate(guids)])
                soft_pse_targets = torch.tensor(soft_pse_targets, device=client_trainer.device)
                # soft_loss = -torch.sum(torch.softmax(soft_pse_targets/sharpen_temp, dim=-1) * torch.log_softmax(outputs["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), dim=1)
                soft_loss = TF.kl_div(torch.log_softmax(outputs["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), torch.softmax(soft_pse_targets/sharpen_temp, dim=1), reduction="batchmean")
                
                if not client_trainer.args.no_label_refine:
                    if (sup_targets != -100).sum() > 0:
                        loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], sup_targets, ignore_index=-100, reduction="none")
                        loss = loss.mean()
                    else:
                        loss = torch.tensor(0., device=client_trainer.device)
                else:
                    sup_targets = targets["cls_head"]
                    noisy_loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], sup_targets, ignore_index=-100, reduction="none")
                    loss = (noisy_loss * (~noisy_mask)).mean() # no label refine, only clean set

                # loss += soft_loss.mean() * self.soft_loss_weight(client_trainer)
                loss += soft_loss * self.soft_loss_weight(client_trainer)
            else:
                labels = kwargs["labels"]
                clean_mask  = kwargs["clean_mask"]
                # self.p_meter.update(torch.mean(clean_mask.float()).item(), len(clean_mask))
                
                sup_targets = targets["cls_head"]
                hard_pse_targets = torch.tensor([self.relabels[cid].get(g, -100) for i, g in enumerate(guids)], device=client_trainer.device, dtype=torch.long)
                soft_pse_targets = np.array([self.soft_pse_labels[client_trainer.g_cid].get(g, np.zeros(CLASS_NUM[client_trainer.args.dataset])) for i, g in enumerate(guids)])
                soft_pse_targets = torch.tensor(soft_pse_targets, device=client_trainer.device)

                # soft_loss = -torch.sum(torch.softmax(soft_pse_targets/sharpen_temp, dim=-1) * torch.log_softmax(outputs["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), dim=1)
                soft_loss = TF.kl_div(torch.log_softmax(outputs["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), torch.softmax(soft_pse_targets/sharpen_temp, dim=1), reduction="batchmean")

                noisy_loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], sup_targets, ignore_index=-100, reduction="none")
                pse_loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], hard_pse_targets, ignore_index=-100, reduction="none")
                
                # loss = noisy_loss * c_probs + pse_loss * (1. - c_probs) + soft_loss * self.soft_loss_weight(client_trainer)
                # loss = loss.mean()

                if not client_trainer.args.no_label_refine:
                    loss = noisy_loss * c_probs + pse_loss * (1. - c_probs)
                else:
                    loss = (noisy_loss * (~noisy_mask)).mean() # no label refine, only clean set
                loss = loss.mean() + soft_loss * self.soft_loss_weight(client_trainer)
        
        return loss

    def soft_loss_weight(self,client_trainer, *args, **kwargs):
        weight = 1.0
        a = client_trainer.args.soft_linear_up_round
        b = client_trainer.args.soft_silent_round
        assert a >= 0
        assert b >= 0
        if a == 0 and client_trainer.round >= b:
            return weight
        elif a == 0 and client_trainer.round < b:
            return 0
        else:
            return np.clip(1/a * client_trainer.round - b/a, 0, 1) * weight