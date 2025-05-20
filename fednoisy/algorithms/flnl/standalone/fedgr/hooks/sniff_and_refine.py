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
import torch.nn.functional as TF

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)

from fednoisy.models.orchestra_models.container import EncoderDecoder
from fednoisy.utils.misc import lid_term, AverageMeter
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
)


class SniffAndRefineClientHook(SerialClientTrainerHook):
    def __init__(self) -> None:
        SerialClientTrainerHook.__init__(self)

    def on_init(self, client_trainer, *args, **kwargs):
        client_trainer.overall_clean_guids = None
        client_trainer.overall_noisy_guids = None
        client_trainer.overall_guids = None
        client_trainer.overall_probs = None

        client_trainer.local_noisy_guids = [None for _ in range(client_trainer.num_clients)]
        client_trainer.local_clean_guids = [None for _ in range(client_trainer.num_clients)]
        client_trainer.local_guids = [None for _ in range(client_trainer.num_clients)]
        client_trainer.local_probs = [None for _ in range(client_trainer.num_clients)]

        client_trainer.cs_metrics = None

        self.sample_metric_container = [{} for _ in range(client_trainer.num_clients)]
        self.pse_labels = {cid: {} for cid in range(client_trainer.num_clients)}
        self.sample_probs = {}
        client_trainer.est_cid_noise = [0] * client_trainer.num_clients
        client_trainer.cid_hard_label_size = [0] * client_trainer.num_clients
        client_trainer.cid_global_reps = [{}] * client_trainer.num_clients

        self.epoch_cnt = 0

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        if client_trainer.args.gmm_selection == 'intra':
            client_trainer.overall_noisy_guids = np.concatenate([n_guids if n_guids is not None else np.zeros(1) for n_guids in client_trainer.local_noisy_guids])
            client_trainer.overall_clean_guids = np.concatenate([c_guids if c_guids is not None else np.zeros(1) for c_guids in client_trainer.local_clean_guids])
            client_trainer.overall_guids = np.concatenate([guids if guids is not None else np.zeros(1) for guids in client_trainer.local_guids])
            client_trainer.overall_probs = np.concatenate([probs if probs is not None else np.zeros(1) for probs in client_trainer.local_probs])
        elif client_trainer.args.gmm_selection == 'inter':
            p = 2
            client_trainer.overall_clean_guids = client_trainer.cur_payload[p].numpy()
            client_trainer.overall_noisy_guids = client_trainer.cur_payload[p+1].numpy()
            client_trainer.overall_guids = client_trainer.cur_payload[p+2].numpy()
            client_trainer.overall_probs = client_trainer.cur_payload[p+3].numpy()

        self.get_est_cid_noise(client_trainer)

        if client_trainer.round >= client_trainer.args.sniffing_round:
            self.sample_probs.update({g: c for g,c in zip(client_trainer.overall_guids.tolist(), client_trainer.overall_probs.tolist())})

    def on_local_process_end(self, client_trainer, *args, **kwargs):
        if client_trainer.round == client_trainer.args.com_round - 1 and client_trainer.args.dataset != 'clothing1m':
            for cid in range(client_trainer.num_clients):
                dataset = client_trainer.dataset.get_dataset(cid=cid, train=True)
                pred = np.array([0 if g in client_trainer.overall_noisy_guids else 1 for g in dataset.guids])
                clean_mask = np.array(dataset.labels) == np.array(dataset.noisy_labels)
                y_true = clean_mask.astype(int)
                client_trainer._LOGGER.info(
                    f"Sample Selection client-{cid} recall: {recall_score(y_true,pred)*100:.2f}%, "
                    f"percision: {precision_score(y_true,pred)*100:.2f}%, "
                    f"f1_score: {f1_score(y_true,pred)*100:.2f}%"
                )

    def on_client_training_start(self, client_trainer, *args, **kwargs):
        sample_outputs, pse_hard_labels = self.get_global_sample_outputs(client_trainer, client_trainer.args.metric_model)
        sample_metrics = self.sample_statistics_processing(sample_outputs, client_trainer.l_cid, client_trainer.round)
        client_trainer.cs_metrics = self.get_cs_metric(client_trainer, sample_metrics, *args, **kwargs)

        cid = client_trainer.g_cid
        if client_trainer.round >= client_trainer.args.sniffing_round:
            self.pse_labels[cid].update(pse_hard_labels) # contain clean labels

            cnt = len(self.pse_labels[cid])
            client_trainer.cid_hard_label_size[cid] = cnt
            client_trainer._LOGGER.info(f"Round {client_trainer.round} client-{cid} pse hard label size {cnt}/{len(sample_outputs)}")
    
    def on_client_training_end(self, client_trainer, *args, **kwargs):
        if client_trainer.args.gmm_selection == 'intra':
            self.local_gmm(client_trainer, client_trainer.cs_metrics[0], client_trainer.cs_metrics[1], client_trainer.l_cid)

        self.epoch_cnt = 0

    def on_training_epoch_start(self, client_trainer, *args, **kwargs):
        if self.epoch_cnt == 1: # and client_trainer.args.partition != "iid":
            sample_outputs, _ = self.get_global_sample_outputs(client_trainer, "local")
            sample_metrics = self.sample_statistics_processing(sample_outputs, client_trainer.l_cid, client_trainer.round)
            client_trainer.cs_metrics = self.get_cs_metric(client_trainer, sample_metrics, *args, **kwargs)

    def on_training_epoch_end(self, client_trainer, *args, **kwargs):
        self.epoch_cnt += 1
    
    def get_global_sample_outputs(self, client_trainer, metric_model: str): 
        if metric_model == "global":      
            model = client_trainer.cur_global_model
        if metric_model == "local":
            model = client_trainer.model
        ema_model = client_trainer.local_ema_models[client_trainer.g_cid]
        bsz = 64
        eval_train_dataloader = client_trainer.dataset.get_semiws_dataloader(cid=client_trainer.g_cid, train=True, batch_size=bsz, drop_last=False) 
        device = client_trainer.device
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        sample_outputs, pse_hard_labels = self.forward_and_fixmatch(client_trainer, model, ema_model, eval_train_dataloader, loss_fn, device, client_trainer.args.fixmatch_threshold)

        return sample_outputs, pse_hard_labels

    def get_cs_metric(self, client_trainer, sample_metrics: pd.DataFrame, *args, **kwargs) -> List[torch.Tensor]:

        guids = sample_metrics["guid"].to_numpy()
        clean_mask = sample_metrics["is_clean"].to_numpy()
        if client_trainer.args.cs_metric == "loss":
            cs_metrics = sample_metrics["loss"].to_numpy()
        elif client_trainer.args.cs_metric == "loss_mean":
            cs_metrics = sample_metrics["loss_mean"].to_numpy()

        guids = torch.from_numpy(guids)
        clean_mask = torch.from_numpy(clean_mask)
        cs_metrics = torch.from_numpy(cs_metrics)

        return [cs_metrics, guids, clean_mask]

    def forward_and_fixmatch(self, client_trainer, model, ema_model, dataloader, loss_fn, device, fixmatch_threshold):
        model.eval()
        ema_model.eval()

        sample_outputs = {}

        pse_hard_labels = {}
        ema_soft_targets = {}
        pse_hard_meter = AverageMeter()
        ignore_index = -1
        cid = client_trainer.g_cid

        with torch.no_grad():
            for batch in dataloader:
                img, img_w, noisy_labels, labels, guids = batch["img"], batch["img_w"], batch["noisy_label"], batch["label"], batch["guid"]
                is_clean = (noisy_labels == labels)
                img = img.to(device, non_blocking=True)
                img_w = img_w.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                noisy_labels = noisy_labels.to(device, non_blocking=True)

                all_outputs = model(torch.cat([img, img_w],dim=0),return_dict=True, full_heads=True)
                all_logits = all_outputs["cls_head"]["cls_logits"]
                all_reps = all_outputs["cls_head"]["cls_embedding"]
                logits, logits_w = torch.chunk(all_logits, 2, dim=0)
                reps, reps_w = torch.chunk(all_reps, 2, dim=0)
                ema_logits = ema_model(img_w)

                clean_loss = loss_fn(logits, labels)
                noisy_loss = loss_fn(logits, noisy_labels)

                max_confi_w, preds_w = torch.max(torch.softmax(logits_w,dim=-1), dim=-1)
                ema_max_confi_w, _ = torch.max(torch.softmax(ema_logits,dim=-1), dim=-1)
                fixmatch_mask = (max_confi_w > fixmatch_threshold)
                # fixmatch_mask = fixmatch_mask | (ema_max_confi_w > fixmatch_threshold)
                fixmatch_labels = torch.where(fixmatch_mask, preds_w, ignore_index)

                pse_hard_meter.update(fixmatch_mask.sum().item()/len(fixmatch_mask), len(fixmatch_mask))

                for guid,i_c,c_loss,n_loss,c_y,n_y,f_y,ema_logit,rep in zip(
                    guids,
                    is_clean,
                    clean_loss,
                    noisy_loss,
                    labels,
                    noisy_labels,
                    fixmatch_labels,
                    ema_logits,
                    reps_w,
                ):
                    sample_outputs[guid.item()] = {
                        "is_clean": i_c.item(),
                        "clean_loss": c_loss.item(),
                        "noisy_loss": n_loss.item(),
                        "label": c_y.item(),
                        "noisy_label": n_y.item(),
                    }
                    if self.epoch_cnt == 0:
                        if guid.item() not in client_trainer.overall_noisy_guids and client_trainer.est_cid_noise[cid] < client_trainer.args.upper_rate_threshold:
                            pse_hard_labels[guid.item()] = n_y.to("cpu")
                        else:
                            if f_y != ignore_index:
                                pse_hard_labels[guid.item()] = f_y.to("cpu")
                        client_trainer.cid_soft_targets[client_trainer.g_cid].update({guid.item(): ema_logit.to("cpu")})
                        client_trainer.cid_global_reps[client_trainer.g_cid].update({guid.item(): rep.to("cpu")})

        client_trainer._LOGGER.info(f"Round {client_trainer.round} client-{client_trainer.l_cid} global fixmatch {fixmatch_threshold} size {pse_hard_meter.sum}/{pse_hard_meter.count}")
    
        return sample_outputs, pse_hard_labels
    
    def sample_statistics_processing(self, sample_dynamics, cid, round) -> pd.DataFrame:
        def scaling(x):
            return (1. + x + x**2/2)

        sample_metric_container = self.sample_metric_container[cid]
        for guid,d in sample_dynamics.items():
            if guid not in sample_metric_container:
                sample_metric_container[guid] = {
                    "is_clean": d["is_clean"],
                    "label": d["label"],
                    "noisy_label": d["noisy_label"],

                    "loss": 0,
                    "loss_mean": 0,
                    "loss_vari": 0,

                    "cnt": 0,
                }

            cnt = sample_metric_container[guid]["cnt"]
            
            sample_metric_container[guid]["loss"] = d["noisy_loss"]

            # rolling mean and variance
            mean_n_1 = sample_metric_container[guid]["loss_mean"] 
            mean_n = mean_n_1 + (d["noisy_loss"] - mean_n_1) / (cnt + 1)
            sigma2_n_1 = sample_metric_container[guid]["loss_vari"]
            sigma2_n = sigma2_n_1 + ((d["noisy_loss"] - mean_n_1) * (d["noisy_loss"] - mean_n) - sigma2_n_1) / (cnt + 1)
            
            sample_metric_container[guid]["loss_mean"] = mean_n
            sample_metric_container[guid]["loss_vari"] = sigma2_n

            sample_metric_container[guid]["cnt"] += 1

        df = pd.DataFrame(
                [
                    [
                        g,
                        m["loss"],
                        m["is_clean"],
                        m["loss_mean"],
                    ] for g,m in sample_metric_container.items()
                ],
                columns=[
                    "guid",
                    "loss",
                    "is_clean",
                    "loss_mean",
                ]
            )

        return df
    
    def local_gmm(self, client_trainer, local_metrics, guids, cid):
        gmm_threshold = 0.5
        local_metrics = local_metrics.numpy()
        guids = guids.numpy()

        gmm = GaussianMixture(n_components=2,max_iter=20,tol=1e-2,reg_covar=5e-4)
        gmm.fit(local_metrics.reshape((-1, 1)))
        probs = gmm.predict_proba(local_metrics.reshape((-1, 1)))
        probs = probs[:,gmm.means_.argmin()] 
        y_pred = (probs >= gmm_threshold)
        
        noisy_guids = guids[~y_pred]
        clean_guids = guids[y_pred]

        client_trainer.local_noisy_guids[cid] = noisy_guids
        client_trainer.local_clean_guids[cid] = clean_guids
        client_trainer.local_probs[cid] = probs
        client_trainer.local_guids[cid] = guids

    def get_est_cid_noise(self, client_trainer, *args, **kwargs):
        for cid in client_trainer.id_list:
            dataset = client_trainer.dataset.get_dataset(cid=cid, train=True)
            n_mask = [True if g in client_trainer.overall_noisy_guids else False for g in dataset.guids]
            client_trainer.est_cid_noise[cid] = sum(n_mask)/len(n_mask)

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        guids = kwargs["guids"]
        cid = client_trainer.g_cid
        ignore_index = -1
        if client_trainer.round < client_trainer.args.sniffing_round:
            loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], targets["cls_head"])
        else:
            noisy_mask = torch.tensor([True if g in client_trainer.overall_noisy_guids else False for g in guids.numpy().tolist()]).to(client_trainer.device, non_blocking=True)
            pse_labels = torch.tensor([self.pse_labels[cid].get(g.item(), ignore_index) for g in guids]).to(client_trainer.device, non_blocking=True)
            # ema_pse_labels = torch.tensor([client_trainer.cid_ema_pse_labels[cid].get(g, ignore_index) for g in guids]).to(client_trainer.device, non_blocking=True) # ema
            c_probs = torch.tensor([self.sample_probs.get(g, 1) for g in guids.numpy().tolist()]).unsqueeze(1).to(client_trainer.device, non_blocking=True)
            client_noise_ratio = client_trainer.est_cid_noise[cid]

            # pse_ignore_mask = pse_labels != ignore_index
            # pse_labels = torch.where(pse_ignore_mask, pse_labels, ema_pse_labels)

            if client_noise_ratio > client_trainer.args.upper_rate_threshold:                             
                if not client_trainer.args.no_label_refine:
                    if (pse_labels != ignore_index).sum() > 0:
                        pse_loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], pse_labels, ignore_index=ignore_index, reduction="none")
                        loss = pse_loss.mean()
                    else:
                        loss = torch.tensor(0., device=client_trainer.device)
                else:
                    noisy_loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], targets["cls_head"], ignore_index=ignore_index, reduction="none")
                    loss = (noisy_loss * (~noisy_mask)).mean() # no label refine, only clean set
            else:
                pse_loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], pse_labels, ignore_index=ignore_index, reduction="none")
                noisy_loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], targets["cls_head"], ignore_index=ignore_index, reduction="none")

                if not client_trainer.args.no_label_refine:
                    loss = noisy_loss * c_probs + pse_loss * (1. - c_probs)
                    loss = loss.mean()
                else:
                    loss = (noisy_loss * (~noisy_mask)).mean() # no label refine, only clean set

        return loss
    

class SniffAndRefineServerHook(SyncServerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, server_handler, *args, **kwargs):
        server_handler.clean_guids = np.zeros(1)
        server_handler.noisy_guids = np.zeros(1)
        server_handler.overall_guids = np.zeros(1)
        server_handler.overall_probs = np.zeros(1)
        
        server_handler.metrics_container = {} # metrics container

        server_handler.recv_guids = np.zeros(1)
        server_handler.recv_metrics = None
        server_handler.recv_clean_mask = None

        server_handler.est_cid_noise = [0] * server_handler.args.num_clients

        self.gmm_threshold = 0.5
        self.window_sz = 0

    def on_global_update_start(self, server_handler, *args, **kwargs):
        if server_handler.args.gmm_selection == 'inter':
            if server_handler.round < server_handler.args.sniffing_round + server_handler.args.warmup_round or not server_handler.args.freeze_sniffing: # update the noise estimation
                gmm_model, server_handler.clean_guids, \
                    server_handler.noisy_guids, \
                            server_handler.overall_guids, \
                                server_handler.overall_probs = self.central_sieving(
                                    server_handler,
                                    server_handler.recv_metrics, 
                                    server_handler.recv_guids, 
                                    server_handler.recv_clean_mask
                                )
                self.client_noise_sniffing(server_handler, gmm_model, server_handler.recv_cid_list, server_handler.recv_metrics)
            else:
                self.central_sieving(
                    server_handler,
                    server_handler.recv_metrics, 
                    server_handler.recv_guids, 
                    server_handler.recv_clean_mask
                )
            
    def client_noise_sniffing(self, server_handler, gmm_model: GaussianMixture, recv_cid_list, recv_metrics, *args, **kwargs):
        for cid, metrics in zip(recv_cid_list,recv_metrics):
            probs = gmm_model.predict_proba(metrics.reshape((-1, 1)))
            probs = probs[:,gmm_model.means_.argmin()]
            preds = (probs < self.gmm_threshold) # 0: clean, 1: noisy
            server_handler.est_cid_noise[cid] = preds.mean()

    def central_sieving(self, server_handler, recv_metrics, recv_guids, recv_clean_mask):
        # update the metrics
        recv_metrics = np.concatenate(recv_metrics,axis=0)
        recv_guids = np.concatenate(recv_guids,axis=0)
        recv_clean_mask = np.concatenate(recv_clean_mask,axis=0)
        self.update_server_metrics(recv_guids, recv_metrics, server_handler.metrics_container, recv_clean_mask, server_handler.round)

        # get metrics for gmm
        df = self.metric_dict2df(server_handler.metrics_container)
        df["metric"] = df["metric_raw"] 

        df["selected"] = df["uploaded_round"].apply(lambda x: server_handler.round - x <= self.window_sz) # the gmm range of previous participated clients
        gmm_metrics = df[(df["selected"] == True)]["metric"].to_numpy()
        gmm_y_true = df[(df["selected"] == True)]["is_clean"].to_numpy()
        gmm_guids = df[(df["selected"] == True)]["guid"].to_numpy()

        gmm_model, y_pred, probs = self.gmm(gmm_metrics)

        if server_handler.args.dataset != 'clothing1m':
            server_handler._LOGGER.info(
                f"Round [{server_handler.round}/{server_handler.global_round}] {self.window_sz} window cs, "
                f"{server_handler.args.metric_model} {server_handler.args.cs_metric} "
                f"gmm {self.gmm_threshold}, accuracy: {accuracy_score(gmm_y_true,y_pred)*100:.2f}%, "
                f"recall: {recall_score(gmm_y_true,y_pred)*100:.2f}%, "
                f"percision: {precision_score(gmm_y_true,y_pred)*100:.2f}%, "
                f"f1_score: {f1_score(gmm_y_true,y_pred)*100:.2f}%"
            )
        
        if server_handler.wandb_logger is not None and server_handler.args.dataset != 'clothing1m':
            server_handler.wandb_logger.run.log(
                {
                    "server/cs-accuracy": accuracy_score(gmm_y_true,y_pred),
                    "server/cs-recall": recall_score(gmm_y_true,y_pred),
                    "server/cs-precision": precision_score(gmm_y_true,y_pred),
                    "server/cs-f1": f1_score(gmm_y_true,y_pred),
                },
                commit=False,
                # step=server_handler.round,
            )

            # gmm plot all metrics
            # sns.set_theme(rc={"figure.figsize":(12.8,7.2),"figure.dpi":300})
            sns.histplot(data=df[(df["selected"] == True)][["metric", "is_clean"]], x="metric", hue="is_clean", kde=True)
            fig = plt.gcf()
            server_handler.wandb_logger.run.log({f"{server_handler.args.cs_metric} gmm": wandb.Image(fig)}, commit=False)
            plt.close(fig)

        self.sample_selection(server_handler.metrics_container, gmm_guids, y_pred, probs)
        filtered_df = self.get_filtered_df(server_handler.metrics_container)
        clean_guids = filtered_df[filtered_df["filtered"] == True]["guid"].to_numpy()
        noisy_guids = filtered_df[filtered_df["filtered"] == False]["guid"].to_numpy()
        gmm_guids = filtered_df["guid"].to_numpy()
        overall_probs = filtered_df["probs"].to_numpy()

        if server_handler.args.dataset != 'clothing1m':
            y_pred = filtered_df["filtered"].to_numpy()
            y_true = filtered_df["is_clean"].to_numpy()
            server_handler._LOGGER.info(
                f"Round [{server_handler.round}/{server_handler.global_round}] filtered samples, "
                f"recall: {recall_score(y_true,y_pred)*100:.2f}%, "
                f"percision: {precision_score(y_true,y_pred)*100:.2f}%, "
                f"f1_score: {f1_score(y_true,y_pred)*100:.2f}%"
            )

        return gmm_model, clean_guids, noisy_guids, gmm_guids, overall_probs
    
    def gmm(self, gmm_input, gmm_test=None):
        gmm = GaussianMixture(n_components=2,max_iter=50,tol=1e-2,reg_covar=5e-4)
        gmm.fit(gmm_input.reshape((-1, 1)))
        if gmm_test is not None:
            probs = gmm.predict_proba(gmm_test.reshape((-1, 1)))
        else:
            probs = gmm.predict_proba(gmm_input.reshape((-1, 1)))
        probs = probs[:,gmm.means_.argmin()] #属于小loss的概率是多少 获得的是真实无噪声样本的概率是多少 该样本无噪声的概率是多少
        y_pred = (probs >= self.gmm_threshold)
        return gmm, y_pred, probs
    
    def sample_selection(self, metrics_container, guids, pred, probs):
        for g, p, c in zip(guids, pred, probs):
            if g in metrics_container:
                metrics_container[g]["filtered"] = p
                metrics_container[g]["probs"] = c
    
    def update_server_metrics(self, guids, recv_metrics, server_metrics_container, clean_mask, round):
        for guid, metric, is_clean in zip(guids, recv_metrics, clean_mask):
            if guid not in server_metrics_container:
                server_metrics_container[guid] = {
                    "metric_raw": metric,
                    "is_clean": is_clean, # for debugging
                    "uploaded_round": round,
                    "filtered": True,
                    "probs": 1.0,
                }
            else:
                server_metrics_container[guid]["metric_raw"] = metric
                server_metrics_container[guid]["uploaded_round"] = round

    def metric_dict2df(self, metrics_container):
        return pd.DataFrame(
            [
                [
                    guid, 
                    m["metric_raw"], 
                    m["is_clean"],
                    m["uploaded_round"],
                ] for guid, m in metrics_container.items()
            ],
            columns=[
                "guid", 
                "metric_raw", 
                "is_clean",
                "uploaded_round",
            ]
        )
    
    def get_filtered_df(self, metrics_container):
        return pd.DataFrame(
            [
                [
                    guid, 
                    m["filtered"], 
                    m["is_clean"],
                    m["probs"],
                ] for guid, m in metrics_container.items()
            ],
            columns=[
                "guid", 
                "filtered", 
                "is_clean",
                "probs",
            ]
        )