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

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)

from fednoisy.models.orchestra_models.container import EncoderDecoder
from fednoisy.utils.misc import lid_term

class SampleMetricEvalClientHook(SerialClientTrainerHook):
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

        self.epoch_cnt = 0

        self.use_local = False if client_trainer.args.partition == "iid" else True

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

    def on_local_process_end(self, client_trainer, *args, **kwargs):
        if client_trainer.round == client_trainer.args.com_round - 1 and client_trainer.args.dataset != 'clothing1m':
            for cid in range(client_trainer.num_clients):
                dataset = client_trainer.dataset.get_dataset(cid=cid, train=True)
                pred = np.array([0 if g in client_trainer.overall_noisy_guids else 1 for g in dataset.guids])
                clean_mask = np.array(dataset.labels) == np.array(dataset.noisy_labels)
                y_true = clean_mask.astype(int)
                client_trainer._LOGGER.info(
                    f"Sample Selection Client {cid} recall: {recall_score(y_true,pred)*100:.2f}%, "
                    f"percision: {precision_score(y_true,pred)*100:.2f}%, "
                    f"f1_score: {f1_score(y_true,pred)*100:.2f}%"
                )
    
    def on_client_training_end(self, client_trainer, *args, **kwargs):
        sample_metrics = self.update_sample_metrics(client_trainer, client_trainer.args.metric_model)
        client_trainer.cs_metrics = self.cs_metric(client_trainer, sample_metrics, *args, **kwargs)

        if client_trainer.args.gmm_selection == 'intra':
            self.local_gmm(client_trainer, client_trainer.cs_metrics[0], client_trainer.cs_metrics[1], client_trainer.l_cid)

        self.epoch_cnt = 0

    def on_training_epoch_end(self, client_trainer, *args, **kwargs):
        self.epoch_cnt += 1
        if self.epoch_cnt == 1 and self.use_local:
            self.update_sample_metrics(client_trainer, metric_model="local")        
    
    def update_sample_metrics(self, client_trainer, metric_model: str):
        if metric_model == "global":
            model = client_trainer.cur_global_model
        elif metric_model == "local":
            model = client_trainer.model
        elif metric_model == "local_ema":
            model = client_trainer.local_ema_models[client_trainer.l_cid]
        else:
            raise ValueError(f"Invalid metric model: {metric_model}")

        eval_train_dataloader = client_trainer.dataset.get_eval_train_dataloader(client_trainer.args.dataset,cid=client_trainer.g_cid, batch_size=128) 
        # eval_train_dataloader = client_trainer.dataset.get_dataloader(client_trainer.g_cid, train=True, batch_size=client_trainer.batch_size)
        device = client_trainer.device
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        multimodel = hasattr(model, "models")
        sample_dynamics = self.get_sample_dynamics(model, eval_train_dataloader, loss_fn, device, multimodel)
        sample_metrics = self.sample_metrics_processing(sample_dynamics, client_trainer.l_cid, client_trainer.round)

        return sample_metrics

    def cs_metric(self, client_trainer, sample_metrics: pd.DataFrame, *args, **kwargs) -> List[torch.Tensor]:

        guids = sample_metrics["guid"].to_numpy()
        clean_mask = sample_metrics["is_clean"].to_numpy()
        if client_trainer.args.cs_metric == "loss":
            cs_metrics = sample_metrics["loss"].to_numpy()
        elif client_trainer.args.cs_metric == "loss_mean":
            cs_metrics = sample_metrics["loss_mean"].to_numpy()
        elif client_trainer.args.cs_metric == "loss_soft_mean":
            cs_metrics = sample_metrics["loss_soft_mean"].to_numpy()
        elif client_trainer.args.cs_metric == "sc":
            cs_metrics = sample_metrics["smoothed_correctness"].to_numpy()
        elif client_trainer.args.cs_metric == "scl":
            cs_metrics = sample_metrics["smoothed_correctness_loss"].to_numpy()
        elif client_trainer.args.cs_metric == "loss_ema":
            cs_metrics = sample_metrics["loss_ema"].to_numpy()

        if self.use_local:
            cs_metrics = cs_metrics * 1.0

        guids = torch.from_numpy(guids)
        clean_mask = torch.from_numpy(clean_mask)
        cs_metrics = torch.from_numpy(cs_metrics)

        return [cs_metrics, guids, clean_mask]

    def get_sample_dynamics(self, model, dataloader, loss_fn, device, multimodel=False, top_k=1):
        if multimodel is False:
            model.eval()
        else:
            for net in model.models:
                net.eval()

        sample_dynamics = {}

        with torch.no_grad():
            for batch in dataloader:
                inputs, noisy_labels, labels, guids = batch["img"], batch["noisy_label"], batch["label"], batch["guid"]
                is_clean = (noisy_labels == labels).to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                noisy_labels = noisy_labels.to(device)
                batch_size = len(noisy_labels)

                outputs = model(inputs)
                if multimodel is True:
                    # sum over outputs of all nets
                    outputs = torch.sum(torch.stack(outputs), dim=0)

                clean_loss = loss_fn(outputs, labels)
                noisy_loss = loss_fn(outputs, noisy_labels)

                _, preds = torch.max(outputs, 1)
                pred_vec = torch.softmax(outputs,dim=-1)
                pred_confi = torch.gather(pred_vec, dim=-1, index=noisy_labels.view(-1,1))
                pred_corr = preds.eq(noisy_labels).cpu()
                for guid,p_c,i_c,c_loss,n_loss,p_confi,c_y,n_y,pred,p_v in zip(guids,pred_corr,is_clean,clean_loss,noisy_loss,pred_confi,labels,noisy_labels,preds,pred_vec):
                    sample_dynamics[guid.item()] = {
                        "correctness":p_c.item(),
                        "is_clean": i_c.item(),
                        "clean_loss": c_loss.item(),
                        "noisy_loss": n_loss.item(),
                        "confidence": p_confi.item(),
                        "label": c_y.item(),
                        "noisy_label": n_y.item(),
                        "pred": pred.item(),
                        "prediction": p_v.cpu().numpy(),
                    }
 
        return sample_dynamics
    
    def sample_metrics_processing(self, sample_dynamics, cid, round) -> pd.DataFrame:
        def scaling(x):
            return (1. + x + x**2/2)

        sample_metric_container = self.sample_metric_container[cid]
        for guid,d in sample_dynamics.items():
            if guid not in sample_metric_container:
                sample_metric_container[guid] = {
                    "loss": 0,
                    "is_clean": d["is_clean"],
                    "label": d["label"],
                    "noisy_label": d["noisy_label"],

                    "accumulated_correctnes": 0,
                    "cur_correctness": 0,
                    "smoothed_correctness": 0,

                    "loss_mean": 0,
                    "loss_vari": 0,
                    "loss_soft_mean": 0,

                    "loss_ema": d["noisy_loss"],

                    "pred_history": [],

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

            alpha = 0.9
            sample_metric_container[guid]["loss_ema"] = (1. - alpha) * d["noisy_loss"] + alpha * sample_metric_container[guid]["loss_ema"]
            
            sample_metric_container[guid]["loss_soft_mean"] = (sample_metric_container[guid]["loss_soft_mean"] * cnt + scaling(d["noisy_loss"])) / (cnt + 1)
            
            sample_metric_container[guid]["accumulated_correctnes"] += d["correctness"]
            sample_metric_container[guid]["cur_correctness"] = d["correctness"]

            sample_metric_container[guid]["smoothed_correctness"] = math.e**(-(float(d["correctness"]) + sample_metric_container[guid]["accumulated_correctnes"]/(cnt+1+1e-8)))
            sample_metric_container[guid]["smoothed_correctness_loss"] = sample_metric_container[guid]["smoothed_correctness"] + sample_metric_container[guid]["loss_mean"]

            sample_metric_container[guid]["cnt"] += 1

            sample_metric_container[guid]["pred_history"].append(d["pred"])

        df = pd.DataFrame(
                [
                    [
                        g,
                        m["loss"],
                        m["is_clean"],
                        m["loss_mean"],
                        m["loss_soft_mean"],
                        m["smoothed_correctness"],
                        m["smoothed_correctness_loss"],
                        m["loss_ema"],
                    ] for g,m in sample_metric_container.items()
                ],
                columns=[
                    "guid",
                    "loss",
                    "is_clean",
                    "loss_mean",
                    "loss_soft_mean",
                    "smoothed_correctness",
                    "smoothed_correctness_loss",
                    "loss_ema",
                ]
            )

        return df
    
    def local_gmm(self, client_trainer, local_metrics, guids, cid):
        local_metrics = local_metrics.numpy()
        guids = guids.numpy()

        gmm = GaussianMixture(n_components=2,max_iter=20,tol=1e-2,reg_covar=5e-4)
        gmm.fit(local_metrics.reshape((-1, 1)))
        probs = gmm.predict_proba(local_metrics.reshape((-1, 1)))
        probs = probs[:,gmm.means_.argmin()] 
        y_pred = (probs >= 0.5)
        
        noisy_guids = guids[~y_pred]
        clean_guids = guids[y_pred]

        client_trainer.local_noisy_guids[cid] = noisy_guids
        client_trainer.local_clean_guids[cid] = clean_guids
        client_trainer.local_probs[cid] = probs
        client_trainer.local_guids[cid] = guids
    

class SampleMetricEvalServerHook(SyncServerHook):
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

        df["selected"] = df["uploaded_round"].apply(lambda x: server_handler.round - x <= self.window_sz)
        gmm_metrics = df[(df["selected"] == True)]["metric"].to_numpy()
        gmm_y_true = df[(df["selected"] == True)]["is_clean"].to_numpy()
        gmm_guids = df[(df["selected"] == True)]["guid"].to_numpy()

        gmm_model, y_pred, probs = self.gmm(gmm_metrics)

        server_handler._LOGGER.info(
            f"Round [{server_handler.round}/{server_handler.global_round}] {self.window_sz} window cs, "
            f"{server_handler.args.metric_model} {server_handler.args.cs_metric} "
            f"gmm {self.gmm_threshold}, accuracy: {accuracy_score(gmm_y_true,y_pred)*100:.2f}%, "
            f"recall: {recall_score(gmm_y_true,y_pred)*100:.2f}%, "
            f"percision: {precision_score(gmm_y_true,y_pred)*100:.2f}%, "
            f"f1_score: {f1_score(gmm_y_true,y_pred)*100:.2f}%"
        )
        
        if server_handler.wandb_logger is not None:
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
    

class DatasetCartography(SyncServerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, server_handler, *args, **kwargs):
        self.sample_metric_container = [{} for _ in range(server_handler.args.num_clients)]
        
    def on_global_update_end(self, server_handler, *args, **kwargs):
        if self.every_n_round(server_handler, 5):
            model = server_handler.model
            sample_metrics_list = []
            for cid in server_handler.cid_list:
                eval_train_dataloader = server_handler.dataset.get_eval_train_dataloader(server_handler.args.dataset,cid=cid, batch_size=128) 
                device = server_handler.device
                loss_fn = nn.CrossEntropyLoss(reduction="none")
                multimodel = hasattr(model, "models")
                sample_dynamics = self.get_sample_dynamics(model, eval_train_dataloader, loss_fn, device, multimodel)
                sample_metrics = self.sample_metrics_processing(sample_dynamics, cid, server_handler.round)
                sample_metrics_list.append(sample_metrics)
            df = pd.concat(sample_metrics_list)

            # bin the confi_mean
            # sample_metrics["confi_mean_bin"] = pd.cut(sample_metrics["confi_mean"], bins=10, labels=[i for i in range(10)])

            sns.set_theme(rc={"figure.figsize":(12.8,7.2),"figure.dpi":300},context='paper',font_scale=1.6,style='whitegrid')
            # scatter plot the df and lengend the hue with a colorbar
            ax = sns.scatterplot(
                data=sample_metrics,
                x="loss_mean",
                y="loss_vari",
                hue="confi_mean",
                style="is_clean",
            )
            
            plt.savefig(f"tmp/datamap_{server_handler.round:04}.png",)
            plt.close()

            # df["metric"] = df["fack"]
            # window_sz = 5
            # df["selected"] = df["uploaded_round"].apply(lambda x: server_handler.round - x <= window_sz)
            # gmm_metrics = df[(df["selected"] == True)]["metric"].to_numpy()
            # gmm_y_true = df[(df["selected"] == True)]["is_clean"].to_numpy()
            # gmm_guids = df[(df["selected"] == True)]["guid"].to_numpy()

            # y_pred, probs = self.gmm(gmm_metrics)

            # server_handler._LOGGER.info(
            #     f"Round [{server_handler.round}/{server_handler.global_round}] "
            #     f"recall: {recall_score(gmm_y_true,y_pred)*100:.2f}%, "
            #     f"percision: {precision_score(gmm_y_true,y_pred)*100:.2f}%, "
            #     f"f1_score: {f1_score(gmm_y_true,y_pred)*100:.2f}%"
            # )

    def gmm(self, gmm_input, gmm_test=None):
        gmm = GaussianMixture(n_components=2,max_iter=50,tol=1e-2,reg_covar=5e-4)
        gmm.fit(gmm_input.reshape((-1, 1)))
        if gmm_test is not None:
            probs = gmm.predict_proba(gmm_test.reshape((-1, 1)))
        else:
            probs = gmm.predict_proba(gmm_input.reshape((-1, 1)))
        #print("prob1=", prob)
        #print("gmm.means_.argmin()=", gmm.means_.argmin())
        probs = probs[:,gmm.means_.argmin()] #属于小loss的概率是多少 获得的是真实无噪声样本的概率是多少 该样本无噪声的概率是多少
        y_pred = (probs >= 0.5)
        return y_pred, probs

    def get_sample_dynamics(self, model, dataloader, loss_fn, device, multimodel=False, top_k=1):
        if multimodel is False:
            model.eval()
        else:
            for net in model.models:
                net.eval()

        sample_dynamics = {}

        with torch.no_grad():
            for batch in dataloader:
                inputs, noisy_labels, labels, guids = batch["img"], batch["noisy_label"], batch["label"], batch["guid"]
                is_clean = (noisy_labels == labels).to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                noisy_labels = noisy_labels.to(device)
                batch_size = len(noisy_labels)

                outputs = model(inputs)
                if multimodel is True:
                    # sum over outputs of all nets
                    outputs = torch.sum(torch.stack(outputs), dim=0)

                clean_loss = loss_fn(outputs, labels)
                noisy_loss = loss_fn(outputs, noisy_labels)

                _, preds = torch.max(outputs, 1)
                pred_vec = torch.softmax(outputs,dim=-1)
                pred_confi = torch.gather(pred_vec, dim=-1, index=preds.view(-1,1)) # predicted confi
                pred_corr = preds.eq(noisy_labels).cpu()
                for guid,p_c,i_c,c_loss,n_loss,p_confi,c_y,n_y,pred,p_v in zip(guids,pred_corr,is_clean,clean_loss,noisy_loss,pred_confi,labels,noisy_labels,preds,pred_vec):
                    sample_dynamics[guid.item()] = {
                        "correctness":p_c.item(),
                        "is_clean": i_c.item(),
                        "clean_loss": c_loss.item(),
                        "noisy_loss": n_loss.item(),
                        "confidence": p_confi.item(),
                        "label": c_y.item(),
                        "noisy_label": n_y.item(),
                        "pred": pred.item(),
                        "prediction": p_v.cpu().numpy(),
                    }
 
        return sample_dynamics
    
    def sample_metrics_processing(self, sample_dynamics, cid, round) -> pd.DataFrame:
        def scaling(x):
            return (1. + x + x**2/2)

        sample_metric_container = self.sample_metric_container[cid]
        for guid,d in sample_dynamics.items():
            if guid not in sample_metric_container:
                sample_metric_container[guid] = {
                    "loss": 0,
                    "is_clean": d["is_clean"],
                    "label": d["label"],
                    "noisy_label": d["noisy_label"],
                    "prediction": d["prediction"],

                    "uploaded_round": round,

                    "accumulated_correctnes": 0,
                    "cur_correctness": 0,
                    "smoothed_correctness": 0,

                    "loss_mean": 0,
                    "loss_vari": 0,
                    "loss_soft_mean": 0,

                    "confi_mean": 0,

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
            
            sample_metric_container[guid]["loss_soft_mean"] = (sample_metric_container[guid]["loss_soft_mean"] * cnt + scaling(d["noisy_loss"])) / (cnt + 1)
            
            sample_metric_container[guid]["accumulated_correctnes"] += d["correctness"]
            sample_metric_container[guid]["cur_correctness"] = d["correctness"]

            sample_metric_container[guid]["smoothed_correctness"] = math.e**(-(float(d["correctness"]) + sample_metric_container[guid]["accumulated_correctnes"]/(cnt+1+1e-8)))
            sample_metric_container[guid]["smoothed_correctness_loss"] = sample_metric_container[guid]["smoothed_correctness"] + sample_metric_container[guid]["loss_mean"]

            sample_metric_container[guid]["confi_mean"] = (sample_metric_container[guid]["confi_mean"] * cnt + d["confidence"]) / (cnt + 1)

            sample_metric_container[guid]["uploaded_round"] = round

            sample_metric_container[guid]["cnt"] += 1

        df = pd.DataFrame(
                [
                    [
                        g,
                        m["loss"],
                        m["is_clean"],
                        m["loss_mean"],
                        m["loss_vari"],
                        m["loss_soft_mean"],
                        m["smoothed_correctness"],
                        m["smoothed_correctness_loss"],
                        m["confi_mean"],
                        
                        m["uploaded_round"],
                    ] for g,m in sample_metric_container.items()
                ],
                columns=[
                    "guid",
                    "loss",
                    "is_clean",
                    "loss_mean",
                    "loss_vari",
                    "loss_soft_mean",
                    "smoothed_correctness",
                    "smoothed_correctness_loss",
                    "confi_mean",
                    
                    "uploaded_round",
                ]
            )

        return df