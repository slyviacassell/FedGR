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
)


class SampleMetricEvalClientHook(SerialClientTrainerHook):
    def __init__(self) -> None:
        SerialClientTrainerHook.__init__(self)

    def on_init(self, client_trainer, *args, **kwargs):
        client_trainer.overall_clean_guids = None
        client_trainer.overall_noisy_guids = None

        client_trainer.cs_metrics = None

        self.sample_metric_container = [{} for _ in range(client_trainer.num_clients)]

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        
        p = 1
        client_trainer.overall_clean_guids = client_trainer.cur_payload[p].numpy()
        client_trainer.overall_noisy_guids = client_trainer.cur_payload[p+1].numpy()
        client_trainer.overall_guids = client_trainer.cur_payload[p+2].numpy()
    
    def on_client_training_end(self, client_trainer, *args, **kwargs):
        client_trainer.cs_metrics = self.cs_metric(client_trainer, *args, **kwargs)

    def cs_metric(self, client_trainer, *args, **kwargs) -> List[torch.Tensor]:
        if client_trainer.args.metric_model == "global":
            model = client_trainer.cur_global_model
        elif client_trainer.args.metric_model == "local":
            model = client_trainer.model
        else:
            raise ValueError(f"Invalid metric model: {client_trainer.args.metric_model}")

        eval_train_dataloader = client_trainer.dataset.get_eval_train_dataloader(client_trainer.args.dataset,cid=client_trainer.g_cid, batch_size=128) 
        # eval_train_dataloader = client_trainer.dataset.get_dataloader(client_trainer.g_cid, train=True, batch_size=client_trainer.batch_size)
        device = client_trainer.device
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        multimodel = hasattr(model, "models")
        sample_dynamics = self.get_sample_dynamics(model, eval_train_dataloader, loss_fn, device, multimodel)
        sample_metrics = self.sample_metrics_processing(sample_dynamics, client_trainer.l_cid, client_trainer.round)

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
                for guid,p_c,i_c,c_loss,n_loss,p_c,c_y,n_y,pred,p_v in zip(guids,pred_corr,is_clean,clean_loss,noisy_loss,pred_confi,labels,noisy_labels,preds,pred_vec):
                    sample_dynamics[guid.item()] = {
                        "correctness":p_c.item(),
                        "is_clean": i_c.item(),
                        "clean_loss": c_loss.item(),
                        "noisy_loss": n_loss.item(),
                        "confidence": p_c.item(),
                        "label": c_y.item(),
                        "noisy_label": n_y.item(),
                        "pred": pred.item(),
                        "prediction": p_v.cpu().numpy(),
                    }
 
        return sample_dynamics
    
    def sample_metrics_processing(self, sample_dynamics, cid, round) -> pd.DataFrame:
        sample_metric_container = self.sample_metric_container[cid]
        for guid,d in sample_dynamics.items():
            if guid not in sample_metric_container:
                sample_metric_container[guid] = {
                    "loss": 0,
                    "is_clean": d["is_clean"],
                    "label": d["label"],
                    "noisy_label": d["noisy_label"],
                    "prediction": d["prediction"],

                    "cur_round": round,
                    "perv_round": round,

                    "loss_mean": 0,
                }
            
            sample_metric_container[guid]["prev_round"] = sample_metric_container[guid]["cur_round"]
            sample_metric_container[guid]["cur_round"] = round

            sample_metric_container[guid]["loss"] = d["noisy_loss"]

            sample_metric_container[guid]["loss_mean"] = (sample_metric_container[guid]["loss_mean"] * round + d["noisy_loss"]) / (round + 1)

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
    

class SampleMetricEvalServerHook(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, server_handler, *args, **kwargs):
        server_handler.clean_guids = np.zeros(1)
        server_handler.noisy_guids = np.zeros(1)
        server_handler.overall_guids = np.zeros(1)
        
        server_handler.metrics_container = {} # metrics container

        server_handler.recv_guids = np.zeros(1)
        server_handler.recv_metrics = None
        server_handler.recv_clean_mask = None

    def on_global_update_start(self, server_handler, *args, **kwargs):
        server_handler.clean_guids, \
            server_handler.noisy_guids, \
                    server_handler.overall_guids = self.central_sieving(
                        server_handler,
                        server_handler.recv_metrics, 
                        server_handler.recv_guids, 
                        server_handler.recv_clean_mask
                    )

    def central_sieving(self, server_handler, recv_metrics, recv_guids, recv_y_true):
        # update the metrics
        recv_metrics = np.concatenate(recv_metrics,axis=0)
        recv_guids = np.concatenate(recv_guids,axis=0)
        recv_y_true = np.concatenate(recv_y_true,axis=0)
        self.update_server_metrics(recv_guids, recv_metrics, server_handler.metrics_container, recv_y_true, server_handler.round)

        # get metrics for gmm
        df = self.metric_dict2df(server_handler.metrics_container)
        df["metric"] = df["metric_raw"] 

        df["selected"] = df["uploaded_round"].apply(lambda x: server_handler.round - x <= 5)
        gmm_metrics = df[(df["selected"] == True)]["metric"].to_numpy()
        gmm_y_true = df[(df["selected"] == True)]["y_true"].to_numpy()
        
        y_pred, probs = self.gmm(gmm_metrics)

        server_handler._LOGGER.info(
            f"Round [{server_handler.round}/{server_handler.global_round}] server sieving, "
            f"{server_handler.args.metric_model} {server_handler.args.cs_metric} "
            f"gmm 0.5, accuracy: {accuracy_score(gmm_y_true,y_pred)*100:.4f}%, "
            f"recall: {recall_score(gmm_y_true,y_pred)*100:.4f}%, "
            f"percision: {precision_score(gmm_y_true,y_pred)*100:.4f}%, "
            f"f1_score: {f1_score(gmm_y_true,y_pred)*100:.4f}%"
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

        gmm_guids = df[(df["selected"] == True)]["guid"].to_numpy()

        self.sample_selection(server_handler.metrics_container, gmm_guids, y_pred)
        selected_df = self.get_selected_df(server_handler.metrics_container)
        clean_guids = selected_df[selected_df["selected"] == True]["guid"].to_numpy()
        noisy_guids = selected_df[selected_df["selected"] == False]["guid"].to_numpy()
        gmm_guids = selected_df["guid"].to_numpy()

        return clean_guids, noisy_guids, gmm_guids
    
    def gmm(self, gmm_input):
        gmm = GaussianMixture(n_components=2,max_iter=50,tol=1e-2,reg_covar=5e-4)
        gmm.fit(gmm_input.reshape((-1, 1)))
        probs = gmm.predict_proba(gmm_input.reshape((-1, 1)))
        #print("prob1=", prob)
        #print("gmm.means_.argmin()=", gmm.means_.argmin())
        probs = probs[:,gmm.means_.argmin()] #属于小loss的概率是多少 获得的是真实无噪声样本的概率是多少 该样本无噪声的概率是多少
        y_pred = (probs >= 0.5)
        return y_pred, probs
    
    def sample_selection(self, metrics_container, guids, pred):
        for g, p in zip(guids, pred):
            if g in metrics_container:
                metrics_container[g]["selected"] = p
    
    def update_server_metrics(self, guids, recv_metrics, server_metrics_container, y_true, round):
        for guid, metric, y in zip(guids, recv_metrics, y_true):
            if guid not in server_metrics_container:
                server_metrics_container[guid] = {
                    "metric_raw": metric,
                    "y_true": y, # for debugging
                    "uploaded_round": round,
                    "is_clean": True,
                    "selected": True,
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
                    m["y_true"],
                    m["uploaded_round"],
                ] for guid, m in metrics_container.items()
            ],
            columns=[
                "guid", 
                "metric_raw", 
                "y_true",
                "uploaded_round",
            ]
        )
    
    def get_selected_df(self, metrics_container):
        return pd.DataFrame(
            [
                [
                    guid, 
                    m["selected"], 
                ] for guid, m in metrics_container.items()
            ],
            columns=[
                "guid", 
                "selected", 
            ]
        )
    