from pydoc import cli
import torch
import torch.nn as nn
import torch.nn.functional as TF

from fedlab.contrib.algorithm import (
    SGDSerialClientTrainer,
    SyncServerHandler,
)

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook
)


class AverageMeter(object):
    """Compute and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TestHook(SerialClientTrainerHook, SyncServerHook):
    def __init__(self, test_interval=1) -> None:
        SerialClientTrainerHook.__init__(self)
        SyncServerHook.__init__(self)
        self.test_interval = test_interval

    def on_client_training_end(self, client_trainer, *args, **kwargs):
        if self.every_n_round(client_trainer, self.test_interval):
            self.eval_fn(client_trainer)
    
    def on_global_update_end(self, server_handler, *args, **kwargs):
        if self.every_n_round(server_handler, self.test_interval):
            self.eval_fn(server_handler)
    
    def eval_fn(self, trainer_or_handler):
        model = trainer_or_handler.model
        test_dataloader = trainer_or_handler.dataset.get_dataloader(train=False, batch_size=128)
        device = trainer_or_handler.device
        loss_fn = nn.CrossEntropyLoss()
        multimodel = hasattr(model, "models")
        loss,acc = self.test(model, test_dataloader, loss_fn, device, multimodel)
        if isinstance(trainer_or_handler, SyncServerHandler):
            trainer_or_handler._LOGGER.info(
                f"Round [{trainer_or_handler.round}/{trainer_or_handler.global_round}] server test acc: {acc*100:.2f}%, loss: {loss:.4f}"
            )
            if trainer_or_handler.wandb_logger is not None:
                trainer_or_handler.wandb_logger.run.log(
                        {
                            "server/test-loss": loss,
                            "server/test-acc": acc,
                        },
                        commit=False,
                        # step=trainer_or_handler.round,
                    ) 
        elif isinstance(trainer_or_handler, SGDSerialClientTrainer):
            trainer_or_handler._LOGGER.info(
                    f"Round {trainer_or_handler.round} client-{trainer_or_handler.g_cid} local test acc: {acc*100:.2f}%, loss: {loss:.4f}"
                )
            if trainer_or_handler.wandb_logger is not None:
                logs = {
                    f"client-{trainer_or_handler.g_cid}/test-acc": acc,
                    f"client-{trainer_or_handler.g_cid}/test-loss": loss,
                }
                trainer_or_handler.wandb_logger.run.log(logs, commit=False)

    def test(self, model, dataloader, loss_fn, device, multimodel=False, top_k=1):
        if multimodel is False:
            model.eval()
        else:
            for net in model.models:
                net.eval()

        loss_ = AverageMeter()
        acc_ = AverageMeter()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch["img"], batch["label"]
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = len(labels)

                outputs = model(inputs)
                if multimodel is True:
                    # sum over outputs of all nets
                    outputs = torch.sum(torch.stack(outputs), dim=0)

                loss = loss_fn(outputs, labels)

                # _, predicted = torch.max(outputs, 1)
                loss_.update(loss.item(), batch_size)
                # acc_.update(torch.sum(predicted.eq(labels)).item() / batch_size, batch_size)
                _, top_pred = torch.topk(outputs, dim=-1, k=top_k)
                acc_.update(torch.sum(top_pred.eq(labels.view(-1,1)).sum(dim=-1)).item() / batch_size, batch_size)
                # rank_mask = torch.tensor([1] + [i for i in range(1,k)]).view(-1, k).to(device)
                # rank_mask = 1./rank_mask
                # acc_.update(torch.sum((top_pred.eq(labels.view(-1,1)) * rank_mask).sum(dim=-1)).item() / batch_size, batch_size)
        return loss_.avg, acc_.avg
    

class EvaluateTrainHook(SerialClientTrainerHook):
    def __init__(self, model, log_annotation, eval_interval=1, eval_local_epoch: int=None) -> None:
        SerialClientTrainerHook.__init__(self)
        self.model = model
        self.log_annotation = log_annotation
        self.eval_interval = eval_interval
        self.eval_local_epoch = eval_local_epoch
        self.local_epoch_cnt = 0

    def reset_local_epoch_cnt(self):
        self.local_epoch_cnt = 0

    def local_epoch_cnt_update(self):
        self.local_epoch_cnt += 1

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        self.reset_local_epoch_cnt()

    def on_client_training_end(self, client_trainer, *args, **kwargs):
        self.reset_local_epoch_cnt()

    def on_training_epoch_end(self, client_trainer, *args, **kwargs):
        self.local_epoch_cnt_update()
        if self.every_n_round(client_trainer, self.eval_interval):
            if self.eval_local_epoch is None and self.local_epoch_cnt == client_trainer.epochs:
                return self.eval_fn(client_trainer)
            elif self.local_epoch_cnt == self.eval_local_epoch:
                return self.eval_fn(client_trainer)
    
    def eval_fn(self, trainer):
        log = self.log_annotation
        # model = trainer.model
        model = self.model
        eval_train_dataloader = trainer.dataset.get_eval_train_dataloader(trainer.args.dataset,cid=trainer.g_cid, batch_size=128) 
        # eval_train_dataloader = trainer.dataset.get_dataloader(trainer.g_cid, train=True, batch_size=trainer.batch_size)
        device = trainer.device
        loss_fn = nn.CrossEntropyLoss()
        multimodel = hasattr(model, "models")
        eval_res = self.eval_train(model, eval_train_dataloader, loss_fn, device, multimodel)

        n_discrepancy = trainer.dataset.get_noisy_discrepancy(trainer.g_cid)

        trainer._LOGGER.info(
            f"Round {trainer.round} client-{trainer.g_cid} eval train, {log}, "
            f"acc:{eval_res['acc']*100:.2f}%(c:{eval_res['clean2overall_acc']*100:.2f}%,n:{eval_res['noisy2overall_clean_acc']*100:.2f}%), "
            f"topk_acc:{eval_res['topk_acc']*100:.2f}%, "
            f"c_acc:{eval_res['clean_acc']*100:.2f}%, "
            f"n_acc:{eval_res['noisy_acc']*100:.2f}%, "
            f"loss:{eval_res['loss']:.4f}, "
            f"c_loss:{eval_res['clean_loss']:.4f}, "
            f"n_loss:{eval_res['noisy_loss']:.4f}, "
            f"noise_rate:{eval_res['noise_ratio']*100:.2f}%, "
            # f"entropy:{eval_res['entropy']:.4f}, "
            # f"n_discrepancy:{n_discrepancy:.4f}, "
        )
        if trainer.wandb_logger is not None:
            logs = {
                f"client-{trainer.g_cid}/{log}-train-acc": eval_res['acc'],
                f"client-{trainer.g_cid}/{log}-train-loss": eval_res['loss'],
                f"client-{trainer.g_cid}/{log}-train-clean-loss": eval_res['clean_loss'],
                f"client-{trainer.g_cid}/{log}-train-clean-acc": eval_res['clean_acc'],
                f"client-{trainer.g_cid}/{log}-train-noisy-loss": eval_res['noisy_loss'],
                f"client-{trainer.g_cid}/{log}-train-noisy-acc": eval_res['noisy_acc'],
                f"client-{trainer.g_cid}/{log}-train-c2co-acc": eval_res['clean2overall_acc'],
                f"client-{trainer.g_cid}/{log}-train-n2co-acc": eval_res['noisy2overall_clean_acc'],
            }
            trainer.wandb_logger.run.log(logs, commit=False)

    def eval_train(self, model, dataloader, loss_fn, device, multimodel=False):
        if multimodel is False:
            model.eval()
        else:
            for net in model.models:
                net.eval()

        loss_ = AverageMeter()
        clean_set_loss_ = AverageMeter()
        noisy_set_loss_ = AverageMeter()
        acc_ = AverageMeter()
        clean_set_acc_ = AverageMeter()
        noisy_set_acc_ = AverageMeter()

        clean2overall_acc_ = AverageMeter()
        noisy2overall_clean_acc_ = AverageMeter()
        noise_ratio_ = AverageMeter()

        entropy_ = AverageMeter()

        topk_acc_ = AverageMeter()

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels, noisy_labels = batch["img"], batch["label"], batch["noisy_label"]
                inputs = inputs.to(device)
                labels = labels.to(device)
                noisy_labels = noisy_labels.to(device)
                is_clean = labels == noisy_labels
                batch_size = len(labels)

                outputs = model(inputs)
                if multimodel is True:
                    # sum over outputs of all nets
                    outputs = torch.sum(torch.stack(outputs), dim=0)

                _, predicted = torch.max(outputs, 1)

                entropy = TF.softmax(outputs, dim=1) * TF.log_softmax(outputs, dim=1)
                entropy = -1.0 * entropy.sum(dim=1)
                entropy = entropy.mean()
                
                clean_loss = loss_fn(outputs, labels)
                loss_.update(clean_loss.item(), batch_size)
                acc_.update(torch.sum(predicted.eq(labels)).item() / batch_size, batch_size)
                entropy_.update(entropy.item(), batch_size)

                _, topk_preds = torch.topk(outputs, dim=-1, k=5)
                topk_acc_.update(torch.sum(topk_preds.eq(labels.view(-1,1)).sum(dim=-1)).item() / batch_size, batch_size)

                if torch.sum(is_clean) != 0:
                    clean_set_loss = loss_fn(outputs[is_clean], labels[is_clean])
                    clean_set_loss_.update(clean_set_loss.item(), torch.sum(is_clean).item())
                    clean_set_acc_.update(torch.sum(predicted[is_clean].eq(labels[is_clean])).item() / torch.sum(is_clean).item(), torch.sum(is_clean).item())
                    clean2overall_acc_.update(torch.sum(predicted[is_clean].eq(labels[is_clean])).item() / len(labels), len(labels))
                    
                if torch.sum(~is_clean) != 0:
                    noisy_set_loss = loss_fn(outputs[~is_clean], noisy_labels[~is_clean])
                    noisy_set_loss_.update(noisy_set_loss.item(), torch.sum(~is_clean).item())
                    noisy_set_acc_.update(torch.sum(predicted[~is_clean].eq(noisy_labels[~is_clean])).item() / torch.sum(~is_clean).item(), torch.sum(~is_clean).item())
                    noisy2overall_clean_acc_.update(torch.sum(predicted[~is_clean].eq(labels[~is_clean])).item() / len(labels), len(labels))

                noise_ratio_.update(torch.sum(~is_clean).item() / len(labels), len(labels))

        return {
            "loss": loss_.avg, 
            "acc": acc_.avg,
            "clean_loss": clean_set_loss_.avg,
            "clean_acc": clean_set_acc_.avg,
            "noisy_loss": noisy_set_loss_.avg,
            "noisy_acc": noisy_set_acc_.avg,
            "clean2overall_acc": clean2overall_acc_.avg,
            "noisy2overall_clean_acc": noisy2overall_clean_acc_.avg, # for pseudo label
            "noise_ratio": noise_ratio_.avg,
            "entropy": entropy_.avg,
            "topk_acc": topk_acc_.avg,
        }

