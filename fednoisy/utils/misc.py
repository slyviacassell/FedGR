from re import A
import numpy as np
import random
import os
import json
import ast
import torch

from typing import Dict, List, OrderedDict

from fednoisy.data import (
    CLASS_NUM,
)
import torch.nn.functional as TF
from collections import Counter
from torch.utils.data import Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import pickle

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


def setup_seed(seed: int = 0):
    """
    Args:
        seed (int): random seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, criterion, test_loader, device, multimodel=False, k=1):
    """Evaluate classify task model accuracy, allow ``model`` contains multiple networks .

    Returns:
        (loss.avg, acc.avg)
    """
    if multimodel is False:
        model.eval()
    else:
        for net in model.models:
            net.eval()

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for batch in test_loader:
            if "img_w" in batch:
                inputs, labels = batch["img_w"], batch["label"]
            else:
                inputs, labels = batch["img"], batch["label"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = len(labels)

            outputs = model(inputs)
            if multimodel is True:
                # sum over outputs of all nets
                outputs = torch.sum(torch.stack(outputs), dim=0)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item(), batch_size)
            # acc_.update(torch.sum(predicted.eq(labels)).item() / batch_size, batch_size)
            _, top_pred = torch.topk(outputs, dim=-1, k=k)
            acc_.update(torch.sum(top_pred.eq(labels.view(-1,1)).sum(dim=-1)).item() / batch_size, batch_size)
            # rank_mask = torch.tensor([1] + [i for i in range(1,k)]).view(-1, k).to(device)
            # rank_mask = 1./rank_mask
            # acc_.update(torch.sum((top_pred.eq(labels.view(-1,1)) * rank_mask).sum(dim=-1)).item() / batch_size, batch_size)

    return loss_.avg, acc_.avg

def eval_tmp(model, criterion, train_loader, device, multimodel=False, k=1):
    """Evaluate classify task model accuracy, allow ``model`` contains multiple networks .

    Returns:
        (loss.avg, acc.avg)
    """
    if multimodel is False:
        model.eval()
    else:
        for net in model.models:
            net.eval()

    overall_clean_loss_ = AverageMeter()
    overall_clean_acc_ = AverageMeter()
    overall_noisy_loss_ = AverageMeter()
    overall_noisy_acc_ = AverageMeter()
    clean_acc_ = AverageMeter()
    noisy_acc_ = AverageMeter()
    noisy_top_probs_ = AverageMeter()
    clean_top_probs_ = AverageMeter()

    with torch.no_grad():
        for batch in train_loader:
            if "img_w" in batch:
                inputs, labels, noisy_labels = batch["img_w"], batch["label"], batch["noisy_label"]
            else:
                inputs, labels, noisy_labels = batch["img"], batch["label"], batch["noisy_label"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            noisy_labels = noisy_labels.to(device)
            batch_size = len(labels)

            outputs = model(inputs)
            if multimodel is True:
                # sum over outputs of all nets
                outputs = torch.sum(torch.stack(outputs), dim=0)
            
            clean_mask = noisy_labels == labels

            clean_loss = criterion(outputs, labels)
            noisy_loss = criterion(outputs, noisy_labels)

            _, predicted = torch.max(outputs, 1)

            _, top_pred = torch.topk(outputs, dim=-1, k=k)

            top_probs,_=torch.topk(torch.softmax(outputs,dim=-1),dim=-1,k=k)

            overall_clean_loss_.update(clean_loss.item(), batch_size)
            overall_clean_acc_.update(torch.sum(predicted.eq(labels)).item() / batch_size, batch_size)
            overall_noisy_loss_.update(noisy_loss.item(), batch_size)
            overall_noisy_acc_.update(torch.sum(predicted.eq(noisy_labels)).item() / batch_size, batch_size)

            clean_acc_.update(torch.sum(predicted.eq(labels) & clean_mask).item() / (torch.sum(clean_mask).item() + 1e-8), (torch.sum(clean_mask).item() + 1e-8))
            # noisy_acc_.update(torch.sum(predicted.eq(noisy_labels) & ~clean_mask).item() / (torch.sum(~clean_mask).item() + 1e-8), (torch.sum(~clean_mask).item() + 1e-8))
            noisy_acc_.update(torch.sum(top_pred[~clean_mask].eq(labels[~clean_mask].view(-1,1))).item() / (torch.sum(~clean_mask).item() + 1e-8), (torch.sum(~clean_mask).item() + 1e-8))
            clean_top_probs_.update(top_probs[clean_mask].sum(dim=0).cpu().numpy()/ (torch.sum(clean_mask).item() + 1e-8), (torch.sum(clean_mask).item() + 1e-8))
            noisy_top_probs_.update(top_probs[~clean_mask].sum(dim=0).cpu().numpy()/ (torch.sum(~clean_mask).item() + 1e-8), (torch.sum(~clean_mask).item() + 1e-8))

    return overall_clean_loss_.avg, overall_clean_acc_.avg, overall_noisy_loss_.avg, overall_noisy_acc_.avg, clean_acc_.avg, noisy_acc_.avg, clean_top_probs_.avg, noisy_top_probs_.avg

def eval_gl(g_model, l_model, criterion, train_loader, device, multimodel=False, k=1):
    """Evaluate classify task model accuracy, allow ``model`` contains multiple networks .

    Returns:
        (loss.avg, acc.avg)
    """
    if multimodel is False:
        g_model.eval()
        l_model.eval()
    else:
        for net in g_model.models:
            net.eval()
        for net in l_model.models:
            net.eval()

    clean_acc_ = AverageMeter()
    noisy_acc_ = AverageMeter()
    noist_top_probs_ = AverageMeter()
    clean_label_num_ = AverageMeter()
    noisy_label_num_ = AverageMeter()

    t11 = AverageMeter()
    t12 = AverageMeter()
    t21 = AverageMeter()
    t22 = AverageMeter()
    t31 = AverageMeter()
    t32 = AverageMeter()
    t41 = AverageMeter()
    t42 = AverageMeter()

    g_top_2_ratio = []
    l_top_2_ratio = []
    g_top_2_labels = []
    true_labels = []
    t_mask_ = []
    with torch.no_grad():
        for batch in train_loader:
            if "img_w" in batch:
                inputs, labels, noisy_labels = batch["img_w"], batch["label"], batch["noisy_label"]
            else:
                inputs, labels, noisy_labels = batch["img"], batch["label"], batch["noisy_label"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            noisy_labels = noisy_labels.to(device)
            batch_size = len(labels)

            g_outputs = g_model(inputs)
            l_outputs = l_model(inputs)
            if multimodel is True:
                # sum over outputs of all nets
                g_outputs = torch.sum(torch.stack(g_outputs), dim=0)
                l_outputs = torch.sum(torch.stack(l_outputs), dim=0)
            
            clean_mask = noisy_labels == labels

            _, predicted = torch.max(g_outputs, 1)

            _, g_top_pred = torch.topk(g_outputs, dim=-1, k=k)
            _, l_top_pred = torch.topk(l_outputs, dim=-1, k=k)

            cnt=torch.tensor([len(torch.unique(t)) for t in torch.concat([g_top_pred,l_top_pred],dim=1)]).to(device)
            clean_label_num_.update(sum(cnt[clean_mask])/(torch.sum(clean_mask).item() + 1e-8),(torch.sum(clean_mask).item() + 1e-8))
            noisy_label_num_.update(sum(cnt[~clean_mask])/(torch.sum(~clean_mask).item() + 1e-8),(torch.sum(~clean_mask).item() + 1e-8))

            g_top_probs,_=torch.topk(torch.softmax(g_outputs,dim=-1),dim=-1,k=k)
            l_top_probs,_=torch.topk(torch.softmax(l_outputs,dim=-1),dim=-1,k=k)

            g_top_2_ratio.append((g_top_probs[~clean_mask][:,0]/g_top_probs[~clean_mask][:,1]).cpu().numpy())
            l_top_2_ratio.append((l_top_probs[~clean_mask][:,0]/l_top_probs[~clean_mask][:,1]).cpu().numpy())
            g_top_2_labels.append(g_top_pred[~clean_mask].cpu().numpy())
            true_labels.append(labels[~clean_mask].cpu().numpy())

            t=g_top_pred[~clean_mask].eq(labels[~clean_mask].view(-1,1))
            t_mask_.append(t.cpu().numpy())
            m=t[:,0].view(-1)
            t11.update(g_top_probs[~clean_mask][m].sum(dim=0).cpu().numpy()/ (torch.sum(m).item() + 1e-8), (torch.sum(m).item() + 1e-8))
            m=t[:,1].view(-1)
            t12.update(g_top_probs[~clean_mask][m].sum(dim=0).cpu().numpy()/ (torch.sum(m).item() + 1e-8), (torch.sum(m).item() + 1e-8))
            t=l_top_pred[~clean_mask].eq(labels[~clean_mask].view(-1,1))
            m=t[:,0].view(-1)
            t21.update(l_top_probs[~clean_mask][m].sum(dim=0).cpu().numpy()/ (torch.sum(m).item() + 1e-8), (torch.sum(m).item() + 1e-8))
            m=t[:,1].view(-1)
            t22.update(l_top_probs[~clean_mask][m].sum(dim=0).cpu().numpy()/ (torch.sum(m).item() + 1e-8), (torch.sum(m).item() + 1e-8))

            clean_acc_.update(torch.sum(predicted.eq(labels) & clean_mask).item() / (torch.sum(clean_mask).item() + 1e-8), (torch.sum(clean_mask).item() + 1e-8))
            # noisy_acc_.update(torch.sum(predicted.eq(noisy_labels) & ~clean_mask).item() / (torch.sum(~clean_mask).item() + 1e-8), (torch.sum(~clean_mask).item() + 1e-8))
            noisy_acc_.update(torch.sum(g_top_pred[~clean_mask].eq(labels[~clean_mask].view(-1,1))).item() / (torch.sum(~clean_mask).item() + 1e-8), (torch.sum(~clean_mask).item() + 1e-8))
            noist_top_probs_.update(g_top_probs[~clean_mask].sum(dim=0).cpu().numpy()/ (torch.sum(~clean_mask).item() + 1e-8), (torch.sum(~clean_mask).item() + 1e-8))
    print('t11', t11.avg,'t12',t12.avg,'t21',t21.avg,'t22',t22.avg)
    
    g_top_2_ratio = np.concatenate(g_top_2_ratio)
    t_mask_ = np.concatenate(t_mask_)
    print('top-1 median',np.median(g_top_2_ratio[t_mask_[:,0]]),'top-2 median',np.median(g_top_2_ratio[t_mask_[:,1]]))
    g_top_2_ratio = np.exp(2*g_top_2_ratio)
    # g_top_2_ratio = (g_top_2_ratio-g_top_2_ratio.min())/(g_top_2_ratio.max()-g_top_2_ratio.min())
    l_top_2_ratio = np.concatenate(l_top_2_ratio)
    true_labels = np.concatenate(true_labels)
    g_top_2_labels = np.concatenate(g_top_2_labels, axis=0)
    p_labels = np.take_along_axis(g_top_2_labels,np.zeros(true_labels.shape,dtype=np.int32).reshape(-1,1),axis=-1)
    p_labels = p_labels.squeeze()
    print('g_p_acc 0',np.sum(p_labels==true_labels)/len(true_labels))
    p_labels = np.take_along_axis(g_top_2_labels,np.ones(true_labels.shape,dtype=np.int32).reshape(-1,1),axis=-1)
    p_labels = p_labels.squeeze()
    print('g_p_acc 1',np.sum(p_labels==true_labels)/len(true_labels))
    mask = np.zeros(true_labels.shape,dtype=np.int32)
    t_mask = (g_top_2_ratio < ((g_top_2_ratio.mean()-g_top_2_ratio.std())))
    mask[t_mask] = 1
    p_labels = np.take_along_axis(g_top_2_labels,mask.reshape(-1,1),axis=-1)
    p_labels = p_labels.squeeze()
    print('g_p_acc mask',np.sum(p_labels==true_labels)/len(true_labels))

    return clean_acc_.avg, noisy_acc_.avg, noist_top_probs_.avg, clean_label_num_.avg, noisy_label_num_.avg

def evaluate_noisy(model, criterion, train_loader, device, multimodel=False):
    """Evaluate classify task model accuracy, allow ``model`` contains multiple networks .

    Returns:
        (loss.avg, acc.avg)
    """
    if multimodel is False:
        model.eval()
    else:
        for net in model.models:
            net.eval()

    overall_clean_loss_ = AverageMeter()
    overall_clean_acc_ = AverageMeter()
    overall_noisy_loss_ = AverageMeter()
    overall_noisy_acc_ = AverageMeter()
    clean_acc_ = AverageMeter()
    noisy_acc_ = AverageMeter()
    with torch.no_grad():
        for batch in train_loader:
            if "img_w" in batch:
                inputs, labels, noisy_labels = batch["img_w"], batch["label"], batch["noisy_label"]
            else:
                inputs, labels, noisy_labels = batch["img"], batch["label"], batch["noisy_label"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            noisy_labels = noisy_labels.to(device)
            batch_size = len(labels)

            outputs = model(inputs)
            if multimodel is True:
                # sum over outputs of all nets
                outputs = torch.sum(torch.stack(outputs), dim=0)

            clean_loss = criterion(outputs, labels)
            noisy_loss = criterion(outputs, noisy_labels)

            _, predicted = torch.max(outputs, 1)

            overall_clean_loss_.update(clean_loss.item(), batch_size)
            overall_clean_acc_.update(torch.sum(predicted.eq(labels)).item() / batch_size, batch_size)
            overall_noisy_loss_.update(noisy_loss.item(), batch_size)
            overall_noisy_acc_.update(torch.sum(predicted.eq(noisy_labels)).item() / batch_size, batch_size)

            clean_mask = noisy_labels == labels
            clean_acc_.update(torch.sum(predicted.eq(labels) & clean_mask).item() / (torch.sum(clean_mask).item() + 1e-8), (torch.sum(clean_mask).item() + 1e-8))
            noisy_acc_.update(torch.sum(predicted.eq(noisy_labels) & ~clean_mask).item() / (torch.sum(~clean_mask).item() + 1e-8), (torch.sum(~clean_mask).item() + 1e-8))

    return overall_clean_loss_.avg, overall_clean_acc_.avg, overall_noisy_loss_.avg, overall_noisy_acc_.avg, clean_acc_.avg, noisy_acc_.avg

def eval_ebemdding(model, test_loader, device, multimodel=False, ):
    if multimodel is False:
        model.eval()
    else:
        for net in model.models:
            net.eval()
    
    features = OrderedDict()
    acc_ = AverageMeter()
    with torch.no_grad():
        for batch in test_loader:
            if "img_w" in batch:
                inputs, labels, guid, noisy_label = batch["img_w"], batch["label"], batch["guid"], batch["noisy_label"]
            else:
                inputs, labels, guid, noisy_label = batch["img"], batch["label"], batch["guid"], batch["noisy_label"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = len(labels)

            outputs = model.embedding(inputs)
            if multimodel is True:
                # sum over outputs of all nets
                outputs = torch.sum(torch.stack(outputs), dim=0)
            for f,y,ny,g in zip(outputs,labels,noisy_label,guid):
                features[g.item()] = {
                    "feature": f.cpu(),
                    "label": y.cpu(),
                    "noisy_label": ny.cpu(),
                }
    return features

def eval_dynamics(model, data_loader, device, multimodel=False, noisy_label_key="noisy_label"):
    if multimodel is False:
        model.eval()
    else:
        for net in model.models:
            net.eval()

    sample_dynamics = {}

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for batch in data_loader:
            inputs, noisy_labels, labels, guid = batch["img"], batch[noisy_label_key], batch["label"], batch["guid"]
            is_clean = noisy_labels == labels
            inputs = inputs.to(device)
            noisy_labels = noisy_labels.to(device)
            batch_size = len(noisy_labels)

            outputs = model(inputs)
            if multimodel is True:
                # sum over outputs of all nets
                outputs = torch.sum(torch.stack(outputs), dim=0)

            loss = TF.cross_entropy(outputs, noisy_labels,reduction='none')
            _, predicted = torch.max(outputs, 1)
            prediction = torch.softmax(outputs,dim=-1)
            confi = torch.gather(prediction, dim=-1, index=noisy_labels.view(-1,1))
            corr = predicted.eq(noisy_labels).cpu()
            for g,c,ic,l,cf,cl,nl,pd,pt in zip(guid,corr,is_clean,loss,confi,labels,noisy_labels,predicted,prediction):
                sample_dynamics[g.item()] = {
                    "correctness":c.item(),
                    "is_clean": ic.item(),
                    "loss": l.item(),
                    "confidence": cf.item(),
                    "label": cl.item(),
                    "noisy_label": nl.item(),
                    "predicted": pd.item(),
                    "prediction": pt.cpu().numpy(),
                }
    return sample_dynamics

def eval_js_div(model,dataset_name, test_loader, device, multimodel=False):
    """Evaluate classify task model accuracy, allow ``model`` contains multiple networks .

    Returns:
        (loss.avg, acc.avg)
    """
    if multimodel is False:
        model.eval()
    else:
        for net in model.models:
            net.eval()

    pred_ = AverageMeter()
    gt_ = AverageMeter()
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels, noisy_labels = batch["img"], batch["label"], batch["noisy_label"]
            selected = noisy_labels == labels
            selected = selected.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = len(labels)

            outputs = model(inputs)
            if multimodel is True:
                # sum over outputs of all nets
                outputs = torch.sum(torch.stack(outputs), dim=0)

            _, predicted = torch.max(outputs, 1)
            pred = TF.one_hot(predicted,num_classes=CLASS_NUM[dataset_name])+1e-8
            # pred = torch.softmax(outputs,dim=-1)**(1/0.5)
            pred = pred /pred.sum(dim=-1, keepdim=True)
            sum_pred = torch.sum(pred,dim=0)
            gt = TF.one_hot(labels,num_classes=CLASS_NUM[dataset_name])+1e-8
            gt = gt/gt.sum(dim=-1, keepdim=True)
            sum_gt = torch.sum(gt,dim=0)
            
            pred_.update(sum_pred.cpu()/torch.sum(selected).item(),torch.sum(selected).item())
            gt_.update(sum_gt.cpu()/batch_size,batch_size)
    uniform_distri = torch.ones(CLASS_NUM[dataset_name])/CLASS_NUM[dataset_name]
    return js_div(uniform_distri,pred_.avg).item(), js_div(uniform_distri,gt_.avg).item()

def get_label_distri(labels: np.array) -> torch.Tensor:
    labels_distri = []
    label_counter = Counter(labels)
    for cls, count in label_counter.items():
        labels_distri.append(count)
    return torch.Tensor(labels_distri)

def get_correctness(model, criterion, dataset, data_loader, comm_round, device, multimodel=False,pesudo_label=None):
    """Prediction and get per-sample correctness for classify task, allow ``model`` contains multiple networks .

    Returns:
        (pred, labels)
    """
    if multimodel is False:
        model.eval()
    else:
        for net in model.models:
            net.eval()

    correctness_ = {}
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, gold_labels, guids, noisy_labels = batch["img"], batch["label"], batch["guid"], batch["noisy_label"]

            if pesudo_label is not None:
                labels = torch.tensor([pesudo_label[guid.item()] for guid in guids], dtype=gold_labels.dtype)
            else:
                labels = noisy_labels
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = len(labels)

            outputs = model(inputs)

            if multimodel is True:
                # sum over outputs of all nets
                outputs = torch.sum(torch.stack(outputs), dim=0)
            loss = criterion(outputs, labels) # per-sample loss

            _, predicted = torch.max(outputs, 1)

            correctness = predicted.eq(labels)

            for guid, correct, l, nl, loss in zip(guids, correctness, gold_labels, noisy_labels,loss):
                correctness_[guid.item()] = {'correctness': correct.item(), 'gold': l.item(), 'noisy_label': nl.item(),'loss': loss.item()}
    
    return correctness_

def get_dynamics(model, criterion, dataset, data_loader, comm_round, device, multimodel=False, is_train_dataloader=False):
    """Prediction for classify task, allow ``model`` contains multiple networks .

    Returns:
        (pred, labels)
    """
    if multimodel is False:
        model.eval()
    else:
        for net in model.models:
            net.eval()

    dataset_logits = []
    dataset_preds = []
    dataset_labels = []
    dataset_noisy_labels = []
    dataset_guids = []
    dataset_loss = []
    dataset_margin = []

    for name,param in model.named_parameters():
        if 'weight' in name:
            if param.size()[0] == CLASS_NUM[dataset]:
                print('top layer weight name:', name)
                w = param
    weight_norm = torch.norm(w,p=2,dim=1)

    with torch.no_grad():
        for batch in data_loader:
            if is_train_dataloader:
                pass
                inputs, labels, guids, noisy_labels = batch["img"], batch["label"], batch["guid"], batch["noisy_label"]
            else:
                inputs, labels, guids = batch["img"], batch["label"], batch["guid"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = len(labels)

            outputs = model(inputs)

            if multimodel is True:
                # sum over outputs of all nets
                outputs = torch.sum(torch.stack(outputs), dim=0)
            
            loss = criterion(outputs, labels) # per-sample loss

            _, predicted = torch.max(outputs, 1)

            margin = outputs / weight_norm.view(1,-1)
            # select cls margin
            # margin = torch.gather(margin, dim=1, index=predicted.view(-1,1))
            # margin = margin.squeeze()

            dataset_logits.append(outputs.cpu().numpy())
            dataset_margin.append(margin.cpu().numpy())
            dataset_labels.append(labels.cpu().numpy())
            if is_train_dataloader:
                dataset_noisy_labels.append(noisy_labels.cpu().numpy())
            dataset_guids.append(guids.numpy())
            dataset_preds.append(predicted.cpu().numpy())
            dataset_loss.append(loss.cpu().numpy())
    
    dataset_logits = np.concatenate(dataset_logits,axis=0)
    dataset_labels = np.concatenate(dataset_labels,axis=0)
    dataset_margin = np.concatenate(dataset_margin,axis=0)
    if is_train_dataloader:
        dataset_noisy_labels = np.concatenate(dataset_noisy_labels,axis=0)
    dataset_guids = np.concatenate(dataset_guids,axis=0)
    dataset_preds = np.concatenate(dataset_preds,axis=0)
    dataset_loss = np.concatenate(dataset_loss,axis=0)
    
    dataset_dynamics = []
    if is_train_dataloader:
        for logit, label, guid, pred, noisy_label, loss, margin in zip(dataset_logits.tolist(), dataset_labels.tolist(), dataset_guids.tolist(), dataset_preds.tolist(),dataset_noisy_labels.tolist(), dataset_loss.tolist(), dataset_margin.tolist()):
            record = {'guid': guid, 'logits_round_%s'%comm_round: logit, 'gold': label, 'pred': pred, 'device': str(device), 'noisy_label': noisy_label, 'loss': loss, 'margin': margin}
            dataset_dynamics.append(record)
    else:
        for logit, label, guid, pred, loss, margin in zip(dataset_logits.tolist(), dataset_labels.tolist(), dataset_guids.tolist(), dataset_preds.tolist(), dataset_loss.tolist(), dataset_margin.tolist()):
            record = {'guid': guid, 'logits_round_%s'%comm_round: logit, 'gold': label, 'pred': pred, 'device': str(device), 'loss': loss, 'margin': margin}
            dataset_dynamics.append(record)

    return dataset_dynamics

def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

def now():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M")[:-1]

def save_json(file_name, root_dir, content):
    file_path = os.path.join(root_dir, file_name)
    with open(file_path, "w") as out_f:
        json.dump(content, out_f)
    return True

def save_obj(obj, name, pkl_protocal=None):
    with open(name + '.pkl', 'wb') as f:
        # pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(obj, f, pkl_protocal)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.mkdir(dir_path)
        except FileNotFoundError:
            os.makedirs(dir_path)


def make_alg_name(args):
    if args.criterion != "ce":
        alg_name = "FedAvg-RobustLoss"
    else:
        if args.mixup is True:
            alg_name = "FedAvg-Mixup"
        elif args.coteaching is True:
            alg_name = "FedAvg-Coteaching"
        elif args.dynboot is True:
            alg_name = "FedAvg-DynamicBootstrapping"
        elif args.dividemix is True:
            alg_name = "FedAvg-DivideMix"
        else:
            alg_name = "FedAvg"

    return alg_name


def make_exp_name(fed_alg_name="fedavg", args=None) -> str:
    """Make logging name for federated algrithms.

    Args:
        fed_alg_name (str, optional): _description_. Defaults to "fedavg".
        args (_type_, optional): _description_. Defaults to None.

    Returns:
        str: _description_
    """
    if fed_alg_name == "fedavg" or fed_alg_name == 'fedlc':
        noisy_alg_name = None
        arch_name = f"arch={args.model}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        criterion_name = make_criterion_name(args)
        lr_scheduler_name = make_lr_scheduler_name(args)
        if args.mixup is True:
            noisy_alg_name = f"mixup=True-mixup_alpha={args.mixup_alpha:.2f}"
        elif args.coteaching is True:
            noisy_alg_name = f"coteaching=True-coteaching_forget_rate={args.coteaching_forget_rate}-coteaching_num_gradual={args.coteaching_num_gradual}-coteaching_exponent={args.coteaching_exponent}"
        elif args.dynboot is True:
            noisy_alg_name = f"dynboot=True-dynboot_mixup={args.dynboot_mixup}-dynboot_alpha={args.dynboot_alpha:.2f}-dynboot_bootbeta={args.dynboot_bootbeta}-dynboot_reg={args.dynboot_reg:.2f}"

        other_name = f"com_round={args.com_round}-local_epochs={args.epochs}-sample_ratio={args.sample_ratio:.2f}-batch_size={args.batch_size}-seed={args.seed}"

    if noisy_alg_name is None:
        exp_name = "-".join(
            [fed_alg_name, criterion_name, arch_name, opt_name, lr_scheduler_name, other_name]
        )
    else:
        exp_name = "-".join(
            [
                fed_alg_name,
                criterion_name,
                # noisy_alg_name,
                arch_name,
                opt_name,
                lr_scheduler_name,
                other_name,
            ]
        )

    return exp_name


def make_exp_name_centr(alg_name="dividemix", args=None):
    if alg_name == "crossentropy":
        noise_name = f"noise_mode={args.noise_mode}-noise_ratio={args.noise_ratio:.2f}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        other_name = f"num_epochs={args.num_epochs}-batch_size={args.batch_size}-seed={args.seed}"
        exp_name = "-".join([noise_name, opt_name, other_name])

    elif alg_name == "dividemix":
        noise_name = f"noise_mode={args.noise_mode}-noise_ratio={args.noise_ratio:.2f}"
        alg_param = f"p_threshold={args.p_threshold:.2f}-lambda_u={args.lambda_u}-T={args.T:.2f}-alpha={args.alpha:.2f}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        other_name = f"num_epochs={args.num_epochs}-batch_size={args.batch_size}-seed={args.seed}"
        exp_name = "-".join([noise_name, alg_name, alg_param, opt_name, other_name])

    elif alg_name == "coteaching":
        pass
    return exp_name


def make_criterion_name(args):
    criterion_name = f"criterion={args.criterion}"

    if args.criterion == "ce":
        criterion_param = ""
    elif args.criterion == "sce":
        criterion_param = f"sce_alpha={args.sce_alpha:.2f}-sce_beta={args.sce_beta:.2f}"
    elif args.criterion in ["rce", "nce", "nrce"]:
        criterion_param = f"loss_scale={args.loss_scale:.2f}"
    elif args.criterion == "gce":
        criterion_param = f"gce_q={args.gce_q:.2f}"
    elif args.criterion == "ngce":
        criterion_param = f"loss_scale={args.loss_scale:.2f}-gce_q={args.gce_q:.2f}"
    elif args.criterion in ["mae", "nmae"]:
        criterion_param = f"loss_scale={args.loss_scale:.2f}"
    elif args.criterion in ["focal", "nfocal"]:
        if args.focal_alpha is None:
            criterion_param = f"focal_gamma={args.focal_gamma:.2f}-focal_alpha=None"
        else:
            criterion_param = (
                f"focal_gamma={args.focal_gamma:.2f}-focal_alpha={args.focal_alpha:.2f}"
            )
    #==== APL losses ====
    elif args.criterion in ["apl_nce_mae","apl_nce_rce"]:
        criterion_param=f"apl_alpha={args.apl_alpha:.2f}-apl_beta={args.apl_beta:.2f}"
    elif args.criterion in ["apl_nfl_mae","apl_nfl_rce"]:
        criterion_param=f"apl_alpha={args.apl_alpha:.2f}-apl_beta={args.apl_beta:.2f}-apl_focal_gamma={args.focal_gamma:.2f}"
    elif args.criterion in ["apl_ngce_mae","apl_ngce_rce"]:
        criterion_param=f"apl_alpha={args.apl_alpha:.2f}-apl_beta={args.apl_beta:.2f}-apl_gce_q={args.gce_q:.2f}"
    #--------------------
    criterion_name = "-".join([criterion_name, criterion_param])
    return criterion_name

def make_lr_scheduler_name(args):
    if args.lr_scheduler == "none":
        lr_scheduler_name = ""
    else:
        lr_scheduler_name = f"lr_scheduler={args.lr_scheduler}"
        if args.lr_scheduler == "none":
            lr_scheduler_param = ""
        elif args.lr_scheduler == "step":
            lr_scheduler_param = f"step_size={args.step_size}-step_gamma={args.step_gamma}"
        elif args.lr_scheduler == "multistep":
            milestone = [str(i) for i in args.multistep_milestone]
            milestone = "-".join(milestone)
            lr_scheduler_param = f"step_milestone={milestone}-step_gamma={args.step_gamma}"
        elif args.lr_scheduler == "cosine":
            lr_scheduler_param = f""

        lr_scheduler_name = "-".join([lr_scheduler_name, lr_scheduler_param])
    return lr_scheduler_name


def serialize_model(model: torch.nn.Module) -> torch.Tensor:
    parameters = [param.data.view(-1) for param in model.state_dict().values()]
    m_parameters = torch.cat(parameters) # return a new tensor?
    m_parameters = m_parameters.cpu() # deepcopy to cpu?

    return m_parameters


def deserialize_model(
    model: torch.nn.Module, serialized_parameters: torch.Tensor, mode="copy"
):
    current_index = 0  # keep track of where to read from grad_update

    for param in model.state_dict().values():
        numel = param.numel()
        size = param.size()
        if mode == "copy":
            param.copy_(
                serialized_parameters[current_index : current_index + numel].view(size)
            )
        elif mode == "add":
            param.add_(
                serialized_parameters[current_index : current_index + numel].view(size)
            )
        else:
            raise ValueError(
                'Invalid deserialize mode {}, require "copy" or "add" '.format(mode)
            )
        current_index += numel

    assert current_index == serialized_parameters.numel()


def model_to_serilize_dict(model: torch.nn.Module, ) -> OrderedDict:
    current_index = 0  # keep track of where to read from grad_update

    serialized_dict = OrderedDict()
    for name, param in model.state_dict().items():
        numel = param.numel()
        size = param.size()
        serialized_dict[name] = OrderedDict(
            serilized_param=param.data.view(-1),
            numel=numel,
            size=size,
            start_index=current_index,
            end_index=current_index + numel,
        )
        current_index += numel


def result_parser(result_path):
    """_summary_

    Args:
        result_path (str): _description_

    Returns:
        tuple[List[float], List[float], Dict]: _description_
    """
    with open(result_path, "r") as f:
        lines = f.readlines()
    # hist accuracy
    accs = [float(item) for item in lines[1].strip()[5:-1].split(", ")]
    # hist losses
    losses = [float(item) for item in lines[2].strip()[6:-1].split(", ")]
    # hyperparameter setting
    setting_dict = ast.literal_eval(lines[0].strip())
    return accs, losses, setting_dict

def js_div(ideal_distri:torch.Tensor, label_distri:torch.Tensor):
    kl1 = (ideal_distri * (ideal_distri.log() - ((label_distri+ideal_distri)/2).log())).sum()
    kl2 = (label_distri * (label_distri.log() - ((label_distri+ideal_distri)/2).log())).sum()
    js=(kl1+kl2)/2
    return js

def lid_term(X: np.ndarray, batch: np.ndarray, k=20):
    eps = 1e-8
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids