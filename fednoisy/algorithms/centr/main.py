from json import load
import os
import sys
import argparse
import random
from copy import deepcopy
from typing import Any

from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, auc, precision_score, f1_score

sys.path.append(os.getcwd())
from fedlab.utils.logger import Logger
from fednoisy.data.NLLData.CentrNLL import CentrNLLCIFAR10, CentrNLLCIFAR100
from fednoisy.models.build_model import build_model
from fednoisy.utils.wandb_logger import WandbLogger

from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    TRANSITION_MATRIX,
    NORM_VALUES,
    TEST_TRANSFORM,
    TRAIN_TRANSFORM,
)

def now():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M")[:-1]

def read_args() -> Any:
    parser = argparse.ArgumentParser(description="Centralized NLL")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10","cifar100"],
        help="dataset used for training",
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="sym",
        choices=["sym", "asym", "clean"],
        help="noise mode for centralized CIFAR10",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.0,
        help="noise ratio for centralized CIFAR10",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="data",
        help="root directory for dataset",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="../centrNLLdata/cifar10",
        help="output directory for dataset",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ResNet18",
        choices=["VGG16","ResNet18"],
        help="model used for training",
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="number of local epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        help="test batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        help="learning rate scheduler",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="weight decay",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for dataloader",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="log interval",
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="use wandb to log",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="centralized_nll",
        help="wandb project name",
    )
    parser.add_argument("--wandb_group",type=str,default="fednoisy")
    parser.add_argument("--wandb_tags",nargs="+",type=str,default=None)
    parser.add_argument("--wandb_job_type",type=str,default=None)
    parser.add_argument("--wandb_run_name",type=str,default=None)
    return parser.parse_args()

def get_dataset(args):
    if args.dataset == "cifar10":
        dataset = CentrNLLCIFAR10(root_dir=args.raw_data_dir, noise_mode=args.noise_mode, noise_ratio=args.noise_ratio, out_dir=args.out_dir)
        dataset.create_nll_scene(seed=args.seed)
        train_dataset = CentralNLLDataset(args, dataset, train=True)
        test_dataset = CentralNLLDataset(args, dataset, train=False)
    elif args.dataset == "cifar100":
        dataset = CentrNLLCIFAR100(root_dir=args.raw_data_dir, noise_mode=args.noise_mode, noise_ratio=args.noise_ratio, out_dir=args.out_dir)
        dataset.create_nll_scene(seed=args.seed)
        train_dataset = CentralNLLDataset(args, dataset, train=True)
        test_dataset = CentralNLLDataset(args, dataset, train=False)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    return train_dataset, test_dataset

def get_data_loader(args, dataset, train=True):
    if train:
        batch_size = args.batch_size
        shuffle = True
    else:
        batch_size = args.test_batch_size
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

def get_model(args):
    model = build_model(args.model, CLASS_NUM[args.dataset], dataset=args.dataset)
    return model

def get_wandb_logger(args):
    wandb_logger = WandbLogger(
        args.wandb_project_name, 
        exp_cfg=vars(args),
        tags=args.wandb_tags,
        job_type=args.wandb_job_type,
        name=args.time_stamp + '-' + args.wandb_run_name,
    )
    return wandb_logger

def evaluate(args, model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data, target = data["img"].to(device), data["label"].to(device)  # send data to device
            output = model(data)  # forward
            test_loss += F.cross_entropy(output, target, reduction="sum").item()  # compute loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # count correct predictions
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    logger.info(
        "Test set: Avg loss: {:.4f}, Acc: {}/{} ({:.2f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * accuracy,
        )
    )
    return accuracy, test_loss

def validate(args, model, device, data_loader, logger):
    model.eval()
    overall_clean_loss = 0
    overall_noisy_loss = 0
    clean_set_correct = 0
    noisy_set_correct = 0
    clean_correct = clean_cnt = 0
    noisy_set_overfitted = noisy_cnt = 0
    with torch.no_grad():
        for data in data_loader:
            data, target, noisy_target = data["img"].to(device), data["label"].to(device), data["noisy_label"].to(device)  # send data to device
            output = model(data)  # forward
            clean_mask = target == noisy_target
            overall_clean_loss += F.cross_entropy(output, target, reduction="sum").item()  # compute loss
            overall_noisy_loss += F.cross_entropy(output, noisy_target, reduction="sum").item()  # compute loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            noisy_set_correct += pred[~clean_mask].eq(target[~clean_mask].view_as(pred[~clean_mask])).sum().item()  # count correct predictions
            clean_set_correct += pred[clean_mask].eq(target[clean_mask].view_as(pred[clean_mask])).sum().item()  # count correct predictions
            noisy_set_overfitted += pred[~clean_mask].eq(noisy_target[~clean_mask].view_as(pred[~clean_mask])).sum().item()
            clean_cnt += torch.sum(clean_mask).item()
            noisy_cnt += torch.sum(~clean_mask).item()
    overall_clean_loss /= len(data_loader.dataset)
    overall_noisy_loss /= len(data_loader.dataset)
    overall_clean_accuracy = (clean_set_correct+noisy_set_correct) / len(data_loader.dataset)
    overall_noisy_accuracy = (noisy_set_overfitted+clean_set_correct) / len(data_loader.dataset)
    clean_set_acc = clean_set_correct / len(data_loader.dataset)
    noisy_set_acc = noisy_set_correct / len(data_loader.dataset)
    noisy_fit_acc = noisy_set_overfitted / len(data_loader.dataset)
    logger.info(
        f"Train set: Avg loss: {overall_clean_loss:.4f}, Avg N loss: {overall_noisy_loss:.4f}, "
        f"C set acc: {clean_set_correct}/{clean_cnt}({clean_set_correct/len(data_loader.dataset)*100:.2f}%), "
        f"N set acc: {noisy_set_correct}/{noisy_cnt}({noisy_set_correct/len(data_loader.dataset)*100:.2f}%), "
        f"N set fit: {noisy_set_overfitted}/{noisy_cnt}({noisy_set_overfitted/noisy_cnt*100:.2f}%), {noisy_set_overfitted}/{len(data_loader.dataset)}({noisy_set_overfitted/len(data_loader.dataset)*100:.2f}%), "
        f"OC acc: {clean_set_correct+noisy_set_correct}/{len(data_loader.dataset)}({overall_clean_accuracy*100:.2f}%), "
        f"ON acc: {noisy_set_overfitted+clean_set_correct}/{len(data_loader.dataset)}({overall_noisy_accuracy*100:.2f}%)"
    )
    return {
        "overall_noisy_accuracy": overall_noisy_accuracy*100,
        "overall_noisy_test_loss": overall_noisy_loss,

        "overall_clean_accuracy": overall_clean_accuracy*100,
        "overall_clean_test_loss": overall_clean_loss,

        "clean_accuracy": clean_set_acc*100,
        "noisy_accuracy": noisy_set_acc*100,
        "noisy_fit_accuracy": noisy_fit_acc*100,
    }

def train(args, model, device, train_loader, optimizer, scheduler, epoch, logger):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data, target, noisy_target = data["img"].to(device), data["label"].to(device), data["noisy_label"].to(device)  # send data to device
        optimizer.zero_grad()  # zero out gradients
        output = model(data)  # forward
        loss = F.cross_entropy(output, noisy_target)  # compute loss
        loss.backward()  # backward
        optimizer.step()  # optimize
        if batch_idx % args.log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

def eval_train(args, model, device, train_loader, logger):
    model.eval()
    losses = []
    y_true = []
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            data, target, noisy_target = data["img"].to(device), data["label"].to(device), data["noisy_label"].to(device)  # send data to device
            output = model(data)  # forward
            loss = F.cross_entropy(output, noisy_target, reduction="none")  # compute loss
            losses.append(loss.cpu().numpy())
            y_true.append((target == noisy_target).cpu().numpy())
    input_loss = np.concatenate(losses)
    y_true = np.concatenate(y_true)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss.reshape((-1, 1)))
    prob = gmm.predict_proba(input_loss.reshape((-1, 1)))
    prob = prob[:,gmm.means_.argmin()] #属于小loss的概率是多少 获得的是真实无噪声样本的概率是多少 该样本无噪声的概率是多少
    y_pred = prob > 0.5
    gmm_metrics = {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    logger.info(
        f"gmm f1: {gmm_metrics['f1']:.4f}, gmm precision: {gmm_metrics['precision']:.4f}, gmm recall: {gmm_metrics['recall']:.4f}, gmm accuracy: {gmm_metrics['accuracy']:.4f}"
    )
    return gmm_metrics

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = read_args()
    args.time_stamp = now()
    seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger(
        log_name="Centralized NLL",
        log_file=f"fednoisy/algorithms/centr/logs/{args.dataset}_{args.noise_mode}_{args.noise_ratio}.log",
    )
    if args.use_wandb:
        wandb_logger = get_wandb_logger(args)
    logger.info(args)
    train_dataset, test_dataset = get_dataset(args)
    train_loader = get_data_loader(args, train_dataset, train=True)
    val_loader = get_data_loader(args, train_dataset, train=False)
    test_loader = get_data_loader(args, test_dataset, train=False)
    model = get_model(args)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    # )
    if args.lr_scheduler is not None:
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        else:
            raise ValueError(f"lr_scheduler {args.lr_scheduler} not supported.")
    else:
        scheduler = None
    for epoch in range(1, args.num_epochs + 1):
        train(args, model, device, train_loader, optimizer, scheduler, epoch, logger)
        if scheduler is not None:
            scheduler.step()
        train_metrics = validate(args, model, device, val_loader, logger)
        test_acc,test_loss = evaluate(args, model, device, test_loader, logger)
        gmm_metrics = eval_train(args, model, device, train_loader, logger)
        if args.use_wandb:
            wandb_logger.run.log(
                {
                    "train/loss": train_metrics["overall_noisy_test_loss"],
                    "train/acc": train_metrics["overall_noisy_accuracy"],
                    "train/clean_acc": train_metrics["clean_accuracy"],
                    "train/noisy_acc": train_metrics["noisy_accuracy"],
                    "train/noisy_fit_acc": train_metrics["noisy_fit_accuracy"],

                    "test/test_loss": test_loss,
                    "test/test_acc": test_acc,
                    
                    "other/epoch": epoch,
                    "other/lr": optimizer.param_groups[0]["lr"],

                    "gmm/f1": gmm_metrics["f1"],
                    "gmm/precision": gmm_metrics["precision"],
                    "gmm/recall": gmm_metrics["recall"],
                    "gmm/accuracy": gmm_metrics["accuracy"],
                }
            )

class CentralNLLDataset(Dataset):
    def __init__(self, args, dataset,train=True) -> None:
        self.args = args
        self.dataset = dataset
        self.train = train
        if self.train:
            self.data = self.dataset.train_data
            self.labels = self.dataset.train_labels
            self.noisy_labels = self.dataset.noisy_labels
            self.guids = self.dataset.train_guids
            self.transform = TRAIN_TRANSFORM[self.args.dataset]
        else:
            self.data = self.dataset.test_data
            self.labels = self.dataset.test_labels
            self.guids = self.dataset.test_guids
            self.transform = TEST_TRANSFORM[self.args.dataset]
        self.folder_data = False
        if self.folder_data is False:
            shape = self.data[0].shape
            ndim = len(shape)
            if ndim == 2:
                self.mode = "L"  # MNIST
            elif ndim == 3 and shape[-1] == 3:
                self.mode = "RGB"  # CIFAR10 & CIFAR100 & SVHN

    def __getitem__(self, index: Any) -> Any:
        if self.folder_data is False:
            # CIFAR10 & CIFAR100 & SVHN & MNIST
            img, label = self.data[index], self.labels[index]
            img = Image.fromarray(img, mode=self.mode)
        else:
            # clothing1m data
            img_path, label = self.data[index], self.labels[index]
            img = Image.open(img_path).convert("RGB")
        guid = self.guids[index]
        img = self.transform(img)

        if self.train:
            noisy_label = self.noisy_labels[index]
            return {
                    "img": img,
                    "label": label, 
                    "noisy_label": noisy_label,
                    "guid": guid,
                }
        else:
            return {
                "img": img, 
                "label": label,
                "guid": guid,
            }

    
    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    main()