import torch
import argparse
import sys
import os
from typing import Dict, Tuple, List, Optional

from fednoisy.data.NLLData import functional as nllF


def read_fednll_args():
    parser = argparse.ArgumentParser(description="Federated Noisy Labels Preparation")

    parser.add_argument("--exp_name", type=str, default="fedap_cs", help="Experiment name.")

    # ==== Standalone args ====
    parser.add_argument(
        "--num_clients",
        default=10,
        type=int,
        help="Number for clients.",
    )

    # ==== Scale args ====
    parser.add_argument("--ip", default="localhost", type=str,)
    parser.add_argument("--port", default=3002, type=str,)
    parser.add_argument("--world_size", default=2, type=int,help="World size, including the server.")
    # parser.add_argument("--rank", default=0, type=int,)
    parser.add_argument("--ethernet", type=str, default=None)
    # parser.add_argument("--n_gpu", type=int, default=1, help="Number of used GPU.")

    parser.add_argument(
        "--num_clients_per_gpu",
        default=10,
        type=int,
        help="Number for clients for each GPU.",
    )

    # ----FL args----
    parser.add_argument("--com_round", type=int, default=3)
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet18",
        help="Currently only support 'Cifar10Net', 'SimpleCNN',  'LeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'ToyModel', 'ResNet18', 'PreResNet18', 'ResNet20', 'WRN28_10', 'WRN40_2' and 'ResNet34'.",
    )
    parser.add_argument("--sample_ratio", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)

    # ----LR Scheduler args----
    parser.add_argument("--lr_scheduler",type=str,default="none")
    parser.add_argument("--step_size",type=int, default=20)
    parser.add_argument("--step_gamma",type=float, default=0.1)
    parser.add_argument("--multistep_milestone",nargs="+",help="step milestone for multistep lr scheduler")
    
    # ----Wandb args----
    parser.add_argument("--use_wandb",action="store_true")
    parser.add_argument("--wandb_project_name",type=str,default="fednoisy")
    parser.add_argument("--wandb_group",type=str,default="fednoisy")
    parser.add_argument("--wandb_tags",nargs="+",type=str,default=None)
    parser.add_argument("--wandb_job_type",type=str,default=None)

    # ----FedProx args----
    parser.add_argument(
        "--use_fedprox", action="store_true", help="Whether to use FedProx."
    )
    parser.add_argument(
        "--fedprox_mu", type=str, choices=["constant", "adaptive"], default="constant", help="The mu scheduler for FedProx."
    )

    # ----FedRobust args----
    parser.add_argument(
        "--warmup_round", type=int, default=0, help="The warmup round."
    )
    parser.add_argument(
        "--use_cs", action="store_true", help="Whether to use CS model."
    )
    parser.add_argument(
        "--cs_metric", type=str, help="Which metric is used for Centralized Sieving.", choices=["loss", "loss_mean", "loss_soft_mean", "sc", "scl"], default="loss"
    )
    parser.add_argument(
        "--metric_model", type=str, choices=["global", "local"], default="local", help="Which model to use for metric evluation."
    )
    parser.add_argument(
        "--loss", choices=['mask_out_loss', "naive_pseudo_label_loss"], help="Which loss to use for cs."
    )
    parser.add_argument(
        "--use_cs_semi", action="store_true", help="Whether to use CS Semi model."
    )
    
    # todo
    parser.add_argument(
        "--metric_ema_gamma", type=float, default=0.7, help="The gamma for sample ema metric."
    )
    
    parser.add_argument(
        "--local_ema", action="store_true", help="Whether use local ema model for metric evalutaion."
    )
    parser.add_argument(
        "--local_ema_beta", type=float, help="The beta for local ema model.", default=0.99
    )
    parser.add_argument(
        "--global_ema", action="store_true", help="Whether use global ema model."
    )
    parser.add_argument(
        "--global_ema_beta", type=float, help="The beta for global ema model.", default=0.99
    )
    
    parser.add_argument(
        "--local_ema_plus_global", action="store_true", help="Whether use global model to average local ema model."
    )
    parser.add_argument(
        "--local_ema_plus_global_decay", type=float, help="Decay for global -> local EMA.", default=0.9
    )
    parser.add_argument(
        "--grad_clip", action="store_true", help="Whether use grad clip for local update."
    )
    parser.add_argument(
        "--clip_grad_norm", type=float, help="Grad Norm for grad clipping.", default=10.0
    )
    
    # ----KL Distillation options----
    parser.add_argument(
        "--use_kl_distill", action="store_true",
    )
    parser.add_argument(
        "--kl_distill_temperature", type=float, default=0.5, help="The temperature for KL distillation."
    )
    parser.add_argument(
        "--kl_teacher_model", type=str, help="Which model is used for KL distillation.", choices=["local_ema", "global_ema", "global",], default="global"
    )
    parser.add_argument(
        "--lambda_kl", type=float, default=1.0, help="The weight for KL distillation loss."
    )

    # ----ELR options----
    parser.add_argument(
        "--use_elr", action="store_true",
    )
    parser.add_argument(
        "--elr_teacher_model", type=str, help="Which model is used for ELR penalty loss.", choices=["local_ema", "global_ema", "global",], default="global"
    )
    parser.add_argument(
        "--lambda_elr", type=float, default=1.0, help="The weight for ELR penalty loss."
    )

    # ----Locla Mixup options----
    parser.add_argument(
        "--use_local_mixup", action="store_true", help="Whether to use mixup for local model training."
    )
    parser.add_argument(
        "--mixup_alpha", type=float, default=0.5, help="The hyper parameter for mixup beta distribution."
    )

    # ----Other options----
    parser.add_argument(
        "--cecr_beta", type=float, default=2.0, help="The hyper parameter for CECR Loss"
    )
    parser.add_argument(
        "--cache_data_loader", action="store_true", help="Whether to cache the data loader."
    )
    

    # ==== FedNLL data args ====
    parser.add_argument(
        "--centralized",
        default=False,
        help="Centralized setting or federated setting. True for centralized "
        "setting, while False for federated setting.",
    )
    parser.add_argument(
        "--preload", action="store_true", help="Whether to preload dataset into memory."
    )
    # ----Federated Partition----
    parser.add_argument(
        "--partition",
        default="iid",
        type=str,
        choices=["iid", "noniid-#label", "noniid-labeldir", "noniid-quantity"],
        help="Data partition scheme for federated setting.",
    )

    parser.add_argument(
        "--dir_alpha",
        default=0.1,
        type=float,
        help="Parameter for Dirichlet distribution.",
    )
    parser.add_argument(
        "--major_classes_num",
        default=2,
        type=int,
        help="Major class number for 'noniid-#label' partition.",
    )
    parser.add_argument(
        "--min_require_size",
        default=10,
        type=int,
        help="Minimum sample size for each client.",
    )

    # ----Noise setting options----
    parser.add_argument(
        "--noise_mode",
        default=None,
        type=str,
        choices=["clean", "sym", "asym", "real"],
        help="Noise type for centralized setting: 'sym' for symmetric noise; "
        "'asym' for asymmetric noise; 'real' for real-world noise. ",
    )
    parser.add_argument(
        "--globalize",
        action="store_true",
        help="Federated noisy label setting, globalized noise or localized noise.",
    )

    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="Noise ratio for symmetric noise or asymmetric noise.",
    )
    parser.add_argument(
        "--min_noise_ratio",
        default=0.0,
        type=float,
        help="Minimum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )
    parser.add_argument(
        "--max_noise_ratio",
        default=1.0,
        type=float,
        help="Maximum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )
    parser.add_argument(
        "--num_samples",
        default=32 * 2 * 1000,
        type=int,
        help="Number of samples used for Clothing1M training. Defaults as 64000.",
    )

    # ----Robust Loss Function options----
    parser.add_argument(
        "--criterion", type=str, default="ce"
    )  # for robust loss function
    parser.add_argument(
        "--sce_alpha",
        type=float,
        default=0.1,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--sce_beta",
        type=float,
        default=1.0,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=1.0,
        help="scale parameter for loss, for example, scale * RCE, scale * NCE, scale * normalizer * RCE.",
    )
    parser.add_argument(
        "--gce_q",
        type=float,
        default=0.7,
        help="q parameter for Generalized-Cross-Entropy, Normalized-Generalized-Cross-Entropy.",
    )
    parser.add_argument(
        "--gce_trunc_k",
        type=float,
        default=1e-7,
        help="truncate k parameter for Generalized-Cross-Entropy.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=None,
        help="alpha parameter for Focal loss and Normalzied Focal loss.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="gamma parameter for Focal loss and Normalzied Focal loss.",
    )

    #==== APL losses ====
    parser.add_argument(
        "--apl_alpha",
        type=float,
        default=1.0,
        help="alpha parameter for APL losses"
    )
    parser.add_argument(
        "--apl_beta",
        type=float,
        default=1.0,
        help="beta parameter for APL losses"
    )
    #--------------------

    # ----Path options----
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "svhn", "clothing1m", "webvision"],
        help="Dataset for experiment. Current support: ['mnist', 'cifar10', "
        "'cifar100', 'svhn', 'clothing1m', 'webvision']",
    )
    parser.add_argument(
        "--data_dir",
        default="../noisy_label_data",
        type=str,
        help="Directory to save the dataset with noisy labels.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="../checkponit/",
        help="Checkpoint path for log files and report files.",
    )

    # ----Miscs options----
    parser.add_argument(
        "--save_best", action="store_true", help="Whether to save the best model."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")

    args = parser.parse_args()
    return args
