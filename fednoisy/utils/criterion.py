import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def linear_rampup(lambda_u, current, warm_up, rampup_length=16):
    """DivideMix"""
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


class NegEntropy(object):
    """DivideMix"""

    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


class DivideMixSemiLoss(object):
    """DivideMix"""

    def __call__(
        self, outputs_x, targets_x, outputs_u, targets_u, lambda_u, epoch, warm_up
    ):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(lambda_u, epoch, warm_up)


def loss_coteaching(outputs1, outputs2, noisy_label, forget_rate, noise_or_not):
    with torch.no_grad():
        loss1 = F.cross_entropy(outputs1, noisy_label, reduce=False)
        idx1_sorted = torch.argsort(loss1.data)
        loss1_sorted = loss1[idx1_sorted]

        loss2 = F.cross_entropy(outputs2, noisy_label, reduce=False)
        idx2_sorted = torch.argsort(loss2.data)
        loss2_sorted = loss1[idx2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss1_sorted))

        pure_ratio1 = noise_or_not[idx1_sorted[:num_remember]].sum() / num_remember
        pure_ratio2 = noise_or_not[idx2_sorted[:num_remember]].sum() / num_remember

        idx1_update = idx1_sorted[:num_remember]
        idx2_update = idx2_sorted[:num_remember]
    # exchange
    loss1_update = F.cross_entropy(
        outputs1[idx2_update], noisy_label[idx2_update], reduction="mean"
    )
    loss2_update = F.cross_entropy(
        outputs2[idx1_update], noisy_label[idx1_update], reduction="mean"
    )
    return loss1_update, loss2_update, pure_ratio1, pure_ratio2


def loss_coteaching_guessing(outputs1, outputs2, noisy_label, forget_rate, noise_or_not):
    with torch.no_grad():
        loss1 = F.cross_entropy(outputs1, noisy_label, reduce=False)
        idx1_sorted = torch.argsort(loss1.data)
        loss1_sorted = loss1[idx1_sorted]

        bsz = outputs1.size(0)

        loss2 = F.cross_entropy(outputs2, noisy_label, reduce=False)
        idx2_sorted = torch.argsort(loss2.data)
        loss2_sorted = loss1[idx2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss1_sorted))

        pure_ratio1 = noise_or_not[idx1_sorted[:num_remember]].sum() / num_remember
        pure_ratio2 = noise_or_not[idx2_sorted[:num_remember]].sum() / num_remember

        idx1_clean, idx1_noisy = idx1_sorted[:num_remember], idx1_sorted[num_remember:]
        idx2_clean, idx2_noisy = idx2_sorted[:num_remember], idx2_sorted[num_remember:]


    upx1=(torch.softmax(outputs2[idx2_noisy].detach(),dim=-1)+torch.softmax(outputs1[idx2_noisy].detach(),dim=-1))/2
    upx2=(torch.softmax(outputs1[idx1_noisy].detach(),dim=-1)+torch.softmax(outputs2[idx1_noisy].detach(),dim=-1))/2
    upx1=upx1**(1/0.5)
    upx2=upx2**(1/0.5)

    # exchange
    loss1_clean = F.cross_entropy(
        outputs1[idx2_clean], noisy_label[idx2_clean], reduction="mean"
    )
    loss2_clean = F.cross_entropy(
        outputs2[idx1_clean], noisy_label[idx1_clean], reduction="mean"
    )

    # loss1_noisy = torch.mean(torch.sum(-torch.softmax(outputs2[idx2_noisy].detach(),dim=-1) * torch.log_softmax(outputs1[idx2_noisy],dim=-1),dim=-1))
    # loss2_noisy = torch.mean(torch.sum(-torch.softmax(outputs1[idx1_noisy].detach(),dim=-1) * torch.log_softmax(outputs2[idx1_noisy],dim=-1),dim=-1))

    loss1_noisy = torch.mean(torch.sum(-upx1 * torch.log_softmax(outputs1[idx2_noisy],dim=-1),dim=-1))
    loss2_noisy = torch.mean(torch.sum(-upx2 * torch.log_softmax(outputs2[idx1_noisy],dim=-1),dim=-1))

    return num_remember/bsz * loss1_clean+ (bsz-num_remember)/bsz * loss1_noisy, num_remember/bsz * loss2_clean+ (bsz-num_remember)/bsz * loss2_noisy, pure_ratio1, pure_ratio2


# logits calibration 
class FedLCLoss(nn.Module):
    def __init__(self, tau) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, logit, y, label_distrib: torch.Tensor):
        cal_logit = torch.exp(
                logit
                - (
                    self.tau
                    * torch.pow(label_distrib, -1 / 4)
                    .unsqueeze(0)
                    .expand((logit.shape[0], -1))
                )
            )
        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        return loss.sum() / logit.shape[0]

    # def load_dataset(self):
    #     super().load_dataset()
    #     label_counter = Counter(self.dataset.targets[self.trainset.indices].tolist())
    #     self.label_distrib.zero_()
    #     for cls, count in label_counter.items():
    #         self.label_distrib[cls] = max(1e-8, count)

"""Code for robust loss functions is from https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py"""


#### TODO: modified version with no device argument ####
def get_robust_loss(num_classes, args):
    if args.criterion == "ce":
        return nn.CrossEntropyLoss()
    elif args.criterion == "sce":
        return SCELoss(args.sce_alpha, args.sce_beta, num_classes)
    elif args.criterion == "rce":
        return ReverseCrossEntropy(num_classes, args.loss_scale)
    elif args.criterion == "nrce":
        return NormalizedReverseCrossEntropy(num_classes, args.loss_scale)
    elif args.criterion == "nce":
        return NormalizedCrossEntropy(num_classes, args.loss_scale)
    elif args.criterion == "gce":
        return GeneralizedCrossEntropy(num_classes, args.gce_q, args.trunc_k if hasattr(args, "trunc_k") else 1e-7)
    elif args.criterion == "ngce":
        return NormalizedGeneralizedCrossEntropy(
            num_classes, args.loss_scale, args.gce_q
        )
    elif args.criterion == "mae":
        return MeanAbsoluteError(num_classes, args.loss_scale)
    elif args.criterion == "nmae":
        return NormalizedMeanAbsoluteError(num_classes, args.loss_scale)
    elif args.criterion == "focal":
        return FocalLoss(args.focal_gamma, args.focal_alpha)
    elif args.criterion == "nfocal":
        return NormalizedFocalLoss(
            args.loss_scale, args.focal_gamma, num_classes, args.focal_alpha
        )
    #==== APL losses ====
    elif args.criterion == "apl_nce_mae":
        return NCEandMAE(alpha=args.apl_alpha,beta=args.apl_beta,num_classes=num_classes)
    elif args.criterion == "apl_nce_rce":
        return NCEandRCE(alpha=args.apl_alpha,beta=args.apl_beta,num_classes=num_classes)
    elif args.criterion == "apl_nfl_mae":
        return NFLandMAE(alpha=args.apl_alpha,beta=args.apl_beta,num_classes=num_classes,gamma=args.focal_gamma)
    elif args.criterion == "apl_nfl_rce":
        return NFLandRCE(alpha=args.apl_alpha,beta=args.apl_beta,num_classes=num_classes,gamma=args.focal_gamma)
    elif args.criterion == "apl_ngce_mae":
        return NGCEandMAE(alpha=args.apl_alpha,beta=args.apl_beta,num_classes=num_classes,q=args.gce_q)
    elif args.criterion == "apl_ngce_rce":
        return NGCEandRCE(alpha=args.apl_alpha,beta=args.apl_beta,num_classes=num_classes,q=args.gce_q)
    #--------------------
    #==== Other losses ====
    elif args.criterion == "cecr":
        return CECRLoss(args.cecr_beta)
    elif args.criterion == "softce":
        return CELoss()
    #--------------------
    else:
        raise ValueError(
            f"args.criterion='{args.criterion}' is not supported. Only support 'ce', 'sce', 'rce', 'nrce', 'nce', 'gce', 'ngce', 'mae', 'nmae', 'focal', 'nfocal'."
        )


class SCELoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class ReverseCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        return self.scale * rce.mean()


class NormalizedReverseCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        normalizor = 1 / 4 * (self.num_classes - 1)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        return self.scale * normalizor * rce.mean()


class NormalizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * nce.mean()


class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, q=0.7, trunc_k=1e-7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        assert 0.0 < trunc_k < 1.0
        self.trunc_k = trunc_k

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=self.trunc_k, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()


class NormalizedGeneralizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0, q=0.7):
        super(NormalizedGeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        numerators = 1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()


class MeanAbsoluteError(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        mae = 1.0 - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        return self.scale * mae.mean()


class NormalizedMeanAbsoluteError(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedMeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        normalizor = 1 / (2 * (self.num_classes - 1))
        mae = 1.0 - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * normalizor * mae.mean()


class NCEandRCE(nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)


class NCEandMAE(nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)


class GCEandMAE(nn.Module):
    def __init__(self, alpha, beta, num_classes=10, q=0.7):
        super(GCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.mae(pred, labels)


class GCEandRCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.rce(pred, labels)


class GCEandNCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.nce = NormalizedCrossEntropy(num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.nce(pred, labels)


class NGCEandNCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.nce(pred, labels)


class NGCEandMAE(nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.mae(pred, labels)


class NGCEandRCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.rce(pred, labels)


class MAEandRCE(nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(MAEandRCE, self).__init__()
        self.num_classes = num_classes
        self.mae = MeanAbsoluteError(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.mae(pred, labels) + self.rce(pred, labels)


class NLNL(nn.Module):
    def __init__(self, train_loader, num_classes, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.0
        if not hasattr(train_loader.dataset, "targets"):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (
                    torch.from_numpy(np.array(train_loader.dataset.targets)) == i
                ).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (
            labels.unsqueeze(-1).repeat(1, self.ln_neg)
            + torch.LongTensor(len(labels), self.ln_neg).random_(1, self.num_classes)
        ) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes - 1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(
            labels
        ) * self.ln_neg

        s_neg = torch.log(torch.clamp(1.0 - F.softmax(pred, 1), min=1e-5, max=1.0))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size())
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(
            s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)
        ) * float((labels_neg >= 0).sum())
        loss = (loss + loss_neg) / (
            float((labels >= 0).sum()) + float((labels_neg[:, 0] >= 0).sum())
        )
        return loss


class FocalLoss(nn.Module):
    """
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class NormalizedFocalLoss(nn.Module):
    def __init__(
        self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True
    ):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class NFLandNCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandNCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.nce(pred, labels)


class NFLandMAE(nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


class NFLandRCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


class DMILoss(nn.Module):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = y_onehot @ outputs
        return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)


def mixup_criterion(criterion, pred, y_a, y_b, lmbd):
    """We use cross-entropy loss as criterion here, so it's same as using criterion(pred, lmbd*y_a + (1-lmbd)*y_b)

    Args:
        criterion (_type_): Default as cross-entropy
        pred (_type_): _description_
        y_a (_type_): _description_
        y_b (_type_): _description_
        lmbd (_type_): _description_

    Returns:
        _type_: _description_
    """
    return lmbd * criterion(pred, y_a) + (1 - lmbd) * criterion(pred, y_b)


def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
    

def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse', 'kl']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    elif name == 'kl':
        loss = F.kl_div(F.log_softmax(logits / 0.5, dim=-1), F.softmax(targets / 0.5, dim=-1), reduction='none')
        loss = torch.sum(loss * (1.0 - mask).unsqueeze(dim=-1).repeat(1, torch.softmax(logits, dim=-1).shape[1]), dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None and name != 'kl':
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()
    

class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='mean'):
        return ce_loss(logits, targets, reduction)
    

class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits, targets, name, mask)


class CECRLoss(nn.Module): # refer to cores^2
    def __init__(self, beta):
        super(CECRLoss, self).__init__()
        self.beta = beta

    def forward(self, logits, labels, noise_prior=None):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        # negative_log_logits = -F.log_softmax(logits, dim=-1)
        negative_log_logits = -torch.log(F.softmax(logits,dim=-1)+1e-8) # numerical stability, but why?
        if noise_prior is None:
            cr_penalty = torch.mean(negative_log_logits, dim=-1) # random guess as penalty
        else:
            if not isinstance(noise_prior, torch.Tensor):
                noise_prior = torch.tensor(noise_prior.astype('float32')).to(negative_log_logits.device).unsqueeze(0)
            cr_penalty = torch.sum(negative_log_logits * noise_prior, dim=-1) # broadcasting noise_prior

        return torch.mean(ce_loss - cr_penalty * self.beta)