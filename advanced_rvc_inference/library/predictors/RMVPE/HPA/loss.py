import torch
from torch import nn
import torch.nn.functional as F


def smoothl1(inputs, targets, alpha=None):
    loss_f = nn.SmoothL1Loss(reduce=False)
    weight = torch.ones(inputs.shape, dtype=torch.float).to(inputs.device)
    if alpha is not None:
        weight[targets != 0] = float(alpha)
    loss = torch.mean(loss_f(inputs, targets) * weight)
    return loss


def bce(inputs, targets):
    loss_f = nn.BCELoss()
    loss = loss_f(inputs, targets)
    return loss


def FL(inputs, targets, alpha, gamma):
    loss = F.binary_cross_entropy(inputs, targets, reduce=False)
    weight = torch.ones(inputs.shape, dtype=torch.float).to(inputs.device)
    weight[targets == 1] = float(alpha)
    loss_w = F.binary_cross_entropy(inputs, targets, weight=weight, reduce=False)
    pt = torch.exp(-loss)
    weight_gamma = (1 - pt) ** gamma
    F_loss = torch.mean(weight_gamma * loss_w)
    return F_loss


def mae(input, target):
    l1_loss = nn.L1Loss(reduce=False)
    loss = l1_loss(input, target)
    return torch.mean(loss)


def mse(input, target):
    l2_loss = nn.MSELoss(reduce=False)
    loss = l2_loss(input, target)
    return torch.mean(loss)


def ce(input, target):
    ce = nn.CrossEntropyLoss(reduce=False)
    loss = ce(input, target)
    return torch.mean(loss)
