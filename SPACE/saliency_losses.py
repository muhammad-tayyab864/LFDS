"""
Source:
https://github.com/rdroste/unisal/blob/master/unisal/utils.py
"""

import torch
import torch.nn.functional as F

_eps = 1e-7

def nss(pred, fixations):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    fixations = fixations.reshape(new_size)

    pred_normed = (pred - pred.mean(-1, True)) / (pred.std(-1, keepdim=True) + _eps)
    results = []
    for this_pred_normed, mask in zip(torch.unbind(pred_normed, 0),
                                      torch.unbind(fixations, 0)):
        if mask.sum() == 0:
            print("No fixations.")
            results.append(torch.ones([]).float().to(fixations.device))
            continue
        nss_ = torch.masked_select(this_pred_normed, mask)
        nss_ = nss_.mean(-1)
        results.append(nss_)
    results = torch.stack(results)
    results = results.reshape(size[:2])
    return results


def corr_coeff(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)

    cc = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        xm, ym = x - x.mean(), y - y.mean()
        r_num = torch.mean(xm * ym)
        r_den = torch.sqrt(
            torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
        r = r_num / (r_den + _eps)
        cc.append(r)

    cc = torch.stack(cc)
    cc = cc.reshape(size[:2])
    return cc  # 1 - torch.square(r)

def softmax(x):
    x_size = x.size()
    x = x.view(x.size(0), -1)
    x = F.softmax(x, dim=1)
    return x.view(x_size)

def log_softmax(x):
    x_size = x.size()
    x = x.view(x.size(0), -1)
    x = F.log_softmax(x, dim=1)
    return x.view(x_size)


def kld_loss(pred, target):
    loss = F.kl_div(pred, target, reduction='none')
    loss = loss.sum(-1).sum(-1).sum(-1)
    return loss