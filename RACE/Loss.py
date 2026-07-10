import torch
import torch.nn as nn
import piq
import torch.nn.functional as F

from kornia.filters.filter import filter2d
from kornia.filters.kernels import get_gaussian_kernel2d

def ssim_loss(x, y):
    # SSIM
    x = torch.clamp(x,0,1)
    y = torch.clamp(y,0,1)
    ssimloss = piq.SSIMLoss()
    loss = ssimloss(x, y)

    return loss * 2

def r_loss(out, data, R):
    R = torch.tensor(R).cuda()
    # Power
    power_in  = torch.mean(torch.pow(data,2.2))
    power_out = torch.mean(torch.pow(out,2.2))

    R_score = 1 - (power_out / power_in)
    loss = torch.pow(R - R_score, 2).mean()

    return loss * 10

# TV Loss ddm
def TV_loss(out, data):
    tvloss = piq.TVLoss(norm_type='l2')
    ddm = torch.clamp(data - out, 0, 1)
    loss = tvloss(ddm)

    return loss * 0.01

def contrast_loss(out,data,r):
    out = out.cuda()
    data = data.cuda()
    r = torch.tensor(r).cuda()
    loss = 0.25 * (2 * contrast_loss_G(out,data,r).mean() + contrast_loss_L(out,data,r).mean())

    return loss

def contrast_loss_G(output, target, R, log_ratio=True):
    """
    Global contrast loss.
    """
    w_std = F.relu(1. - 2 * R)
    _c0 = torch.abs(target.mean(dim=(2,3)) - output.mean(dim=(2,3)))
    if log_ratio:
        _c1 = -1 * (1. - w_std) * torch.log((output.var(dim=(2,3)) + 7e-2) / (target.var(dim=(2,3)) + 7e-2))
    else:
        _c1 = -1 * (1. - w_std) * (output.var(dim=(2,3)) / (target.var(dim=(2,3)) + 1e-1))
    _c2 = w_std * torch.abs(target.var(dim=(2,3)) - output.var(dim=(2,3)))
    _c_loss = _c0 + _c1 + _c2

    return _c_loss 

def contrast_loss_L(output, target, R, window_size=(11,11), log_ratio=True):
    """Adapted from https://kornia.readthedocs.io/en/latest/_modules/kornia/metrics/ssim.html#ssim"""
    """
    Local contrast loss.
    """
    
    # prepare kernel
    kernel = get_gaussian_kernel2d((window_size[0], window_size[1]), (1.5, 1.5))

    # compute local mean per channel
    output_mu = filter2d(output, kernel=kernel)
    target_mu = filter2d(target, kernel=kernel)

    output_mu_sq = output_mu ** 2
    target_mu_sq = target_mu ** 2

    # compute local sigma per channel
    # use ReLU to avoid numerical error - variance equal to zero
    output_sigma_sq = F.relu(filter2d(output ** 2, kernel=kernel) - output_mu_sq)
    target_sigma_sq = F.relu(filter2d(target ** 2, kernel=kernel) - target_mu_sq)

    # weighting factor
    w_std = F.relu(1. - 2 * R).view(-1,1,1,1)

    # compute contrast loss 
    _c0 = torch.abs(output_mu - target_mu)
    if log_ratio:
        _c1 = -1 * (1 - w_std) * torch.log((output_sigma_sq + 7e-2) / (target_sigma_sq + 7e-2))
    else:
        _c1 = -1 * (1 - w_std) * (output_sigma_sq / (target_sigma_sq + 1e-1))
    _c2 =  w_std * torch.abs(target_sigma_sq - output_sigma_sq)
    _c_loss = _c0 + _c1 + _c2

    return _c_loss.mean(dim=(2,3))