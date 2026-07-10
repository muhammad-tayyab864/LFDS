import torch
import torch.nn as nn
import piq
import torch.nn.functional as F

def ssim_loss(x, y):
    # SSIM
    x = torch.clamp(x,0,1)
    y = torch.clamp(y,0,1)
    ssimloss = piq.SSIMLoss()
    loss = ssimloss(x, y)
    return loss*1

def r_loss(out, data, R):
    R = torch.tensor(R)
    # Power
    power_in = torch.mean(torch.pow(data,2.2))
    power_out = torch.mean(torch.pow(out,2.2))
    loss = torch.pow(power_out - (1-R)*power_in, 2).mean()
    return loss * 2000

def MAE_loss(out,data):
    maeloss = nn.L1Loss()
    loss = maeloss(out,data)
    return loss * 0.5

def TV_loss(out):
    TVloss = piq.TVLoss(norm_type='l2_squared')
    TV = TVloss(out) / (out.size()[2] * out.size()[3])
    return TV * 10