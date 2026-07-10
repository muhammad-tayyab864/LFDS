import torch
import torch.nn as nn
import piq
import torch.nn.functional as F
from kornia.filters.filter import filter2d
from kornia.filters.kernels import get_gaussian_kernel2d


# == Speedup ==
def speedup_encode(fm, rate=2):
    (b, c0, h0, w0) = fm.size()    # b c wl wr
    fm = F.pixel_unshuffle(fm, rate)  # b c*rate*rate wl/rate wr/rate
    fm = fm.view(b*(rate*rate),c0,h0//rate,w0//rate) # b*rate*rate c wl/rate wr/rate
    return fm

def speedup_decode_F(fm, rate=2):
    (b,c,h,w) = fm.size()
    fm = fm.view(b//(rate*rate),(rate*rate),c,h,w).permute(0,2,1,3,4)
    fm = F.pixel_shuffle(fm, rate)
    fm.view(b//(rate*rate),c,h*rate,w*rate)
    fm = torch.squeeze(fm,2)
    return fm # (b/4,c,2h,2w)

# == SSIM Loss ==
def ssim_loss(x, y):
    x = torch.clamp(x,0,1)
    y = torch.clamp(y,0,1)
    ssimloss = piq.SSIMLoss()
    loss = ssimloss(x, y)
    return loss

# == Power Loss ==
def r_loss(out, data, R):
    power_in  = torch.mean(torch.pow(data,2.2))
    power_out = torch.mean(torch.pow(out,2.2))
    R_score = 1 - (power_out / power_in)
    loss = torch.pow(R - R_score,2)
    return loss

# == Contrast Loss ==
def contrast_loss(out, data, mask=None, valid=None, sampling_factor=2):
    (b0,h0,w0,_) = mask.size()
    (b,c,h,w) = out.size()
    pad_size = 16
    value = 0.5

    if(b != b0):
        out  = F.pad(out,  (pad_size*4, pad_size*4, pad_size*4, pad_size*4), mode='constant', value=value)
        data = F.pad(data, (pad_size*4, pad_size*4, pad_size*4, pad_size*4), mode='constant', value=value)
        valid= F.pad(valid,(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        
        (b0,h0,w0,_) = mask.size()
        (b,c,h,w) = out.size()
        
        x_base = torch.linspace(0, 1, w0).repeat(b0,4,h0,1).cuda()   # 4
        disp = torch.bmm(mask.view((b0*h0),w0,w0), x_base.permute(0,2,3,1).contiguous().view((b0*h0),w0,4)).view(b0,h0,w0,4).permute(0,3,1,2) # contiguous(), 4
        disp = torch.abs(disp-x_base) 
        disp = speedup_decode_F(disp) # decode
        disp = F.pad(disp, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=value)

    else:
        out  = F.pad(out,  (pad_size*4, pad_size*4, pad_size*4, pad_size*4), mode='constant', value=value)
        data = F.pad(data, (pad_size*4, pad_size*4, pad_size*4, pad_size*4), mode='constant', value=value)
        mask = F.pad(mask, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        valid= F.pad(valid,(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        
        (b0,h0,w0,_) = mask.size()
        (b,c,h,w) = out.size()

        x_base = torch.linspace(0, 1, w0).repeat(b0,1,h0,1).cuda()   # 1
        disp = torch.bmm(mask.view((b0*h0),w0,w0), x_base.permute(0,2,3,1).view((b0*h0),w0,1)).view(b0,h0,w0,1).permute(0,3,1,2) # 1
        disp = torch.abs(disp-x_base) 

    disp = (disp * valid)

    out  = torch.clamp(F.interpolate(out,scale_factor=(1/sampling_factor),mode='bicubic'), 0, 1)

    data = torch.clamp(F.interpolate(data,scale_factor=(1/sampling_factor),mode='bicubic'), 0, 1)

    R2_loss = contrast_loss_L(out, data, disp).mean() + contrast_loss_G(out, data).mean()

    return R2_loss

def contrast_loss_G(output, target):
    """
    Global contrast loss.
    """

    _c0 = torch.abs(target.mean(dim=(2,3)) - output.mean(dim=(2,3)))
    
    _c1 = torch.log((target.var(dim=(2,3)) + 7e-2) / (output.var(dim=(2,3)) + 7e-2))


    _c_loss = _c0 + _c1

    return _c_loss 

def contrast_loss_L(output, target, disp):
    """Adapted from https://kornia.readthedocs.io/en/latest/_modules/kornia/metrics/ssim.html#ssim"""
    """
    Local contrast loss.
    """
    
    # prepare kernel
    window_size=(11,11)
    kernel = get_gaussian_kernel2d((window_size[0], window_size[1]), (1.5, 1.5))

    # compute local mean per channel
    output_mu = filter2d(output, kernel=kernel)
    target_mu = filter2d(target, kernel=kernel)
    disp_mu   = filter2d(disp, kernel=kernel)

    output_mu_sq = output_mu ** 2
    target_mu_sq = target_mu ** 2
    disp_mu_sq   = disp_mu ** 2

    # compute local sigma per channel
    # use ReLU to avoid numerical error - variance equal to zero
    output_sigma_sq = F.relu(filter2d(output ** 2, kernel=kernel) - output_mu_sq)
    target_sigma_sq = F.relu(filter2d(target ** 2, kernel=kernel) - target_mu_sq)

    # compute contrast loss 

    _c0 = torch.abs(output_mu - target_mu)
    
    _c1 = disp_mu_sq * torch.log((target_sigma_sq + 7e-2) / (output_sigma_sq + 7e-2))

    _c_loss = _c0 + _c1

    return _c_loss

# == TV Loss ==
def TV_loss(out, data):
    tvloss = piq.TVLoss(norm_type='l2')
    ddm = torch.clamp(data - out, 0, 1)
    loss = tvloss(ddm)

    return loss

# == Consistency Loss ==
def consistency_loss(outL, outR, M_right_to_left, M_left_to_right, V_left, V_right, sampling_factor):
    outL  = F.interpolate(outL,  scale_factor=(1/sampling_factor), mode='bicubic')
    outR  = F.interpolate(outR,  scale_factor=(1/sampling_factor), mode='bicubic')

    loss_consistency = consistencyLoss(outL,outR, M_right_to_left,M_left_to_right, V_left, V_right)

    return loss_consistency


def consistencyLoss(Res_left,Res_right,M_right_to_left, M_left_to_right,V_left,V_right):
    criterion = nn.L1Loss()

    (b, c, h, w) = Res_left.size()
    (b0, h0, w0, _) = M_right_to_left.size()

    if(b != b0):
        Res_left_down  = speedup_encode(Res_left)
        Res_right_down = speedup_encode(Res_right)
        (b, c, h, w) = Res_left_down.size()

        Res_leftT  = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w), Res_right_down.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2) # down
        Res_rightT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w), Res_left_down.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2) # down
        
        Res_leftT  = speedup_decode_F(Res_leftT)
        Res_rightT = speedup_decode_F(Res_rightT)

    else:
        Res_leftT  = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w), Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        Res_rightT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w), Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
    
    loss_cons = criterion(Res_left * V_left.repeat(1, 3, 1, 1), Res_leftT * V_left.repeat(1, 3, 1, 1)) + \
                criterion(Res_right * V_right.repeat(1, 3, 1, 1), Res_rightT * V_right.repeat(1, 3, 1, 1))
    
    return loss_cons

# == Disp Loss == 
def cross_loss(dataL, dataR, M_right_to_left, M_left_to_right, V_left, V_right, sampling_factor):
    dataL = F.interpolate(dataL, scale_factor=(1/sampling_factor), mode='bicubic')
    dataR = F.interpolate(dataR, scale_factor=(1/sampling_factor), mode='bicubic')

    resL = dataL
    resR = dataR

    loss_photo = photometricLoss(resL,resR, M_right_to_left,M_left_to_right, V_left,V_right)
    loss_smooth = smoothLoss_mask(M_right_to_left, M_left_to_right)
    loss_cycle = cycleLoss(resL,resR, M_right_to_left,M_left_to_right, V_left,V_right)

    return loss_photo + loss_smooth + loss_cycle

# == Photometric Loss ==
def photometricLoss(Res_left, Res_right, M_right_to_left, M_left_to_right,V_left,V_right):
    criterion = nn.L1Loss()
    (b, c, h, w) = Res_left.size()
    (b0, h0, w0, _) = M_right_to_left.size()

    if(b != b0):
        Res_left_down  = speedup_encode(Res_left)
        Res_right_down = speedup_encode(Res_right)

        (b, c, h, w) = Res_left_down.size()

        Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_right_down.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2) # down
        Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_left_down.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2) # down
        
        Res_leftT  = speedup_decode_F(Res_leftT)
        Res_rightT = speedup_decode_F(Res_rightT)

    else:
        Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

    loss_photo = criterion(Res_left * V_left.repeat(1, 3, 1, 1), Res_leftT * V_left.repeat(1, 3, 1, 1)) + \
                    criterion(Res_right * V_right.repeat(1, 3, 1, 1), Res_rightT * V_right.repeat(1, 3, 1, 1))
    return loss_photo

# == Smooth Loss ==
def smoothLoss_mask(M_right_to_left, M_left_to_right):
    criterion = nn.L1Loss()
    loss_h = criterion(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                criterion(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
    loss_w = criterion(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                criterion(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
    loss_smooth = loss_w + loss_h

    return loss_smooth

# == Cycle Loss ==
def cycleLoss(Res_left, Res_right, M_right_to_left, M_left_to_right,V_left,V_right):
    criterion = nn.L1Loss()
    
    (b, c, h, w) = Res_left.size()
    (b0, h0, w0, _) = M_right_to_left.size()

    if(b != b0):
        Res_left_down  = speedup_encode(Res_left)
        Res_right_down = speedup_encode(Res_right)

        (b, c, h, w) = Res_left_down.size()

        Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_right_down.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2) # down
        Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_left_down.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2) # down
        Res_left_cycle = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_rightT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        Res_right_cycle = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_leftT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        
        Res_left_cycle  = speedup_decode_F(Res_left_cycle)
        Res_right_cycle = speedup_decode_F(Res_right_cycle)

    else:
        Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        Res_left_cycle = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_rightT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        Res_right_cycle = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_leftT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

    loss_cycle = criterion(Res_left * V_left.repeat(1, 3, 1, 1), Res_left_cycle * V_left.repeat(1, 3, 1, 1)) + \
                    criterion(Res_right * V_right.repeat(1, 3, 1, 1), Res_right_cycle * V_right.repeat(1, 3, 1, 1))
    
    return loss_cycle








