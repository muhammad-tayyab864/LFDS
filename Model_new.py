import torch
import torch.nn as nn
import torch.nn.functional as F

from variable_luminance import rgb_to_ycbcr, ycbcr_to_rgb

import time

activation = nn.LeakyReLU()
output_activation = nn.Tanh()

extract_layer = 4

def decompose_imgs(imgs, luminance_const = [0.206, 0.339, 0.454]):
    """
    RGB to YCbCr, returns luminance and chrominance.
    """
    _ycbcr = rgb_to_ycbcr(imgs, luminance_const=luminance_const)
    _y = _ycbcr[:,:1]
    _cbcr = _ycbcr[:,1:]
    return _y, _cbcr

def compose_imgs(y_imgs, cbcr_imgs, luminance_const = [0.206, 0.339, 0.454]):
    """
    YCbCr to RGB, returns RGB.
    """
    _ycbcr = torch.cat((y_imgs, cbcr_imgs), dim=1)
    rgb_imgs = ycbcr_to_rgb(_ycbcr, luminance_const=luminance_const)
    return rgb_imgs

class MODEL(nn.Module):
    def __init__(self, hidden_size, sampling_factor, blending_factor, speedup, test=0):
        super(MODEL, self).__init__()
        input_channel  = 1
        output_channel = 1
        self.blending_factor = blending_factor
        self.speedup = speedup
        self.test = test
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # == Downsampling ==
        self.L1_down = DownSampling(input_channel, hidden_size, sampling_factor=sampling_factor)
        # == Disp ==
        self.L2 = []
        for i in range(extract_layer):
            self.L2.append(ASPP(hidden_size, hidden_size))
        # == PAM ==
        self.conv = ASPP(hidden_size*extract_layer, hidden_size)
        self.bbipam  = bbiPAM(hidden_size)
        # == L3 ==
        self.L3 = PowerAttention(hidden_size, hidden_size)
        # == Upsampling ==
        self.L4_up  =   UpSampling(hidden_size, hidden_size, sampling_factor=sampling_factor)
        # == Output ==
        self.L4_out =   OutputBlock(hidden_size, output_channel)

    def forward(self, xL, xR, r):
        # ===============================
        #   == Downsampling == Layer ==
        # ===============================
        # start_time = time.time()
        fm1L  = self.L1_down(xL)
        fm1R  = self.L1_down(xR)
        # end_time = time.time()
        # print(f'down_time     ={end_time - start_time}')
        # =======================
        #   == PAM == Layer ==
        # =======================
        # start_time = time.time()
        fmL_list = []
        fmR_list = []
        fmL,fmR  = self.L2[0](fm1L,r), self.L2[0](fm1R,r)
        fmL_list.append(fmL)
        fmR_list.append(fmR)
        for i in range(1,len(self.L2)):
            fmL = self.L2[i](fmL,r)
            fmR = self.L2[i](fmR,r) 
            fmL_list.append(fmL)
            fmR_list.append(fmR)
        
        fmL_list = torch.cat(fmL_list,dim=1)
        fmR_list = torch.cat(fmR_list,dim=1)
        # end_time = time.time()
        # print(f'ASPP_time     ={end_time - start_time}')
        # start_time = time.time()
        # fmL = self.conv(fmL_list,r)
        # fmR = self.conv(fmR_list,r)
        if(self.test == 0):
            fmL, fmR, (M_right_to_left, M_left_to_right), (V_left, V_right) = self.bbipam(fm1L, fm1R, fmL_list, fmR_list, self.blending_factor, self.speedup, test=0)
        elif(self.test == 1):
            fmL, fmR = self.bbipam(fm1L, fm1R, fmL_list, fmR_list, self.blending_factor, self.speedup, test=1)
        # end_time = time.time()
        # print(f'PAM_core_time ={end_time - start_time}')
        # ======================================
        #   == Feature == Extract == Layer ==
        # ======================================
        # for i in range(len(self.L3)):
        # start_time = time.time()
        fmL = self.L3(fmL,r)
        fmR = self.L3(fmR,r)
        # end_time = time.time()
        # print(f'L3_time       ={end_time - start_time}')
        # ==============================
        #   == Upsampling == Layer ==
        # ==============================
        # start_time = time.time()
        fmL = self.L4_up(fmL)
        fmR = self.L4_up(fmR)
        # =========================
        #   == Output == Layer ==
        # =========================
        fmL = self.L4_out(fmL)
        fmR = self.L4_out(fmR)
        # ============================
        #   == Add == to == Input ==
        # ============================
        # outL = yL + fmL
        # outR = yR + fmR
        outL = xL + fmL
        outR = xR + fmR
        # outL, outR = outL.cpu(), outR.cpu()
        # outL = compose_imgs(outL, cbcrL)
        # outR = compose_imgs(outR, cbcrR)

        outL = torch.clamp(outL , 0 ,1)
        outR = torch.clamp(outR , 0 ,1)

        # end_time = time.time()
        # print(f'up&out_time   ={end_time - start_time}')

        if(self.test == 0):
            return outL, outR, M_right_to_left, M_left_to_right, V_left, V_right
        elif(self.test == 1):
            return outL, outR
# ==============
#   == ASPP ==
# ==============
class ASPP(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(ASPP, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L0 = nn.Sequential(nn.Conv2d(input_channel, output_channel//4, kernel_size=3, stride=1, padding=1, dilation=1),
                                activation
                                ).to(self.device)
        self.L1 = nn.Sequential(nn.Conv2d(input_channel, output_channel//4, kernel_size=3, stride=1, padding=2, dilation=2),
                                activation
                                ).to(self.device)
        self.L2 = nn.Sequential(nn.Conv2d(input_channel, output_channel//4, kernel_size=3, stride=1, padding=3, dilation=3),
                                activation
                                ).to(self.device)
        self.L3 = nn.Sequential(nn.Conv2d(input_channel, output_channel//4, kernel_size=3, stride=1, padding=4, dilation=4),
                                activation
                                ).to(self.device)

    def forward(self, x, R=0):
        x_list = []
        x_list.append(self.L0(x))
        x_list.append(self.L1(x))
        x_list.append(self.L2(x))
        x_list.append(self.L3(x))

        x = torch.cat(x_list,dim=1)
        
        return x
# =====================================
#   == Power == Attention == Block ==
# =====================================
class PowerAttention(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(PowerAttention, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L0 = nn.Sequential(nn.Conv2d(input_channel+1, output_channel//4, kernel_size=3, stride=1, padding=1, dilation=1),
                                activation
                                ).to(self.device)
        self.L1 = nn.Sequential(nn.Conv2d(input_channel+1, output_channel//4, kernel_size=3, stride=1, padding=2, dilation=2),
                                activation
                                ).to(self.device)
        self.L2 = nn.Sequential(nn.Conv2d(input_channel+1, output_channel//4, kernel_size=3, stride=1, padding=3, dilation=3),
                                activation
                                ).to(self.device)
        self.L3 = nn.Sequential(nn.Conv2d(input_channel+1, output_channel//4, kernel_size=3, stride=1, padding=4, dilation=4),
                                activation
                                ).to(self.device)

    def forward(self, x, R):
        R = R * torch.ones_like(x)[:,0:1,:,:].to(self.device)
        x = torch.cat([x,R],dim=1)

        x_list = []
        x_list.append(self.L0(x))
        x_list.append(self.L1(x))
        x_list.append(self.L2(x))
        x_list.append(self.L3(x))

        x = torch.cat(x_list,dim=1)
        
        return x

# ===============================
#   == Downsampling == Block ==
# ===============================
class DownSampling(nn.Module):
    def __init__(self,input_channel,output_channel,sampling_factor):
        super(DownSampling, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L1 = nn.Sequential(nn.Conv2d(input_channel, input_channel*4, kernel_size=3, stride=1, padding=1, dilation=1),
                                nn.AvgPool2d(sampling_factor),
                                nn.Conv2d(input_channel*4, output_channel, kernel_size=3, stride=1, padding=1, dilation=1), 
                                activation
                                )
    def forward(self, x):
        x = self.L1(x) 
        return x

# =============================
#   == Upsampling == Block ==
# =============================
class UpSampling(nn.Module):
    def __init__(self,input_channel,output_channel,sampling_factor):
        super(UpSampling, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.L0 = ASPP(input_channel,input_channel)
        self.L1 = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3,stride=1,padding=1,dilation=1),
                                nn.PixelShuffle(sampling_factor),
                                nn.Conv2d(input_channel//(sampling_factor**2), output_channel, kernel_size=3,stride=1,padding=1,dilation=1),
                                activation
                                )
    def forward(self, x):
        # x = self.L0(x)
        x = self.L1(x)
        return x

# =========================
#   == Output == Block ==
# =========================
class OutputBlock(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(OutputBlock, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L1 = nn.Sequential(nn.Conv2d(input_channel, input_channel,   kernel_size=3,stride=1,padding=1,dilation=1), 
                                nn.Conv2d(input_channel, output_channel,  kernel_size=1,stride=1,padding=0,dilation=1), 
                                output_activation
                                )
    def forward(self, x):
        # x = x.to(self.device)
        x = self.L1(x) 
        return x

# =================
#   == Speedup ==
# =================
def speedup_encode(fm, rate=2):
    (b, c0, h0, w0) = fm.size()    # b c wl wr
    fm = F.pixel_unshuffle(fm, rate)  # b c*rate*rate wl/rate wr/rate
    fm = fm.view(b*rate*rate,c0,h0//rate,w0//rate) # b*rate*rate c wl/rate wr/rate
    return fm

def speedup_decode_V(valid, size, rate=2):
    (b, h, w) = size
    valid = valid.view(b//(rate*rate),(rate*rate),h,w)   # b 4 h/2 w/2 
    valid = F.pixel_shuffle(valid,2) # b 1 h w
    valid = valid.view(-1,1,h*rate,w*rate)
    return valid # (b h) wl wr

def speedup_decode_F(fm, rate=2):
    (b,c,h,w) = fm.size()
    fm = fm.view(b//(rate*rate),(rate*rate),c,h,w).permute(0,2,1,3,4)
    fm = F.pixel_shuffle(fm, rate)
    fm.view(b//(rate*rate),c,h*rate,w*rate)
    fm = torch.squeeze(fm,2)
    return fm # (b/4,c,2h,2w)

# ==========================================
#   == Parallax == Attention == Module ==
# ==========================================
class bbiPAM(nn.Module):
    def __init__(self, channels):
        super(bbiPAM, self).__init__()
        # self.conv = nn.Conv2d(channels*extract_layer, channels, kernel_size=3, padding=1, dilation=1)
        self.bq = nn.Conv2d(extract_layer*channels, channels, 1, 1, 0, groups=extract_layer, bias=True)
        self.bs = nn.Conv2d(extract_layer*channels, channels, 1, 1, 0, groups=extract_layer, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(extract_layer * channels)
        self.bn = nn.BatchNorm2d(extract_layer * channels)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, blending_factor, speedup, test):
        if(speedup == 1):
            catfea_left  = speedup_encode(catfea_left)
            catfea_right = speedup_encode(catfea_right)
        
        b0, c0, h0, w0 = x_left.shape
        Q = self.bq(self.rb(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr

        # M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        # M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl
        
        if(speedup == 1):
            # M_right_to_left_relaxed = M_Relax(self.softmax(score), num_pixels=2)
            V_left = torch.bmm(M_Relax(self.softmax(score), num_pixels=2).contiguous().view(-1, w).unsqueeze(1),
                            self.softmax(score.permute(0, 2, 1)).permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                            ).detach().contiguous().view(b0*4, 1, h, w)  # b0*4
            # M_left_to_right_relaxed = M_Relax(self.softmax(score.permute(0, 2, 1)), num_pixels=2)
            V_right = torch.bmm(M_Relax(self.softmax(score.permute(0, 2, 1)), num_pixels=2).contiguous().view(-1, w).unsqueeze(1),  
                                self.softmax(score).permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                            ).detach().contiguous().view(b0*4, 1, h, w)   # b0*4
        else:
            # M_right_to_left_relaxed = M_Relax(self.softmax(score), num_pixels=2)
            V_left = torch.bmm(M_Relax(self.softmax(score), num_pixels=2).contiguous().view(-1, w0).unsqueeze(1),
                            self.softmax(score.permute(0, 2, 1)).permute(0, 2, 1).contiguous().view(-1, w0).unsqueeze(2)
                            ).detach().contiguous().view(b0, 1, h0, w0)  # b0
            # M_left_to_right_relaxed = M_Relax(self.softmax(score.permute(0, 2, 1)), num_pixels=2)
            V_right = torch.bmm(M_Relax(self.softmax(score.permute(0, 2, 1)), num_pixels=2).contiguous().view(-1, w0).unsqueeze(1),  
                                self.softmax(score).permute(0, 2, 1).contiguous().view(-1, w0).unsqueeze(2)
                            ).detach().contiguous().view(b0, 1, h0, w0)  # b0

        V_left = torch.tanh(5 * V_left)
        V_right = torch.tanh(5 * V_right)

        if(speedup == 1):
            size = (b,h,w)
            V_left  = speedup_decode_V(V_left,  size)
            V_right = speedup_decode_V(V_right, size)

            x_right_down = speedup_encode(x_right)
            x_left_down  = speedup_encode(x_left)

            (b,c,h,w) = x_right_down.size()
            x_leftT = torch.bmm(self.softmax(score), x_right_down.permute(0, 2, 3, 1).contiguous().view(-1, w, c) # down
                                ).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)                           
            x_rightT = torch.bmm(self.softmax(score.permute(0, 2, 1)), x_left_down.permute(0, 2, 3, 1).contiguous().view(-1, w, c) # down
                                ).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)                              
            
            x_leftT  = speedup_decode_F(x_leftT)
            x_rightT = speedup_decode_F(x_rightT)
        
        else:
            x_leftT = torch.bmm(self.softmax(score), x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                                ).contiguous().view(b0, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
            x_rightT = torch.bmm(self.softmax(score.permute(0, 2, 1)), x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                                ).contiguous().view(b0, h0, w0, c0).permute(0, 3, 1, 2)                              #  B, C0, H0, W0

        out_left  = x_left  * (1 - V_left.repeat(1, c0, 1, 1))  + (blending_factor*x_leftT  + (1-blending_factor)*x_left)  * V_left.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right.repeat(1, c0, 1, 1)) + (blending_factor*x_rightT + (1-blending_factor)*x_right) * V_right.repeat(1, c0, 1, 1)
        
        if(test == 0):
            if(speedup == 1):
                return out_left, out_right, \
                        (self.softmax(score).contiguous().view(b0*4, h, w, w), self.softmax(score.permute(0, 2, 1)).contiguous().view(b0*4, h, w, w)),\
                        (V_left, V_right) # b0*4
            
            else:
                return out_left, out_right, \
                        (self.softmax(score).contiguous().view(b0, h0, w0, w0), self.softmax(score.permute(0, 2, 1)).contiguous().view(b0, h0, w0, w0)),\
                        (V_left, V_right) # b0
        elif(test == 1):
            return out_left, out_right

def M_Relax(M, num_pixels):
    M_relaxed = torch.zeros_like(M)

    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        M_relaxed = torch.sum(torch.cat([M_relaxed.unsqueeze(1),pad(M[:, :-1-i, :]).unsqueeze(1)], 1), dim=1)
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        M_relaxed = torch.sum(torch.cat([M_relaxed.unsqueeze(1),pad(M[:, i+1:, :]).unsqueeze(1)],  1), dim=1)
    return M_relaxed

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=extract_layer, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=extract_layer, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x
    

# def M_Relax(M, num_pixels):
#     _, u, v = M.shape
#     M_list = []
#     M_list.append(M.unsqueeze(1))
#     for i in range(num_pixels):
#         pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
#         pad_M = pad(M[:, :-1-i, :])
#         M_list.append(pad_M.unsqueeze(1))
#     for i in range(num_pixels):
#         pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
#         pad_M = pad(M[:, i+1:, :])
#         M_list.append(pad_M.unsqueeze(1))
#     M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
#     return M_relaxed

# ==========================
# #   == Speedup == Old ==
# # ========================
# def speedup_encode(fm, rate=2):
#     (b, c0, h0, w0) = fm.size()    # b c wl wr
#     fm = F.pixel_unshuffle(fm, rate)  # b c*rate*rate wl/rate wr/rate
#     fm = fm.view(b*rate*rate,c0,h0//rate,w0//rate) # b*rate*rate c wl/rate wr/rate
#     return fm

# def speedup_decode_M(mask, size, rate=2):
#     (b, h, w) = size
#     mask = mask.view(b//(rate*rate),(rate*rate),h,w,w)   # b 4 h/2 wl/2 wr/2
#     mask = mask.permute(0,2,3,4,1)      # b h/2 wl/2 wr/2 4
#     mask = mask.view(b//(rate*rate),h,w,w,rate,rate) # b h/2 wl/2 wr/2 2 2

#     zeros = torch.zeros_like(mask).repeat(1,1,1,1,1,rate-1)
#     mask = torch.cat([mask,zeros], dim=5) # b h/2 wl/2 wr/2 2 [2 2]
#     mask = torch.cat([mask[:,:,:,:,:,0:1],mask[:,:,:,:,:,2:3],mask[:,:,:,:,:,3:4],mask[:,:,:,:,:,1:2]], dim=5)

#     mask = mask.permute(0,1,4,5,2,3)      # b h/2 2 4 wl/2 wr/2
#     mask = mask.contiguous().view(b//(rate*rate),h*rate,(rate*rate),w,w)    # b h 4 wl/2 wr/2
#     mask = F.pixel_shuffle(mask,2) # b h 1 wl wr
#     mask = mask.view(-1,w*rate,w*rate)
#     return mask # (b h) wl wr

# def speedup_decode_V(valid, size, rate=2):
#     (b, h, w) = size
#     valid = valid.view(b//(rate*rate),(rate*rate),h,w)   # b 4 h/2 w/2 
    
#     valid = F.pixel_shuffle(valid,2) # b 1 h w
#     valid = valid.view(-1,1,h*rate,w*rate)
#     return valid # (b h) wl wr



