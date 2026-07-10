import torch
import torch.nn as nn
import torch.nn.functional as F
from models.power_mlp import MLP, ones_pad, power_attention
from utils import hpf, instance_norm
from guided_filter.guided_filter import FastGuidedFilter2d

class InvResidualModule(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, filter_size=3):
        super().__init__()
        self.expand = in_channels != exp_channels
        if self.expand:
            self.project = nn.Conv2d(in_channels, exp_channels, (1,1))
        self.use_residual = in_channels == out_channels
        self.conv = nn.Conv2d(exp_channels, exp_channels, (filter_size,filter_size), padding=int(filter_size / 2), dilation=1, groups=exp_channels)
        self.fuse = nn.Conv2d(exp_channels, out_channels, (1,1))

    def forward(self, x_input):
        x = x_input
        if self.expand:
            x = self.project(x)
            x = instance_norm(x)
            x = F.leaky_relu(x)
        x = self.conv(x)
        x = self.fuse(x)
        x = instance_norm(x)
        x = F.leaky_relu(x)
        if self.use_residual:
            x = x + x_input
        return x

class Project(nn.Module):
    def __init__(self, in_channels, out_channels, activate=False):
        super().__init__()
        self.project = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1))
        self.activate = activate

    def forward(self, x):
        x = self.project(x)
        if self.activate:
            x = F.leaky_relu(x)
        return x

class SPACE(nn.Module):
    """
    The saliency aware PCCE model.
    """
    
    def __init__(self, low_res=384, use_center_bias=True, use_gfcorrection=False, use_len=False):
        super(SPACE, self).__init__()
        scale_factor = 2

        self.use_center_bias = use_center_bias
        
        self.low_res = low_res
        self.M2D = nn.MaxPool2d((scale_factor, scale_factor)) 
        self.L1 = nn.Conv2d(1, 16, (3,3), padding=1, stride=2)
        self.L2 = InvResidualModule(16,24,16)
        self.L3 = InvResidualModule(16,32,24)
        self.L4a = InvResidualModule(24,40,32) if not self.use_center_bias else InvResidualModule(26,40,32)
        self.L4b = InvResidualModule(32,48,32,filter_size=5)
        self.L5a = InvResidualModule(32,64,48,filter_size=5)
        self.L5b = InvResidualModule(48,96,48,filter_size=5)

        self.fuse_2 = nn.Conv2d(16, 1, (1,1), padding=0)
        self.fuse_3 = nn.Conv2d(24, 1, (1,1), padding=0)
        self.fuse_4 = nn.Conv2d(32, 1, (1,1), padding=0)
        self.fuse_5 = nn.Conv2d(48, 1, (1,1), padding=0)
        out_feature_final = 5 #4 #5
        
        self.final = nn.Conv2d(out_feature_final, 1, (1,1), padding=0)
        
        self.power_mlp = MLP(3, 16, self.final.in_channels)

        self.use_gfcorrection = use_gfcorrection
        self.use_len = use_len
        
    def block1(self, x):
        x = self.L1(x)
        x = instance_norm(x)
        x = F.leaky_relu(x)
        return x

    def block2(self, x):
        x = self.L2(x)
        return x

    def block3(self, x):
        x = self.L3(x)
        return x

    def block4(self, x):
        if self.use_center_bias:
            xs = torch.linspace(-1, 1, steps=x.shape[3]).abs() ** 2
            ys = torch.linspace(-1, 1, steps=x.shape[2]).abs() ** 2
            xs, ys = torch.meshgrid(xs, ys, indexing='xy')
            center_bias = torch.stack([xs,ys]).unsqueeze(0)
            center_bias = center_bias.cuda() if x.is_cuda else center_bias 
            x = torch.cat([x, center_bias.repeat(x.shape[0],1,1,1)], dim=1)
        x = self.L4a(x)
        x = self.L4b(x)
        return x

    def block5(self, x):
        x = self.L5a(x)
        x = self.L5b(x)
        return x

    def encode(self, x):
        x = self.block1(x)

        x2 = self.block2(x)
        x = self.M2D(x2)

        x3 = self.block3(x)
        x = self.M2D(x3)

        x4 = self.block4(x)
        x = self.M2D(x4)

        x5 = self.block5(x)
        return x2, x3, x4, x5

    def decode(self, x2, x3, x4, x5, orig_size):

        _x2 = F.leaky_relu(instance_norm(self.fuse_2(x2)))
        _x3 = F.leaky_relu(instance_norm(self.fuse_3(x3)))
        _x4 = F.leaky_relu(instance_norm(self.fuse_4(x4)))
        _x5 = F.leaky_relu(instance_norm(self.fuse_5(x5)))

        x = torch.cat([
            F.interpolate(_x2, size=orig_size, mode='bilinear'),
            F.interpolate(_x3, size=orig_size, mode='bilinear'),
            F.interpolate(_x4, size=orig_size, mode='bilinear'),
            F.interpolate(_x5, size=orig_size, mode='bilinear')
        ], dim=1)

        return x

    def forward(self, x, R : float = 0.81):
        # Multiresolution preprocesssing
        
        orig_size = x.shape[2], x.shape[3]
        x_orig = x.clone()
        x_small = F.interpolate(x, (self.low_res, self.low_res), mode='bilinear')
        x = x_small.clone()

        # Encode decode

        x2, x3, x4, x5 = self.encode(x)
        x = self.decode(x2, x3, x4, x5, (self.low_res, self.low_res))       

        # Power 

        x = ones_pad(x)
        x_m = power_attention(x_small, torch.Tensor([R]))
        x_m = self.power_mlp(x_m)
        x = x * x_m

        # import matplotlib.pyplot as plt
        # f, axes = plt.subplots(1,5, figsize=(15,15))
        # axes[0].imshow(x[0,0].squeeze())
        # axes[1].imshow(x[0,1].squeeze())
        # axes[2].imshow(x[0,2].squeeze())
        # axes[3].imshow(x[0,3].squeeze())
        # axes[4].imshow(x[0,4].squeeze())
        # plt.show()
        # print(x.mean(dim=(2,3)))
        # print(x.std(dim=(2,3)))

        # Final

        x = self.final(x)
        if self.use_gfcorrection:
            min_length = self.low_res 
            g_scale = 1 
            # default radius radius=int(min_length / 6) 
            gf = FastGuidedFilter2d(radius=int(min_length / 6), eps=5e-7, s=g_scale)
            x = gf(x, x_small)
            x = F.interpolate(x, orig_size, mode='bilinear')
            if self.use_len:
                um1 = -F.relu(-hpf(x_orig)) # hpf(x_orig) # -F.relu(-hpf(x_orig))
                x = x + um1
        else:
            x = F.interpolate(x, orig_size, mode='bilinear')
            
        # import matplotlib.pyplot as plt
        # plt.imshow(x.squeeze())
        # plt.show()
        # print("mean {:.3f} max {:.3f} min {:.3f}".format(x.mean(), x.max(), x.min()))

        x = x_orig + x

        return torch.clamp(x, min=0., max=1.)

    