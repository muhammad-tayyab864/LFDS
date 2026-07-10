import torch
import torch.nn as nn
import torch.nn.functional as F

class ACE(nn.Module):
    def __init__(self):
        super(ACE, self).__init__()
        self.learning_rate = 1e-4
        
        input_channel  = 1
        hidden_size    = 32
        output_channel = 1
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Downsampling
        self.down = DownSampling(input_channel, hidden_size)        # L0
        # Power Attention 
        self.pa_1 = PowerAttention(hidden_size, hidden_size, 2)     # L1
        self.pa_2 = PowerAttention(hidden_size, hidden_size, 4)     # L2
        self.pa_3 = PowerAttention(hidden_size, hidden_size, 8)     # L3
        self.pa_4 = PowerAttention(hidden_size, hidden_size, 16)    # L4
        # Upsampling
        self.up   = UpSampling(hidden_size, hidden_size)
        # Output
        self.out  = OutputBlock(hidden_size, output_channel)       # L5, L6

    def forward(self, x, r):
        r = torch.tensor(r)
        x = x.to(self.device)

        k = torch.pow((1 - r), (1/2.2))
        x = x * k
        # ===============================
        #   == Downsampling == Layer ==
        # ===============================
        fm  = self.down(x,r) # L0
        # ==================================
        #   == Power Attention == Layer ==
        # ==================================
        fm = self.pa_1(fm,r) # L1
        fm = self.pa_2(fm,r) # L2
        fm = self.pa_3(fm,r) # L3
        fm = self.pa_4(fm,r) # L4
        # fm = self.pa_5(fm,r) # L5
        # ==============================
        #   == Upsampling == Layer ==
        # ==============================
        fm = self.up(fm) # L6
        # =========================
        #   == Output == Layer ==
        # =========================
        out = self.out(fm) # L7
        out = torch.clamp(out , 0 ,1)

        return out

# =====================================
#   == Power == Attention == Block ==
# =====================================
class PowerAttention(nn.Module):
    def __init__(self,input_channel,output_channel,dilation):
        super(PowerAttention, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L1 = nn.Sequential(nn.ReflectionPad2d(dilation),
                                nn.Conv2d(input_channel+1, output_channel, kernel_size=3, stride=1, padding=0, dilation=dilation),
                                nn.LeakyReLU())
        
    def forward(self, x, R):
        R = R * torch.ones_like(x)[:,0:1,:,:].to(self.device)

        x = torch.cat([x,R],dim=1)
        x = self.L1(x)
        
        return x

# ===============================
#   == Downsampling == Block ==
# ===============================
class DownSampling(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(DownSampling, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L1 = nn.Sequential(nn.AvgPool2d(2),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(input_channel+1, output_channel, kernel_size=3, stride=1, padding=0, dilation=1),
                                nn.LeakyReLU())
    def forward(self, x, R):
        R = R * torch.ones_like(x)[:,0:1,:,:].to(self.device)
        x = torch.cat([x,R],dim=1)

        x = self.L1(x) 
        return x

# =============================
#   == Upsampling == Block ==
# =============================
class UpSampling(nn.Module):
    def __init__(self,input_channel, output_channel):
        super(UpSampling, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L1 = nn.Sequential(nn.PixelShuffle(2),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(input_channel // 4, output_channel, kernel_size=3,stride=1,padding=0,dilation=1),
                                nn.LeakyReLU())

    def forward(self, x):
        x = self.L1(x)
        return x

# =========================
#   == Output == Block ==
# =========================
class OutputBlock(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(OutputBlock, self).__init__()
        self.L1 = nn.Sequential(nn.Conv2d(input_channel, output_channel,  kernel_size=1,stride=1,padding=0,dilation=1))

    def forward(self, x):
        x = self.L1(x) 
        return x
