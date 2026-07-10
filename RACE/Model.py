import torch
import torch.nn as nn
import torch.nn.functional as F

class RACE(nn.Module):
    def __init__(self):
        super(RACE, self).__init__()
        self.learning_rate = 1e-4
        
        input_channel  = 1
        hidden_size    = 32
        output_channel = 1
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Downsampling
        self.down = DownSampling(input_channel, hidden_size)
        # Power Attention 
        self.pa_1 = PowerAttention(hidden_size, hidden_size, 2)
        self.pa_2 = PowerAttention(hidden_size, hidden_size, 4)
        self.pa_3 = PowerAttention(hidden_size, hidden_size, 8)
        self.pa_4 = PowerAttention(hidden_size, hidden_size, 16)
        # Upsampling
        self.up   =  UpSampling(hidden_size, hidden_size)
        # Output
        self.out  =  OutputBlock(hidden_size, output_channel)

    def forward(self, x, r):
        r = torch.tensor(r)
        x = x.to(self.device)
        # ===============================
        #   == Downsampling == Layer ==
        # ===============================
        fm  = self.down(x)
        # ==================================
        #   == Power Attention == Layer ==
        # ==================================
        fm = self.pa_1(fm,r)
        fm = self.pa_2(fm,r)
        fm = self.pa_3(fm,r)
        fm = self.pa_4(fm,r)
        # ==============================
        #   == Upsampling == Layer ==
        # ==============================
        fm = self.up(fm)
        # =========================
        #   == Output == Layer ==
        # =========================
        out = self.out(fm)
        # ===========================
        #   == Residual == Layer ==
        # ===========================
        out = x + out
        out = torch.clamp(out , 0 ,1)

        return out


# =====================================
#   == Power == Attention == Block ==
# =====================================
class PowerAttention(nn.Module):
    def __init__(self,input_channel,output_channel,dilation=1):
        super(PowerAttention, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L1 = nn.Sequential(nn.ReplicationPad2d(dilation),
                                nn.Conv2d(input_channel+1, input_channel, kernel_size=3, stride=1, padding=0, dilation=dilation),
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
                                nn.ReplicationPad2d(1),
                                nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=0, dilation=1),
                                nn.LeakyReLU())
    def forward(self, x):
        x = self.L1(x) 
        return x

# =============================
#   == Upsampling == Block ==
# =============================
class UpSampling(nn.Module):
    def __init__(self,input_channel, output_channel):
        super(UpSampling, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L1 = nn.Sequential(nn.PixelShuffle(2),
                                nn.ReplicationPad2d(1),
                                nn.Conv2d(input_channel//4, output_channel,   kernel_size=3,stride=1,padding=0,dilation=1))

    def forward(self, x):
        x = self.L1(x)
        return x


# =========================
#   == Output == Block ==
# =========================
class OutputBlock(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(OutputBlock, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L1 = nn.Sequential(nn.Conv2d(input_channel, output_channel,  kernel_size=1,stride=1,padding=0,dilation=1))

    def forward(self, x):
        x = self.L1(x) 
        return x
