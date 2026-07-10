import torch
import torch.nn as nn
import torch.nn.functional as F

class Meur(nn.Module):
    def __init__(self):
        super(Meur, self).__init__()
        input_channel = 1
        hidden_size = [4,8,20]
        output_channel = 1
        self.learning_rate = 1e-3

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.L0 = nn.Sequential(nn.Conv2d(input_channel,  hidden_size[0], kernel_size=3,stride=1,padding=1, dilation=1),  nn.ReLU())
        self.L1 = nn.Sequential(nn.AvgPool2d(1))

        self.L2_aspp_1 = nn.Sequential(nn.Conv2d(hidden_size[0], hidden_size[0], kernel_size=3,stride=1,padding=1, dilation=1),  nn.ReLU())
        self.L2_aspp_2 = nn.Sequential(nn.Conv2d(hidden_size[0], hidden_size[0], kernel_size=3,stride=1,padding=2, dilation=2),  nn.ReLU())
        self.L2_aspp_3 = nn.Sequential(nn.Conv2d(hidden_size[0], hidden_size[0], kernel_size=3,stride=1,padding=4, dilation=4),  nn.ReLU())
        self.L2_aspp_4 = nn.Sequential(nn.Conv2d(hidden_size[0], hidden_size[0], kernel_size=3,stride=1,padding=8, dilation=8),  nn.ReLU())
        self.L2_aspp_5 = nn.Sequential(nn.Conv2d(hidden_size[0], hidden_size[0], kernel_size=3,stride=1,padding=16,dilation=16), nn.ReLU())

        self.L3_attention = AttentionModule(hidden_size[2])

        self.L4 = nn.Sequential(nn.Conv2d(hidden_size[2], hidden_size[1], kernel_size=3,stride=1,padding=1,dilation=1), nn.ReLU())
        self.L5 = nn.Sequential(nn.Conv2d(hidden_size[1], output_channel, kernel_size=3,stride=1,padding=1,dilation=1), nn.ReLU())

    def forward(self, x, r):
        r = torch.tensor(r).to(self.device)
        x = x.to(self.device)

        fm = self.L0(x)

        fm = self.L1(fm)

        fm_1 = self.L2_aspp_1(fm)
        fm_2 = self.L2_aspp_2(fm)
        fm_3 = self.L2_aspp_3(fm)
        fm_4 = self.L2_aspp_4(fm)
        fm_5 = self.L2_aspp_5(fm)
        fm = torch.cat([fm_1,fm_2,fm_3,fm_4,fm_5],1)

        fm  = self.L3_attention(fm)
        fm  = self.L4(fm)
        out = self.L5(fm)

        out = out * (r / 0.4)
        out = x - out
        out = torch.clamp(out , 0, 1)

        return out

# ==============================
#   == Channel == Attention ==
# ==============================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels))

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        avg_out = avg_out.view(avg_out.size(0), -1)
        max_out = max_out.view(max_out.size(0), -1)

        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        channel_attention = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * channel_attention

# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=4):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(in_channels // reduction_ratio, in_channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         channel_attention = self.avg_pool(x)

#         channel_attention = channel_attention.view(channel_attention.size(0), -1)

#         channel_attention = self.fc1(channel_attention)
#         channel_attention = self.fc2(channel_attention)
#         channel_attention = channel_attention.unsqueeze(2).unsqueeze(3)
#         return channel_attention * x

# ==============================
#   == Spatial == Attention ==
# ==============================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined_out = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = torch.sigmoid(self.conv(combined_out))
        return x * spatial_attention
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.L1 = nn.Conv2d(20, 4, kernel_size=3, padding=4,dilation=4)
#         self.L2 = nn.Conv2d(4,  4, kernel_size=3, padding=4,dilation=4)
#         self.L3 = nn.Conv2d(4,  4, kernel_size=3, padding=4,dilation=4)
#         self.L4 = nn.Conv2d(4,  1, kernel_size=1, padding=0,dilation=1)

#     def forward(self, x):
#         spatial_attention = self.L1(x)
#         spatial_attention = self.L2(spatial_attention)
#         spatial_attention = self.L3(spatial_attention)
#         spatial_attention = self.L4(spatial_attention)
#         return spatial_attention * x
# =========================================
#   == Channel == Spatial == Attention ==
# =========================================
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x_channel = self.channel_attention(x)
        x_spatial = self.spatial_attention(x)

        # print(x_channel)
        # print(x_spatial)

        x = torch.softmax(x_channel + x_spatial, dim=1) + x
        return x