import torch
from torch import nn
from einops.layers.torch import Rearrange
from ultralytics.nn.modules.conv import Conv,DWConv

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )
    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
# 只对部分通道部分通道做特征融合
# class CGAupsample(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(CGAupsample, self).__init__()
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(dim, reduction)
#         self.pa = PixelAttention(dim)
#         self.conv = nn.Conv2d(dim, dim, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()
#         self.Upsample=nn.Upsample(None,2,'nearest')
#     def forward(self, parameter):
#
#         x = parameter[0]
#         y = parameter[1]
#
#         x_batch=x.shape[1]
#         y_batch=y.shape[1]
#         if x_batch>y_batch:
#             x=self.Upsample(x)
#             multiple=x_batch//y_batch
#             x = list(x.chunk(multiple, 1))
#             initial = y+ x[0]
#             cattn = self.ca(initial)
#             sattn = self.sa(initial)
#             pattn1 = sattn + cattn
#             pattn2 = self.sigmoid(self.pa(initial, pattn1))
#             result = initial + pattn2 * x[0] + (1 - pattn2) * y
#             result = self.conv(result)
#             x[0]=result
#             return torch.cat(x,1)
#         y=self.Upsample(y)
#         multiple = y_batch//x_batch
#         y = list(y.chunk(multiple, 1))
#         initial = x + y[0]
#         cattn = self.ca(initial)
#         sattn = self.sa(initial)
#         pattn1 = sattn + cattn
#         pattn2 = self.sigmoid(self.pa(initial, pattn1))
#         result = initial + pattn2 * x + (1 - pattn2) * y[0]
#         result = self.conv(result)
#         y[0] = result
#         return torch.cat(y, 1)
class CGAupsample(nn.Module):
    def __init__(self, high_dim, low_dim, dim, reduction=8):
        super(CGAupsample, self).__init__()
        self.cv2 = Conv(low_dim, dim, 1)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.Upsample = nn.Upsample(None, 2, 'nearest')

    def forward(self, parameter):
        x = parameter[0]
        y = parameter[1]
        x_batch = x.shape[1]
        y_batch = y.shape[1]
        x_size=x.shape[2]
        y_size=y.shape[2]
        if x_size<y_size:
            x=self.Upsample(x)
        if x_size>y_size:
            y=self.Upsample()
        if x_batch>y_batch:
            y=self.cv2(y)
        if x_batch<y_batch:
            x=self.cv2(x)
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
class MAC3T(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 2, c_, 1, 1)
        self.cv3 = Conv(c_ * 2, c_, 1, 1)
        self.cv4 = Conv(c_ * 3, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.a = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        max_y1 = self.m(x)
        avg_y1 = self.a(x)
        y1 = self.cv2(torch.cat((max_y1, avg_y1), dim=1))
        max_y2 = self.m(y1)
        avg_y2 = self.a(y1)
        y2 = self.cv3(torch.cat((max_y2, avg_y2), dim=1))
        max_y3 = self.m(y2)
        avg_y3 = self.a(y2)
        return self.cv4(torch.cat((max_y3, avg_y3,x), dim=1))
'''
    Inverted_Residual_Block
'''
class Inverted_Residual_Block(nn.Module):
    def __init__(self, c1, c2, k):
        super(Inverted_Residual_Block, self).__init__()
        self.conv1 = Conv(c1, c1 * 2, 1)
        self.dw = DWConv(c1 * 2, c1 * 2, k=k)
        self.conv2 = Conv(c1 * 2, c2, 1)
    def forward(self, x):
        return self.conv2(self.dw(self.conv1(x)))
'''
    2C
'''
class Two_Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, k, s, None, g, act=act)
        self.cv2 = Conv(c1, c2, k, s, None, g, act=act)
        # self.cv3 = Conv(c1, c2, k, s, None, g, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        return torch.cat((x, self.cv1(x), self.cv2(x)), 1)
class ResIR2C(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, k=3, shortcut=False, n=1, e=0.5, ):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.shortcut = shortcut
        if shortcut and c1 == c2:
            self.shortcut = shortcut
        else:
            self.shortcut = False
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Inverted_Residual_Block(self.c, self.c, k) for _ in range(n))
        self.g = Two_Conv(self.c, self.c, k)
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y[0] = self.g(y[0])
        y.extend(m(y[-1]) for m in self.m)
        if self.shortcut:
            return self.cv2(torch.cat(y, 1)) + x
        return self.cv2(torch.cat(y, 1))
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))