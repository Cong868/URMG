import torch
from torch import nn
from einops.layers.torch import Rearrange
from ultralytics.nn.modules.conv import Conv,DWConv
from ultralytics.nn.modules.block import SPPF
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
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
# class MAC3T(nn.Module):
#     def __init__(self, c1, c2, k=5):
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 2, c_, 1, 1)
#         self.cv3 = Conv(c_ * 2, c_, 1, 1)
#         self.cv4 = Conv(c_ * 3, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.a = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
#
#     def forward(self, x):
#         x = self.cv1(x)
#         max_y1 = self.m(x)
#         avg_y1 = self.a(x)
#         y1 = self.cv2(torch.cat((max_y1, avg_y1), dim=1))
#         max_y2 = self.m(y1)
#         avg_y2 = self.a(y1)
#         y2 = self.cv3(torch.cat((max_y2, avg_y2), dim=1))
#         max_y3 = self.m(y2)
#         avg_y3 = self.a(y2)
#         return self.cv4(torch.cat((max_y3, avg_y3,x), dim=1))
class MAC3T(nn.Module):
    def __init__(self, B1,B2,B3,B4,c2, k=5):
        super().__init__()
        c_ = c2// 2  # hidden channels
        # self.cv1 = Conv(704, c_, 1, 1)
        self.cv1 = Conv(B1*2+B2*2+B3*2+B4, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.cv2 = Conv(c_ * 2, c_, 1, 1)
        self.cv3 = Conv(c_ * 2, c_, 1, 1)
        self.cv4 = Conv(c_*3, c2, 1, 1)
        self.m_1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.a_1 = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self,x):
        B5, B4, B3, B2=x[0],x[1],x[2],x[3]
        B5_a=self.a(self.a(self.a(B5)))
        B5_m = self.m(self.m(self.m(B5)))
        B4_a=self.a(self.a(B4))
        B4_m = self.m(self.m(B4))
        B3_a = self.a(B3)
        B3_m= self.a(B3)
        x=torch.cat((B2,B5_a,B5_m,B4_a,B4_m,B3_a,B3_m),dim=1)
        B, C, H, W = x.shape
        res = x.unsqueeze(dim=2)
        x = Rearrange('b c t h w -> b (c t) h w')(res)
        x = self.cv1(x)
        max_y1 = self.m_1(x)
        avg_y1 = self.a_1(x)
        y1 = self.cv2(torch.cat((max_y1, avg_y1), dim=1))
        max_y2 = self.m_1(y1)
        avg_y2 = self.a_1(y1)
        y2 = self.cv3(torch.cat((max_y2, avg_y2), dim=1))
        max_y3 = self.m_1(y2)
        avg_y3 = self.a_1(y2)
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
    def __init__(self, c1, c2, k=3, shortcut=False, n=1, e=0.5):
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
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x=x.to(device="cpu")
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)).to(device="cuda:0")
        print(hw.device)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class MySpatialAttention(nn.Module):
    def __init__(self,dim):
        super(MySpatialAttention, self).__init__()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.avgpool=nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.sa = nn.Conv2d(dim*2, dim, 1, padding=0,stride=1)
    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max= self.avgpool(x)
        sattn = self.sa(torch.concat((x_avg,x_max),1))
        return sattn

class MyChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(MyChannelAttention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class CGAAttention(nn.Module):
    def __init__(self, dim, reduction=8, radio=0.5):
        super(CGAAttention, self).__init__()
        middle_dim = int(dim * radio)
        self.sa = MySpatialAttention(middle_dim)
        self.ca = MyChannelAttention(middle_dim, reduction)
        self.conv = Conv(dim, dim, 1, act=True)

    def forward(self, x):
        input = x
        x, y = list(x.chunk(2, 1))
        cattn = self.ca(x)
        sattn = self.sa(y)
        result = self.conv(torch.cat((cattn, sattn), dim=1))+ input
        return result
