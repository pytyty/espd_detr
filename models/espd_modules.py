import torch
import torch.nn as nn
from .common import Conv, RepConv, EMA, SEAttention, DropPath

class Partial_conv3_Rep(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = RepConv(self.dim_conv3, self.dim_conv3, k=3, act=False, bn=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)


class Faster_Block_Rep_EMA(nn.Module):
    def __init__(self, inc, dim, n_div=4, mlp_ratio=2, drop_path=0.1):
        super().__init__()
        self.adjust_channel = Conv(inc, dim, 1) if inc != dim else nn.Identity()
        self.spatial_mixing = Partial_conv3_Rep(dim, n_div)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = nn.Sequential(
            Conv(dim, int(dim * mlp_ratio), 1),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1, bias=False)
        )
        self.attention = EMA(channels=dim)

    def forward(self, x):
        x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        return shortcut + self.attention(self.drop_path(self.mlp(x)))


class ContextGuideFusionModule(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        self.adjust_conv = Conv(inc[0], inc[1], k=1) if inc[0] != inc[1] else nn.Identity()
        self.se = SEAttention(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        x_concat = self.se(torch.cat([x0, x1], dim=1))
        x0_w, x1_w = torch.split(x_concat, [x0.size(1), x1.size(1)], dim=1)
        return torch.cat([x0 + (x1 * x0_w), x1 + (x0 * x1_w)], dim=1)


class SPDConv(nn.Module):
    def __init__(self, inc, ouc):
        super().__init__()
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(x)

class CSPOmniKernel(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), [int(x.size(1) * self.e), int(x.size(1) * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))