"""
Gaussian Head: EvGGS-style GSRegressor for predicting R, S, α.

Directly adapted from EvGGS:
    D:/cugdocuments/科研/lw/EvGGS-master/lib/network/gsregressor.py

Input: depth(1ch) + intensity(1ch) + VGGT_feature(from tokens) + original_event_frame(3+5=8ch)
        = input_dim in EvGGS is 2+32+8=42, but we adapt for VGGT features.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """From EvGGS gsregressor.py"""
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes) if not (stride == 1 and in_planes == planes) else nn.Identity()
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class EPGGSGaussianHead(nn.Module):
    """
    Matches EvGGS GSRegressor exactly.

    input_dim default: 2 (depth+intensity) + 32 (VGGT_feat projected) + 8 (frame+voxel) = 42
    """

    def __init__(self, input_dim=42, hidden_dim=256, norm_fn='group'):
        super().__init__()
        self.embedding = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1)
        self.res1 = ResidualBlock(hidden_dim, hidden_dim // 4, norm_fn=norm_fn)

        self.rot_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 3, kernel_size=1),
            nn.Softplus(beta=100),
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, input_dim, H, W) — [depth, intensity, vggt_feat, event_frame, event_voxel]

        Returns:
            rot: (B, 4, H, W) normalized quaternion
            scale: (B, 3, H, W) positive scales
            opacity: (B, 1, H, W) [0, 1]
        """
        x = self.embedding(x)
        out = self.res1(x)

        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        scale_out = torch.clamp_max(self.scale_head(out), 0.001)

        opacity_out = self.opacity_head(out)

        return rot_out, scale_out, opacity_out
