"""
Intensity Head: Predict grayscale image from VGGT tokens.

Simple design: final-layer VGGT tokens (2048D) → reshape to 2D → Conv upsampling → grayscale.
"""
import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, activation='relu', norm='BN'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=(norm is None))
        self.norm = nn.BatchNorm2d(out_ch) if norm == 'BN' else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvLayer(in_ch, out_ch, kernel_size, 1, padding)
    def forward(self, x):
        return self.conv(self.up(x))


class IntensityDecoder(nn.Module):
    """
    VGGT tokens → grayscale image.

    Args:
        vggt_dim: Token dimension (2048 for frame+global concat)
        base_channels: Channels after first conv
        output_channels: 1 for grayscale
    """
    def __init__(self, vggt_dim=2048, base_channels=256, output_channels=1):
        super().__init__()
        self.vggt_dim = vggt_dim

        self.token_proj = ConvLayer(vggt_dim, base_channels, kernel_size=1)

        # Upsample 32→64→128→256
        self.up1 = UpsampleConvLayer(base_channels, base_channels // 2)
        self.up2 = UpsampleConvLayer(base_channels // 2, base_channels // 4)
        self.up3 = UpsampleConvLayer(base_channels // 4, base_channels // 4)

        self.pred = nn.Sequential(
            nn.Conv2d(base_channels // 4, output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, vggt_tokens, skip_features=None):
        """
        Args:
            vggt_tokens: (B*V, N, 2048) — final layer patch tokens
            skip_features: unused (kept for API compatibility)

        Returns:
            intensity: (B*V, 1, 256, 256) grayscale image
        """
        B, N, D = vggt_tokens.shape
        patch_res = int(N ** 0.5)  # 32 for 448/14

        x = vggt_tokens.permute(0, 2, 1).reshape(B, D, patch_res, patch_res)
        x = self.token_proj(x)      # (B, 256, 32, 32)
        x = self.up1(x)             # (B, 128, 64, 64)
        x = self.up2(x)             # (B, 64, 128, 128)
        x = self.up3(x)             # (B, 64, 256, 256)
        x = self.pred(x)            # (B, 1, 256, 256)

        return x


# Alias for backward compatibility
UNetDecoder = IntensityDecoder
