"""
Intensity Head: EvGGS-style UNet Decoder for grayscale prediction.

VGGT intermediate tokens → multi-scale features (skip connections)
→ UNet Decoder upsampling path → grayscale image.

Structure mirrors EvGGS E2IM's UNet decoder:
    github.com/Mercerai/EvGGS/blob/main/lib/network/unet.py
"""
import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """Conv2d + optional BatchNorm + activation (from EvGGS submodules.py)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, activation='relu', norm=None):
        super().__init__()
        bias = norm is None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels) if norm == 'BN' else nn.Identity()
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class UpsampleConvLayer(nn.Module):
    """Upsample + Conv2d (from EvGGS submodules.py, avoids checkerboard)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, activation='relu', norm='BN'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvLayer(in_channels, out_channels, kernel_size,
                              stride, padding, activation, norm)

    def forward(self, x):
        return self.conv(self.upsample(x))


class UNetDecoder(nn.Module):
    """
    UNet Decoder (upsampling path) matching EvGGS structure.

    Args:
        vggt_dim: VGGT token dimension (1024 for VGGT-1B)
        base_channels: starting channel count (32, matching EvGGS)
        num_upsample: number of upsampling stages (4 in EvGGS)
        output_channels: 1 for grayscale
        skip_channels: list of channel counts from VGGT intermediate layers
    """

    def __init__(
        self,
        vggt_dim=1024,
        base_channels=32,
        num_upsample=4,
        output_channels=1,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.num_upsample = num_upsample

        # VGGT token → 2D bottleneck
        # bottleneck channels = base * 2^(num_upsample-1) * 2 = 32*8*2 = 512
        bottleneck_ch = base_channels * pow(2, num_upsample)

        # Map VGGT dim to bottleneck channel count
        self.token_proj = nn.Sequential(
            nn.Linear(vggt_dim, bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder: progressive upsampling
        # Each stage: (skip_in + decoder_in) → UpsampleConv → half channels
        decoder_channels = [base_channels * pow(2, i + 1) for i in range(num_upsample)]
        # reversed: [512, 256, 128, 64] for num_upsample=4, base=32

        self.decoders = nn.ModuleList()
        for ch in reversed(decoder_channels):
            # input = 2*ch (skip concat: same-level encoder + previous decoder)
            self.decoders.append(
                UpsampleConvLayer(
                    in_channels=2 * ch,
                    out_channels=ch // 2,
                    kernel_size=5,
                    padding=2,
                    activation='relu',
                    norm='BN',
                )
            )

        # Prediction layer
        self.pred = ConvLayer(
            base_channels * 2,  # skip_sum with head (base*2 = 64)
            output_channels,
            kernel_size=1,
            activation='sigmoid',
        )

    def forward(self, vggt_tokens, skip_features):
        """
        Args:
            vggt_tokens: (B, N, vggt_dim) — VGGT bottleneck tokens
            skip_features: list of (B, C_i, H_i, W_i) — from VGGT intermediate layers
                           Must have num_upsample items, deepest first

        Returns:
            image: (B, output_channels, H, W) — grayscale image [0, 1]
        """
        B, N, D = vggt_tokens.shape

        # Estimate spatial dimensions from token count
        patch_res = int(N ** 0.5)  # e.g. sqrt(256) = 16 for 224/14

        # Global average pool VGGT tokens → 1×1 bottleneck
        bottleneck = self.token_proj(vggt_tokens.mean(dim=1))  # (B, bottleneck_ch)
        x = bottleneck.view(B, -1, 1, 1)  # (B, bottleneck_ch, 1, 1)

        # Resize skip features to proper resolutions
        decoder_inputs = []
        for i, skip in enumerate(skip_features):
            target_size = patch_res * pow(2, self.num_upsample - i - 1)
            if skip.shape[-1] != target_size:
                skip = torch.nn.functional.interpolate(
                    skip, size=(target_size, target_size), mode='bilinear', align_corners=False
                )
            decoder_inputs.append(skip)

        # Decode: upsample + skip connect
        head_out = None
        for i, decoder in enumerate(self.decoders):
            x = decoder(torch.cat([x, decoder_inputs[self.num_upsample - 1 - i]], dim=1))
            if i == 0:
                head_out = x  # save first decoder output for head skip

        # Prediction
        if head_out is not None:
            x = torch.cat([x, head_out], dim=1)
        else:
            # No head skip, duplicate the feature
            x = torch.cat([x, x], dim=1)

        return self.pred(x)
