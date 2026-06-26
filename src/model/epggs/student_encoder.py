"""
Student Encoder: Lightweight ViT that processes event voxels.

Based on REALM's student encoder (DinoVisionTransformer vit_base),
adapted for event data input (8 channels instead of 3 for RGB).

The student is fully trainable — it learns to encode event voxels
into tokens that a Projector maps to DINOv2 latent space.
"""
import torch
import torch.nn as nn
from functools import partial


class PatchEmbed(nn.Module):
    """Patch embedding for event data (adapted from DINOv2 PatchEmbed)."""

    def __init__(self, img_size=336, patch_size=14, in_chans=8, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, H/patch, W/patch)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Simplified attention (no LoRA needed — student is fully trainable)."""

    def __init__(self, dim, num_heads=12, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Standard ViT block with LayerNorm + Attention + MLP."""

    def __init__(self, dim, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class StudentEncoder(nn.Module):
    """
    Lightweight ViT for event data.

    Config matches REALM's student: vit_base-ish, img=336, patch=14.
    But input channels = 8 (event_frame 3ch + event_voxel 5ch).
    """

    def __init__(
        self,
        image_size=336,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        in_chans=8,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, return_all_tokens=True):
        """
        Args:
            x: (B, C, H, W) event data
            return_all_tokens: if True, return CLS + patch tokens.
                              if False, return CLS only.

        Returns:
            tokens: (B, N+1, embed_dim) if return_all_tokens else (B, embed_dim)
        """
        B = x.shape[0]

        # Patch embed
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if return_all_tokens:
            return x  # (B, N+1, embed_dim) — CLS + patches
        else:
            return x[:, 0]  # (B, embed_dim) — CLS only
