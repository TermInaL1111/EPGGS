"""
VGGT Wrapper: Bypass DINOv2 patch_embed, inject pseudo tokens directly.

The VGGT aggregator normally does:
    images → patch_embed(DINOv2) → tokens → camera/register tokens → alternating attention

We replace the first step:
    event voxel → Student ViT + Projector → pseudo tokens → camera/register tokens → VGGT attention
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple


def slice_expand_and_flatten(token: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """Helper: expand a (1, 1, D) or (1, N, D) token tensor to (B*S, N, D)."""
    token = token.expand(B, S, -1, -1)
    return token.reshape(B * S, token.shape[-2], token.shape[-1])


class VGGTTokenInjector:
    """
    Wraps VGGT aggregator to accept pre-computed DINOv2-style tokens
    instead of RGB images. Freezes all VGGT weights.
    """

    def __init__(self, aggregator):
        self.agg = aggregator
        self.agg.eval()

        # Freeze everything
        for param in self.agg.parameters():
            param.requires_grad_(False)

        # Cache key attributes
        self.patch_start_idx = self.agg.patch_start_idx  # number of special tokens
        self.depth = self.agg.depth
        self.aa_order = self.agg.aa_order
        self.aa_block_size = self.agg.aa_block_size
        self.aa_block_num = self.depth // self.aa_block_size

    @torch.no_grad()
    def forward(
        self,
        pseudo_tokens: torch.Tensor,
        B: int,
        S: int,
        image_hw: Tuple[int, int],
        intermediate_layer_idx: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            pseudo_tokens: (B*S, N_tokens, 1024) — output of Student+Projector
            B: batch size
            S: number of views (sequence length)
            image_hw: (H, W) of original images for positional encoding
            intermediate_layer_idx: which layers to return tokens from (for DPT)

        Returns:
            (aggregated_tokens_list, patch_start_idx)
        """
        H, W = image_hw
        N = pseudo_tokens.shape[1]

        # Build camera and register tokens (copied from VGGT, moved to device)
        device = pseudo_tokens.device
        camera_token = slice_expand_and_flatten(
            self.agg.camera_token.to(device), B, S
        )
        register_token = slice_expand_and_flatten(
            self.agg.register_token.to(device), B, S
        )

        # Concatenate special tokens with pseudo tokens
        tokens = torch.cat([camera_token, register_token, pseudo_tokens], dim=1)
        _, P, C = tokens.shape

        # Positional encoding
        pos = None
        if self.agg.rope is not None:
            pos = self.agg.position_getter(
                B * S, H // self.agg.patch_size, W // self.agg.patch_size,
                device=device
            )
            # Special tokens get zero position
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Set up layer tracking
        if intermediate_layer_idx is not None:
            required_layers = set(intermediate_layer_idx)
            required_layers.add(self.depth - 1)  # always keep last layer
        else:
            required_layers = set()

        # ── Alternating Frame/Global Attention ──
        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx = self._run_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx = self._run_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )

            # Collect intermediate outputs
            if required_layers and (frame_idx + global_idx - 1) in required_layers:
                output_list.append(tokens.clone())

        # Ensure last layer is always included
        if not output_list or (frame_idx + global_idx - 1) not in required_layers:
            output_list.append(tokens.clone())

        return output_list, self.patch_start_idx

    def _run_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """Run one frame attention layer."""
        tokens_reshaped = tokens.view(B, S, P, C)
        tokens_reshaped = tokens_reshaped.flatten(1, 2)  # (B, S*P, C)

        # Apply frame attention block
        block = self.agg.frame_blocks[frame_idx]
        if pos is not None:
            pos_reshaped = pos.view(B, S, P, 2).flatten(1, 2)
            tokens_reshaped = block(tokens_reshaped, pos=pos_reshaped)
        else:
            tokens_reshaped = block(tokens_reshaped)

        tokens = tokens_reshaped.view(B * S, P, C)
        return tokens, frame_idx + 1

    def _run_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """Run one global attention layer."""
        tokens_reshaped = tokens.view(1, B * S * P, C)

        block = self.agg.global_blocks[global_idx]
        if pos is not None:
            pos_reshaped = pos.view(1, B * S * P, 2)
            tokens_reshaped = block(tokens_reshaped, pos=pos_reshaped)
        else:
            tokens_reshaped = block(tokens_reshaped)

        tokens = tokens_reshaped.view(B * S, P, C)
        return tokens, global_idx + 1
