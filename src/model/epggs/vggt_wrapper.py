"""
VGGT Wrapper: Bypass DINOv2 patch_embed, inject pseudo tokens directly.

The VGGT aggregator normally does:
    images → patch_embed(DINOv2) → tokens → camera/register tokens → alternating attention

We replace the first step:
    event voxel → Student ViT + Projector → pseudo tokens → camera/register tokens → VGGT attention

IMPORTANT: This replicates the exact frame/global attention logic from the original
VGGT aggregator, including the frame+global intermediate concatenation.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple


def slice_expand_and_flatten(token: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """
    Process VGGT special tokens with shape (1, 2, X, C):
    1) First position (index=0): used for the first frame only
    2) Second position (index=1): used for all remaining frames (S-1 frames)
    3) Concatenates to form (B, S, X, C) then flattens to (B*S, X, C)

    If token has shape (1, 1, ...) or (1, S, ...), use simple expand.
    """
    if token.shape[1] == 2 and S > 1:
        query = token[:, 0:1, ...].expand(B, 1, *token.shape[2:])
        others = token[:, 1:, ...].expand(B, S - 1, *token.shape[2:])
        combined = torch.cat([query, others], dim=1)
        return combined.reshape(B * S, *combined.shape[2:])
    else:
        token = token.expand(B, S, -1, -1)
        return token.reshape(B * S, token.shape[-2], token.shape[-1])


class VGGTTokenInjector:
    """
    Wraps VGGT aggregator to accept pre-computed DINOv2-style tokens
    instead of RGB images. Freezes all VGGT weights.

    Replicates the exact alternating-attention logic from Aggregator.forward():
    - Frame blocks operate on (B*S, P, C) format
    - Global blocks operate on (B, S*P, C) format
    - Frame and global intermediates are concatenated along dim=-1 (→ 2048)
    """

    def __init__(self, aggregator):
        self.agg = aggregator
        self.agg.eval()

        for param in self.agg.parameters():
            param.requires_grad_(False)

        self.patch_start_idx = self.agg.patch_start_idx  # 5
        self.depth = self.agg.depth                        # 24
        self.aa_order = self.agg.aa_order                  # ['frame', 'global']
        self.aa_block_size = self.agg.aa_block_size        # 1
        self.aa_block_num = self.depth // self.aa_block_size  # 24
        self.use_checkpoint = getattr(self.agg, 'use_checkpoint', False)

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
            S: number of views
            image_hw: (H, W) for positional encoding
            intermediate_layer_idx: which layers to return (0-indexed)

        Returns:
            (output_list, patch_start_idx)
            output_list[i]: (B, S, total_tokens, 2048) — frame+global concat
        """
        H, W = image_hw
        device = pseudo_tokens.device

        # ── Build tokens (camera + register + pseudo patches) ──
        camera_token = slice_expand_and_flatten(
            self.agg.camera_token.to(device), B, S
        )   # (B*S, 1, 1024)
        register_token = slice_expand_and_flatten(
            self.agg.register_token.to(device), B, S
        )   # (B*S, 4, 1024)

        tokens = torch.cat([camera_token, register_token, pseudo_tokens], dim=1)
        _, P, C = tokens.shape  # P = total_tokens (1+4+N), C = 1024

        # ── Positional encoding ──
        pos = None
        if self.agg.rope is not None:
            pos = self.agg.position_getter(
                B * S, H // self.agg.patch_size, W // self.agg.patch_size,
                device=device
            )
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2,
                device=device, dtype=pos.dtype
            )
            pos = torch.cat([pos_special, pos], dim=1)  # (B*S, P, 2)

        # ── Alternating Frame/Global Attention ──
        frame_idx = 0
        global_idx = 0
        output_list = []

        # Setup required layer tracking
        if intermediate_layer_idx is not None:
            required_layers = set(intermediate_layer_idx)
            required_layers.add(self.depth - 1)  # always keep last
        else:
            required_layers = None

        layer_idx = 0

        for _ in range(self.aa_block_num):
            # Process frame attention blocks
            tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                tokens, B, S, P, C, frame_idx, pos=pos
            )
            # Process global attention blocks
            tokens, global_idx, global_intermediates = self._process_global_attention(
                tokens, B, S, P, C, global_idx, pos=pos
            )

            # Concatenate frame and global intermediates → 2048-dim
            if required_layers is not None:
                for i in range(len(frame_intermediates)):
                    current_layer = layer_idx + i
                    if current_layer in required_layers:
                        concat_inter = torch.cat(
                            [frame_intermediates[i], global_intermediates[i]], dim=-1
                        )  # (B, S, P, 2C) = (B, S, P, 2048)
                        output_list.append(concat_inter)
                layer_idx += self.aa_block_size
            else:
                # No filtering: collect every layer
                for i in range(len(frame_intermediates)):
                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )
                    output_list.append(concat_inter)

        return output_list, self.patch_start_idx

    # ═══════════════════════════════════════════════════════
    # Frame Attention: tokens in (B*S, P, C), pos in (B*S, P, 2)
    # ═══════════════════════════════════════════════════════

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """Exactly matches Aggregator._process_frame_attention."""
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = torch.utils.checkpoint.checkpoint(
                    self.agg.frame_blocks[frame_idx],
                    tokens,
                    pos,
                    use_reentrant=False,
                )
            else:
                tokens = self.agg.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    # ═══════════════════════════════════════════════════════
    # Global Attention: tokens in (B, S*P, C), pos in (B, S*P, 2)
    # ═══════════════════════════════════════════════════════

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """Exactly matches Aggregator._process_global_attention."""
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = torch.utils.checkpoint.checkpoint(
                    self.agg.global_blocks[global_idx],
                    tokens,
                    pos,
                    use_reentrant=False,
                )
            else:
                tokens = self.agg.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates
