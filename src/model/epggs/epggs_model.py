"""
EPGGS: Event-based Generalizable Gaussian Splatting for Pose-Free 3D Reconstruction.

Architecture:
    Event Voxel → Student ViT (trainable) → Projector (trainable) → Pseudo DINOv2 Tokens
    → VGGT Aggregator (frozen) → Camera Head / Depth Head / Intensity Head / Gaussian Head
"""
import torch
import torch.nn as nn
from typing import Optional

# REALM-style Student ViT + Projector (we'll refactor into separate files later)
from src.model.epggs.student_encoder import StudentEncoder


class EPGGSModel(nn.Module):
    def __init__(
        self,
        # Student encoder params (REALM-style)
        student_image_size: int = 336,
        student_patch_size: int = 14,
        student_embed_dim: int = 768,
        student_depth: int = 12,
        student_num_heads: int = 12,
        # Projector params
        projector_num_blocks: int = 2,
        projector_scale: float = 1.0,
        # VGGT is loaded separately via huggingface
    ):
        super().__init__()

        # ── Stage 1: Student Encoder (trainable) ──
        self.student = StudentEncoder(
            image_size=student_image_size,
            patch_size=student_patch_size,
            embed_dim=student_embed_dim,
            depth=student_depth,
            num_heads=student_num_heads,
            in_chans=8,  # 3ch event_frame + 5ch event_voxel (EvGGS format)
        )

        # ── Stage 2: Projector (trainable) ──
        self.projector = nn.Sequential(
            nn.Linear(student_embed_dim, 1024),  # Map to DINOv2 dimension
            nn.LayerNorm(1024),
        )
        self.proj_scale = nn.Parameter(torch.ones(1) * projector_scale)

        # ── Stage 3: VGGT (frozen, loaded later) ──
        # Will be set by load_pretrained()
        self.aggregator = None     # VGGT alternating-attention
        self.camera_head = None    # Pose prediction head
        self.depth_head = None     # Depth prediction head

        # ── Stage 4: EPGGS Heads (trainable) ──
        # Intensity head: predicts grayscale from VGGT tokens
        self.intensity_head = None  # Will be built after VGGT dims are known
        # Gaussian head: predicts R, S, α from depth+intensity+VGGT features
        self.gaussian_head = None   # Adapted from EvGGS GSRegressor

        self._vggt_loaded = False

    def load_pretrained_vggt(self):
        """Load frozen VGGT-1B from HuggingFace."""
        from src.model.encoder.vggt.models.vggt import VGGT
        from src.model.epggs.vggt_wrapper import VGGTTokenInjector

        print("Loading VGGT-1B from HuggingFace...")
        vggt = VGGT.from_pretrained("facebook/VGGT-1B")

        # Freeze VGGT aggregator
        self.aggregator = vggt.aggregator
        for param in self.aggregator.parameters():
            param.requires_grad = False

        # Create token injector (bypasses internal DINOv2 patch_embed)
        self.vggt_injector = VGGTTokenInjector(self.aggregator)

        self.camera_head = vggt.camera_head
        # Camera head IS trainable (fine-tuned on Ev3D-S poses)

        self.depth_head = vggt.depth_head
        for param in self.depth_head.parameters():
            param.requires_grad = False  # Can unfreeze later if needed

        # DPT intermediate layers (like AnySplat, 4 layers for multi-scale features)
        self.intermediate_layers = [5, 11, 17, 23]

        self._vggt_loaded = True
        self._vggt_dim = 1024
        print("VGGT-1B loaded and frozen.")

    def build_heads(self, patch_h: int, patch_w: int):
        """Build EPGGS-specific heads that AnySplat doesn't have."""
        # Intensity head: VGGT tokens → grayscale image
        from src.model.epggs.heads.intensity_head import IntensityDecoder
        self.intensity_head = IntensityDecoder(
            vggt_dim=1024,
            output_channels=1,
            patch_h=patch_h,
            patch_w=patch_w,
        )

        # Gaussian head: depth(1ch) + intensity(1ch) + VGGT_final_token → R,S,α
        from src.model.epggs.heads.gaussian_head import EPGGSGaussianHead
        self.gaussian_head = EPGGSGaussianHead()

    def forward_student_projector(self, event_voxel: torch.Tensor) -> torch.Tensor:
        """
        Event voxel → Student ViT → Projector → Pseudo DINOv2 tokens.

        Args:
            event_voxel: (B, V, C, H, W) where V = number of views, C = 8 channels

        Returns:
            pseudo_tokens: (B*V, N, 1024) DINOv2-style tokens
        """
        B, V, C, H, W = event_voxel.shape
        x = event_voxel.view(B * V, C, H, W)

        # Student ViT forward
        student_tokens = self.student(x)  # (B*V, N, 768)

        # Projector to DINOv2 space
        pseudo_tokens = self.projector(student_tokens) * self.proj_scale  # (B*V, N, 1024)

        return pseudo_tokens

    def forward_vggt(self, pseudo_tokens: torch.Tensor, B: int, V: int):
        """
        Pseudo tokens → VGGT Aggregator → Camera + Depth + EPGGS Heads.

        We bypass VGGT's internal DINOv2 patch_embed and directly inject
        our pseudo tokens into the aggregator's attention layers.

        Args:
            pseudo_tokens: (B*V, N, 1024)
            B, V: batch size, views per scene
        """
        if not self._vggt_loaded:
            raise RuntimeError("Call load_pretrained_vggt() first!")

        device = pseudo_tokens.device
        N = pseudo_tokens.shape[1]

        # ── VGGT Aggregator (skip patch_embed, directly into attention) ──
        # We need to replicate how VGGT processes tokens after patch_embed.
        # The aggregator expects tokens ready for alternating attention.
        # We add camera token + register tokens manually.

        # Camera token (per view, learnable, part of VGGT)
        # Recreate what __build_patch_embed__ does for tokens
        # For now, use the aggregator's existing token infrastructure
        # by calling a modified forward that accepts pre-computed tokens.

        # TODO: This is the key integration point. We'll need to either:
        #   A) Replace aggregator.patch_embed with identity, call aggregator.forward(images_as_tokens)
        #   B) Call aggregator's internal forward directly with pre-made tokens
        # Option B is cleaner.

        pseudo_tokens = pseudo_tokens.view(B, V, N, -1)

        # Placeholder: this will call aggregator's frame/global attention
        # with our pseudo tokens instead of DINOv2-generated tokens.
        aggregated_tokens, camera_tokens = self._run_aggregator(pseudo_tokens)

        # ── Heads ──
        # Pose: from camera tokens
        poses = self.camera_head(camera_tokens)  # (B*V, pose_params)

        # Depth: from aggregated image tokens
        depths = self.depth_head(aggregated_tokens)  # (B*V, 1, H, W)

        # Intensity: from aggregated image tokens
        intensities = self.intensity_head(aggregated_tokens)  # (B*V, 1, H, W)

        # Gaussian: from depth + intensity + aggregated features
        gs_params = self.gaussian_head(
            depths, intensities, aggregated_tokens
        )

        return {
            "poses": poses,
            "depths": depths,
            "intensities": intensities,
            "gs_params": gs_params,
        }

    def _run_aggregator(self, pseudo_tokens: torch.Tensor):
        """
        Run VGGT aggregator with pre-computed pseudo DINOv2 tokens.
        Uses VGGTTokenInjector to bypass the internal patch_embed.
        """
        if self.vggt_injector is None:
            raise RuntimeError("Call load_pretrained_vggt() first!")

        B, V, N, D = pseudo_tokens.shape
        pseudo_tokens_flat = pseudo_tokens.view(B * V, N, D)

        # Run VGGT alternating attention
        aggregated_list, patch_start = self.vggt_injector.forward(
            pseudo_tokens=pseudo_tokens_flat,
            B=B,
            S=V,
            image_hw=(336, 336),  # student image size
            intermediate_layer_idx=self.intermediate_layers,
        )

        # aggregated_list: list of (B*V, total_tokens, D) at selected layers
        # Split into camera tokens and image tokens
        final_tokens = aggregated_list[-1]  # (B*V, total_tokens, 1024)
        camera_tokens = final_tokens[:, 0, :]  # First token is camera token

        # Image tokens (after camera + register tokens)
        image_tokens = final_tokens[:, patch_start:, :]  # (B*V, N_patches, 1024)

        return image_tokens, camera_tokens, aggregated_list

    def forward(self, batch: dict) -> dict:
        """
        Full forward pass.

        batch contains:
            'event_voxel': (B, V, 8, H, W) — 8ch = 3ch event_frame + 5ch voxel bins
            'gt_depth':    (B, V, 1, H, W)
            'gt_gray':     (B, V, 1, H, W)
            'gt_pose':     (B, V, ...)
            'K':           camera intrinsics
        """
        event_voxel = batch['event_voxel']
        B, V = event_voxel.shape[:2]

        # Step 1: Event → Pseudo DINOv2 tokens
        pseudo_tokens = self.forward_student_projector(event_voxel)

        # Step 2: Pseudo tokens → VGGT → Heads
        outputs = self.forward_vggt(pseudo_tokens, B, V)

        return outputs
