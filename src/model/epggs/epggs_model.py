"""
EPGGS: Event-based Generalizable Gaussian Splatting for Pose-Free 3D Reconstruction.

Architecture:
    Event Voxel (5ch) → REALM encoder_ev (frozen) → 768D tokens
    → dino2reg projector (frozen) → pseudo DINOv2 tokens (1024D)
    → VGGT Aggregator (frozen) → 2048D frame+global tokens
    → F_C Pose Head (trainable) + Depth Head (frozen) + Intensity/Gaussian Heads (trainable)
"""
import torch
import torch.nn as nn
from typing import Optional


class EPGGSModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ── Stage 1: REALM encoder_ev (frozen) + projector (frozen) ──
        self.realm = None           # REALM model (encoder_ev)
        self.projector = None       # dino2reg_vitlarge_14: 768→1024
        self._realm_loaded = False

        # ── Stage 2: VGGT (frozen) ──
        self.aggregator = None
        self.camera_head = None
        self.depth_head = None
        self.vggt_injector = None
        self.intermediate_layers = [5, 11, 17, 23]

        # ── Stage 3: EPGGS Heads (trainable) ──
        self.intensity_head = None
        self.gaussian_head = None

        self._vggt_loaded = False
        self._heads_built = False

    # ═══════════════════════════════════════════════
    # Loading methods
    # ═══════════════════════════════════════════════

    def load_pretrained_realm(self):
        """Load REALM encoder_ev + dino2reg projector. No task head."""
        import os
        from realm.model_factory import REALM_creator
        from realm.dune.dune import load_dune_from_checkpoint

        print("Loading REALM encoder_ev from HF (viciopoli/REALM)...")
        # Load mast3r config to get encoder_ev (but we won't use the head)
        self.realm = REALM_creator("mast3r")
        for param in self.realm.parameters():
            param.requires_grad = False

        # Load dino2reg_vitlarge_14 projector (768→1024)
        print("Loading dino2reg projector...")
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '../../../../REALM/realm/checkpoints/dune_vitbase14_448_paper.pth')
        # Try absolute path relative to known location
        for candidate in [
            ckpt_path,
            '/root/REALM/realm/checkpoints/dune_vitbase14_448_paper.pth',
            './checkpoints/dune_vitbase14_448_paper.pth',
        ]:
            if os.path.exists(candidate):
                dune, _ = load_dune_from_checkpoint(candidate)
                break
        else:
            # Download from HF if not found
            from huggingface_hub import hf_hub_download
            path = hf_hub_download('viciopoli/REALM', 'checkpoints/dune_vitbase14_448_paper.pth')
            dune, _ = load_dune_from_checkpoint(path)

        self.projector = dune.projectors['dino2reg_vitlarge_14']
        for param in self.projector.parameters():
            param.requires_grad = False

        # Move to GPU
        self.realm = self.realm.cuda()
        self.projector = self.projector.cuda()

        self._realm_loaded = True
        print(f"REALM encoder_ev + projector ({self.projector.input_dim}→{self.projector.output_dim}) loaded & frozen.")

    def load_pretrained_vggt(self):
        """Load frozen VGGT-1B from HuggingFace."""
        from src.model.encoder.vggt.models.vggt import VGGT
        from src.model.epggs.vggt_wrapper import VGGTTokenInjector

        print("Loading VGGT-1B from HuggingFace...")
        vggt = VGGT.from_pretrained("facebook/VGGT-1B")

        self.aggregator = vggt.aggregator.cuda()
        for param in self.aggregator.parameters():
            param.requires_grad = False

        self.vggt_injector = VGGTTokenInjector(self.aggregator)

        self.camera_head = vggt.camera_head.cuda()
        # F_C is trainable (fine-tuned on Ev3D-S poses)

        self.depth_head = vggt.depth_head.cuda()
        for param in self.depth_head.parameters():
            param.requires_grad = False

        self._vggt_loaded = True
        print("VGGT-1B loaded and frozen.")

    def build_heads(self, patch_h: int, patch_w: int):
        """Build trainable EPGGS heads (on GPU)."""
        from src.model.epggs.heads.intensity_head import UNetDecoder as IntensityDecoder
        from src.model.epggs.heads.gaussian_head import EPGGSGaussianHead

        self.intensity_head = IntensityDecoder(
            vggt_dim=2048,
            output_channels=1,
        ).cuda()
        self.gaussian_head = EPGGSGaussianHead(input_dim=42).cuda()

        self._heads_built = True
        self._patch_h = patch_h
        self._patch_w = patch_w

    # ═══════════════════════════════════════════════
    # Forward methods
    # ═══════════════════════════════════════════════

    def forward_student_projector(self, event_voxel: torch.Tensor) -> torch.Tensor:
        """
        Event voxel → REALM encoder_ev → projector → pseudo DINOv2 tokens.

        Uses REALM._encode to get 768D tokens, then applies the 768→1024 projector.
        Returns (B*V, N_patches, 1024) — patch tokens only (no CLS).
        """
        if not self._realm_loaded:
            raise RuntimeError("Call load_pretrained_realm() first!")

        B, V, C, H, W = event_voxel.shape
        x = event_voxel.view(B * V, C, H, W)

        with torch.no_grad():
            # Encode events → 768D tokens
            features, _, _ = self.realm._encode(x)
            tokens_768 = features['x_norm_patchtokens']  # (B*V, N, 768)

            # Project to DINOv2 1024D space
            pseudo_tokens = self.projector(tokens_768)  # (B*V, N, 1024)

        return pseudo_tokens

    def forward_vggt(self, pseudo_tokens: torch.Tensor, B: int, V: int):
        """
        Pseudo DINOv2 tokens → VGGT Aggregator → Heads.

        Args:
            pseudo_tokens: (B*V, N, 1024)
            B, V: batch size, views per scene
        """
        if not self._vggt_loaded or not self._heads_built:
            raise RuntimeError("Call load_pretrained_vggt() and build_heads() first!")

        # ── VGGT Aggregator (bypasses patch_embed, injects our tokens) ──
        with torch.no_grad():
            aggregated_list, patch_start = self.vggt_injector.forward(
                pseudo_tokens=pseudo_tokens,
                B=B, S=V,
                image_hw=(448, 448),  # REALM input size
                intermediate_layer_idx=self.intermediate_layers,
            )
        # aggregated_list[i]: (B, V, P, 2048) where P = total_tokens (5+N_patches)

        # ── Pose Head (F_C, trainable) ──
        # camera_head expects list of (B, V, P, 2048) tensors
        pose_enc_list = self.camera_head(aggregated_list)
        poses = pose_enc_list[-1]  # (B, V, 9) — final iteration

        # ── Depth Head (frozen) ──
        # depth_head expects list of (B, V, P, 2048) + images tensor for shape info
        # We create a dummy image tensor for shape reference
        dummy_images = torch.zeros(B, V, 3, 448, 448, device=pseudo_tokens.device)
        with torch.no_grad():
            depths, depth_conf = self.depth_head(
                aggregated_list, images=dummy_images,
                patch_start_idx=patch_start
            )
        # depths: (B, V, H, W, 1), depth_conf: (B, V, H, W)

        # ── Intensity Head (trainable) ──
        # Take final-layer image tokens: (B, V, N_patches, 2048)
        final_tokens = aggregated_list[-1]  # (B, V, P, 2048)
        image_tokens = final_tokens[:, :, patch_start:, :]  # (B, V, N_patches, 2048)

        # Flatten to (B*V, N_patches, 2048) for intensity head
        image_tokens_flat = image_tokens.reshape(B * V, -1, 2048)

        intensities = self.intensity_head(image_tokens_flat)  # (B*V, 1, 256, 256)

        # ── Gaussian Head (trainable) ──
        # Prepare input: depth(1) + intensity(1) + vggt_feat(40)
        intensity_resized = nn.functional.interpolate(
            intensities, size=(128, 128), mode='bilinear'
        )
        depth_for_gs = depths.reshape(B * V, 1, 448, 448)
        depth_for_gs = nn.functional.interpolate(
            depth_for_gs, size=(128, 128), mode='bilinear'
        )

        vggt_feat = image_tokens.permute(0, 1, 3, 2).reshape(B * V, 2048, self._patch_h, self._patch_w)
        vggt_feat = nn.functional.interpolate(
            vggt_feat[:, :40, :, :], size=(128, 128), mode='bilinear'
        )

        gs_input = torch.cat([depth_for_gs, intensity_resized, vggt_feat], dim=1)
        rot, scale, opacity = self.gaussian_head(gs_input)

        return {
            "poses": poses,
            "depths": depths,
            "depth_conf": depth_conf,
            "intensities": intensities,
            "rot": rot,
            "scale": scale,
            "opacity": opacity,
        }

    def forward(self, batch: dict) -> dict:
        """
        Full forward pass.

        batch keys:
            'event_voxel': (B, V, 5, H, W) — 5ch voxel grid (REALM format)
            'gt_gray':     (B, V, 1, H, W)
            'gt_depth':    (B, V, 1, H, W)
            'gt_pose':     (B, V, 4, 4)
            'K':           (3, 3)
        """
        event_voxel = batch['event_voxel']
        B, V = event_voxel.shape[:2]

        # Step 1: Event → Pseudo DINOv2 tokens
        pseudo_tokens = self.forward_student_projector(event_voxel)

        # Step 2: Pseudo tokens → VGGT → Heads
        outputs = self.forward_vggt(pseudo_tokens, B, V)

        return outputs
