#!/usr/bin/env python3
"""
EPGGS Training Script with 3-Level Alignment.

Level 1: Token Alignment (REALM-style cosine + smooth L1)
Level 2: Geometric Consistency (same 3D point → similar VGGT tokens across views)
Level 3: Downstream Task Feedback (render loss back to Student+Projector)

Usage:
    python train_epggs.py --data_root /path/to/Ev3D-S --batch_size 2 --epochs 100
"""
import os, sys, argparse, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from src.model.epggs import EPGGSModel
from src.dataset.ev3d_dataset import Ev3DDataset  # Step 2: event data loader


# ═══════════════════════════════════════════
# REALM-style alignment losses
# ═══════════════════════════════════════════

def cosine_loss(pred, target, avg=True):
    """1 - cosine_similarity"""
    cos = F.cosine_similarity(pred, target, dim=-1)
    loss = 1.0 - cos
    return loss.mean() if avg else loss


def smooth_l1_loss(pred, target, beta=1.0, avg=True):
    loss = F.smooth_l1_loss(pred, target, reduction='none', beta=beta).mean(dim=-1)
    return loss.mean() if avg else loss


# ═══════════════════════════════════════════
# Level 1: Token Alignment (REALM-style)
# ═══════════════════════════════════════════

def level1_alignment_loss(pseudo_tokens, teacher_tokens, lam_cos=0.5, lam_sl1=0.5):
    """
    Align pseudo DINOv2 tokens with teacher DINOv2 tokens.

    Args:
        pseudo_tokens:  (B, N, 1024) from Student+Projector
        teacher_tokens: (B, N, 1024) from frozen DINOv2 on GT grayscale
    """
    loss_cos = cosine_loss(pseudo_tokens, teacher_tokens, avg=True)
    loss_sl1 = smooth_l1_loss(pseudo_tokens, teacher_tokens, avg=True)
    return lam_cos * loss_cos + lam_sl1 * loss_sl1


# ═══════════════════════════════════════════
# Level 2: Geometric Consistency
# ═══════════════════════════════════════════

def level2_geometry_loss(vggt_tokens_A, vggt_tokens_B, gt_depth_A, K, pose_A_to_B, patch_h, patch_w):
    """
    Same 3D point → similar VGGT tokens across views.

    For pixels in view A with known depth, project to view B,
    then enforce token similarity at the corresponding positions.

    Args:
        vggt_tokens_A: (B, N_patch, 1024) reshaped to (B, 1024, h, w)
        vggt_tokens_B: (B, N_patch, 1024)
        gt_depth_A:    (B, 1, H, W)
        K:             (3, 3) intrinsics
        pose_A_to_B:   (4, 4) relative pose
        patch_h, patch_w: patch grid dimensions
    """
    B, D, h, w = vggt_tokens_A.shape
    H, W = h * 14, w * 14  # patch_size=14

    # Sample points in view A (random subset for efficiency)
    n_pts = min(1000, H * W // 8)
    u_A = torch.randint(0, W, (n_pts,), device=vggt_tokens_A.device)
    v_A = torch.randint(0, H, (n_pts,), device=vggt_tokens_A.device)

    # Get depth at sampled pixels
    depth_vals = gt_depth_A[0, 0, v_A, u_A]
    valid = depth_vals > 0.1

    if valid.sum() < 10:
        return torch.tensor(0.0, device=vggt_tokens_A.device)

    u_A, v_A = u_A[valid], v_A[valid]
    depth_vals = depth_vals[valid]

    # 3D coordinates in view A (simplified pinhole)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X_A = (u_A.float() - cx) / fx * depth_vals
    Y_A = (v_A.float() - cy) / fy * depth_vals
    Z_A = depth_vals
    pts_A = torch.stack([X_A, Y_A, Z_A, torch.ones_like(Z_A)], dim=1)  # (N, 4)

    # Project to view B
    pts_B = (pose_A_to_B @ pts_A.T).T  # (N, 4)
    u_B = (pts_B[:, 0] / pts_B[:, 2] * fx + cx).long()
    v_B = (pts_B[:, 1] / pts_B[:, 2] * fy + cy).long()

    # Keep only points within image bounds
    valid_B = (u_B >= 0) & (u_B < W) & (v_B >= 0) & (v_B < H)
    if valid_B.sum() < 10:
        return torch.tensor(0.0, device=vggt_tokens_A.device)

    u_B, v_B = u_B[valid_B], v_B[valid_B]

    # Token indices
    pu_A, pv_A = u_A[valid_B] // 14, v_A[valid_B] // 14
    pu_B, pv_B = u_B // 14, v_B // 14

    # Token similarity at corresponding positions
    tokens_A_at_pts = vggt_tokens_A[0, :, pv_A, pu_A]  # (D, N_valid)
    tokens_B_at_pts = vggt_tokens_B[0, :, pv_B, pu_B]  # (D, N_valid)

    # Cosine similarity loss
    cos_sim = F.cosine_similarity(tokens_A_at_pts.T, tokens_B_at_pts.T, dim=1)
    return (1.0 - cos_sim).mean()


# ═══════════════════════════════════════════
# Level 3: Downstream Task Loss
# ═══════════════════════════════════════════

def level3_render_loss(rendered_img, gt_gray):
    """L2 + SSIM on grayscale rendering."""
    from pytorch_msssim import ssim
    loss_l2 = F.mse_loss(rendered_img, gt_gray)
    loss_ssim = 1.0 - ssim(rendered_img, gt_gray, data_range=1.0, size_average=True)
    return loss_l2 + 0.2 * loss_ssim


# ═══════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Model ──
    model = EPGGSModel().to(device)

    # Phase 1: ONLY train Student + Projector (alignment)
    # Freeze all heads
    for param in model.parameters():
        param.requires_grad = False
    for param in model.student.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # ── Data ──
    dataset = Ev3DDataset(
        root=args.data_root,
        image_size=336,
        num_views=args.num_views,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # ── Training ──
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            event_voxel = batch['event_voxel']        # (B, V, 8, H, W)
            gt_gray = batch['gt_gray']                 # (B, V, 1, H, W)
            gt_depth = batch['gt_depth']               # (B, V, 1, H, W)
            gt_pose = batch['gt_pose']                 # (B, V, ...)
            K = batch['K']                             # (3, 3)

            B, V = event_voxel.shape[:2]

            # ── Forward: Student + Projector only (Phase 1) ──
            pseudo_tokens = model.forward_student_projector(event_voxel)
            # pseudo_tokens: (B*V, N, 1024)

            # Get teacher tokens from GT grayscale
            teacher_tokens = model.get_teacher_tokens(gt_gray)
            # teacher_tokens: (B*V, N, 1024) — from frozen DINOv2

            # ── Level 1: Token Alignment ──
            loss_l1 = level1_alignment_loss(pseudo_tokens, teacher_tokens)

            # ── Level 2: Geometric Consistency (every K epochs) ──
            loss_l2 = torch.tensor(0.0, device=device)
            if epoch % args.geo_freq == 0 and V > 1:
                pseudo_tokens_2d = pseudo_tokens[:, 1:, :]  # remove CLS
                pseudo_tokens_2d = pseudo_tokens_2d.reshape(
                    B, V, model.patch_h, model.patch_w, -1
                ).permute(0, 1, 4, 2, 3).contiguous()

                for v in range(V - 1):
                    loss_l2 += level2_geometry_loss(
                        pseudo_tokens_2d[:, v],
                        pseudo_tokens_2d[:, v + 1],
                        gt_depth[:, v],
                        K,
                        _compute_relative_pose(gt_pose, v, v + 1),
                        model.patch_h, model.patch_w,
                    )

            # ── Level 3: Task feedback (after Phase 1 warmup) ──
            loss_l3 = torch.tensor(0.0, device=device)
            if epoch > args.warmup_epochs:
                # Unfreeze heads for Phase 3
                if epoch == args.warmup_epochs + 1:
                    _unfreeze_heads(model, optimizer, args.lr * 0.1)

                outputs = model.forward_vggt(pseudo_tokens.view(B, V, -1, 1024), B, V)
                render = _render(outputs, batch)
                loss_l3 = level3_render_loss(render, gt_gray[:, 0])

            # ── Total Loss ──
            loss = loss_l1 + 0.1 * loss_l2 + 0.01 * loss_l3

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch:3d} | Batch {batch_idx:4d} | "
                      f"L1={loss_l1.item():.4f} L2={loss_l2.item():.4f} L3={loss_l3.item():.4f} | "
                      f"Total={loss.item():.4f}")

        # End of epoch
        dt = time.time() - t0
        print(f"=== Epoch {epoch:3d} | Loss={epoch_loss/len(dataloader):.4f} | Time={dt:.1f}s ===")

        # Checkpoint
        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(args.output_dir, f"epggs_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(dataloader),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


def _compute_relative_pose(gt_pose, v_i, v_j):
    """Compute relative pose between two views from GT absolute poses."""
    # Simplified: assume gt_pose is (B, V, 4, 4) world-to-camera
    pose_i = gt_pose[:, v_i]   # (B, 4, 4)
    pose_j = gt_pose[:, v_j]   # (B, 4, 4)
    rel_pose = pose_j @ torch.linalg.inv(pose_i)  # i → j
    return rel_pose[0]  # First batch item


def _render(outputs, batch):
    """Placeholder: rasterize Gaussians → rendered image."""
    # TODO: integrate gsplat CUDA rasterizer
    # For now, placeholder that returns intensities
    return outputs['intensities']


def _unfreeze_heads(model, optimizer, lr):
    """Unfreeze heads for Phase 3 training."""
    trainable = []
    for name, param in model.named_parameters():
        if any(x in name for x in ['camera_head', 'intensity_head', 'gaussian_head']):
            param.requires_grad = True
            trainable.append(param)
    optimizer.add_param_group({'params': trainable, 'lr': lr})
    print(f"Unfrozen heads: {len(trainable)} parameter groups")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to Ev3D-S dataset')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_views', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Phase 1→Phase 3 switch')
    parser.add_argument('--geo_freq', type=int, default=5, help='Level 2 geometry loss frequency')
    parser.add_argument('--save_freq', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
