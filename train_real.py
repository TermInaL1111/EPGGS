#!/usr/bin/env python3
"""
EPGGS Training Script with REALM + VGGT-1B pretrained weights.

Usage:
    python train_real.py --data_root /root/dataset/EV3DS --epochs 50 --output_dir ./output
"""
import os, sys, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# Path setup — REALM expects cwd in its repo
REALM_DIR = '/root/REALM/realm'
os.chdir(REALM_DIR)
sys.path.insert(0, '/root/EPGGS')

from src.model.epggs.epggs_model import EPGGSModel
from src.dataset.ev3d_real import Ev3DDatasetReal


# ═══════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════

def compute_pose_loss(pred_pose, gt_pose):
    """L2 loss on translation + rotation components."""
    # pred_pose: (B, V, 9) — absT_quaR_FoV encoding
    # gt_pose:   (B, V, 4, 4) — world-to-camera matrices
    # For simplicity: L2 on translation + L2 on rotation 6D
    trans_pred = pred_pose[..., :3]
    # GT translation from 4×4 matrix
    trans_gt = gt_pose[..., :3, 3]  # (B, V, 3)
    return F.mse_loss(trans_pred, trans_gt)

def compute_depth_loss(pred_depth, gt_depth):
    """L1 loss on depth, masked by valid GT.
    pred_depth: (B, V, H, W, 1) or (B, V, H, W)
    gt_depth:   (B, V, 1, H, W)
    """
    pred = pred_depth.squeeze(-1)  # (B, V, H, W)
    gt = gt_depth.squeeze(2)       # (B, V, H, W)
    valid = (gt > 0.1) & (gt < 100.0)
    if valid.sum() < 10:
        return torch.tensor(0.0, device=pred.device)
    return F.l1_loss(pred[valid], gt[valid])

def compute_intensity_loss(pred_intensity, gt_gray):
    """L1 loss on predicted vs GT grayscale."""
    # pred_intensity: (B*V, 1, 256, 256)
    # gt_gray: (B, V, 1, 448, 448)
    B, V = gt_gray.shape[:2]
    gt_resized = F.interpolate(
        gt_gray.view(B*V, 1, 448, 448),
        size=pred_intensity.shape[-2:], mode='bilinear'
    )
    return F.l1_loss(pred_intensity, gt_resized)


# ═══════════════════════════════════════════
# Training
# ═══════════════════════════════════════════

def train(args):
    device = torch.device('cuda')
    print(f"Device: {device}")

    # ── Model ──
    print("Building model...")
    model = EPGGSModel()
    model.load_pretrained_realm()
    model.load_pretrained_vggt()
    model.build_heads(patch_h=32, patch_w=32)  # 448/14

    # ── Set trainable parameters ──
    # REALM + VGGT aggregator + depth_head are frozen (set in load methods)
    # camera_head, intensity_head, gaussian_head are trainable
    # Ensure all params on GPU
    model = model.to(device)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable params ({len(trainable)} groups):")
    for n in trainable:
        print(f"  {n}: {sum(1 for _ in model.get_parameter(n).view(-1))}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # ── Data ──
    dataset = Ev3DDatasetReal(
        root=args.data_root,
        split='train',
        image_size=448,
        num_views=args.num_views,
    )
    # Subsample for faster iterations
    if args.max_samples > 0 and args.max_samples < len(dataset):
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"Subsampled to {len(dataset.samples)} views ({len(dataset)} samples)")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logger = open(os.path.join(args.output_dir, 'train.log'), 'w')

    # ── Training Loop ──
    model.train()
    global_step = 0

    print("Starting training...", flush=True)
    logger.write("Starting training...\n"); logger.flush()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            B, V = batch['event_voxel'].shape[:2]

            # ── Forward ──
            outputs = model.forward(batch)

            # ── Losses ──
            loss_pose = compute_pose_loss(
                outputs['poses'], batch['gt_pose']
            )
            loss_depth = compute_depth_loss(
                outputs['depths'].squeeze(-1), batch['gt_depth'].squeeze(1)
            )
            loss_intensity = compute_intensity_loss(
                outputs['intensities'], batch['gt_gray']
            )

            # Total loss
            loss = loss_pose * 1.0 + loss_depth * 0.1 + loss_intensity * 0.5

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            global_step += 1

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                msg = (f"E{epoch:3d} B{batch_idx:4d} | "
                       f"pose={loss_pose.item():.4f} "
                       f"depth={loss_depth.item():.4f} "
                       f"intensity={loss_intensity.item():.4f} | "
                       f"total={loss.item():.4f}")
                print(msg, flush=True)
                logger.write(msg + '\n'); logger.flush()

        dt = time.time() - t0
        avg_loss = epoch_loss / len(dataloader)
        msg = f"=== Epoch {epoch:3d} | Loss={avg_loss:.4f} | Time={dt:.1f}s ==="
        print(msg, flush=True)
        logger.write(msg + '\n'); logger.flush()

        # Checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.output_dir, f"epggs_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EPGGS Training')
    parser.add_argument('--data_root', type=str, default='/root/dataset/EV3DS')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_views', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Max samples per epoch (-1 = all, 2000 for ~12min/epoch)')
    args = parser.parse_args()

    train(args)
