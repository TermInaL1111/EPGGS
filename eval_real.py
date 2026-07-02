#!/usr/bin/env python3
"""
EPGGS Evaluation Script.

Tests trained model on validation set (last 20 scenes of Ev3D-S).
Reports depth metrics + intensity quality + pose accuracy.
"""
import sys, os, argparse
os.chdir('/root/REALM/realm')
sys.path.insert(0, '/root/EPGGS')

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from src.model.epggs.epggs_model import EPGGSModel
from src.dataset.ev3d_real import Ev3DDatasetReal


def evaluate(args):
    device = torch.device('cuda')
    torch.cuda.empty_cache()

    # ── Load model ──
    print("Loading model...", flush=True)
    model = EPGGSModel()
    model.load_pretrained_realm()
    model.load_pretrained_vggt()
    model.build_heads(32, 32)
    model = model.to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Epoch: {ckpt.get('epoch', '?')}, Loss: {ckpt.get('loss', '?'):.4f}")
    model.eval()

    # ── Validation data ──
    ds = Ev3DDatasetReal(args.data_root, split='val', num_views=args.num_views)
    if args.max_samples > 0:
        ds.samples = ds.samples[:args.max_samples]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"Evaluating on {len(loader)} batches...", flush=True)

    # ── Metrics ──
    metrics = {
        'pose_mse': [], 'depth_absrel': [], 'intensity_mse': [],
        'pose_err_cm': [],
    }

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            out = model.forward(batch)

            # Pose error (translation L2, cm)
            pred_trans = out['poses'][..., :3]  # (1, V, 3)
            gt_trans = batch['gt_pose'][..., :3, 3]  # (1, V, 3)
            pose_err = (pred_trans - gt_trans).norm(dim=-1).mean().item()
            metrics['pose_err_cm'].append(pose_err * 100)  # m→cm

            # Depth error (AbsRel)
            pred_depth = out['depths'].squeeze(-1)  # (1, V, H, W)
            gt_depth = batch['gt_depth'].squeeze(2)  # (1, V, H, W)
            valid = (gt_depth > 0.5) & (gt_depth < 50.0)
            if valid.sum() > 10:
                absrel = ((pred_depth[valid] - gt_depth[valid]).abs() / gt_depth[valid]).mean()
                metrics['depth_absrel'].append(absrel.item())

            # Intensity quality
            gt_gray = F.interpolate(
                batch['gt_gray'].view(3, 1, 448, 448),
                size=out['intensities'].shape[-2:], mode='bilinear'
            )
            mse = F.mse_loss(out['intensities'], gt_gray).item()
            metrics['intensity_mse'].append(mse)

            if i % 20 == 0:
                print(f"  [{i:4d}/{len(loader)}] "
                      f"pose={pose_err*100:.1f}cm "
                      f"absrel={metrics['depth_absrel'][-1] if metrics['depth_absrel'] else 0:.3f} "
                      f"int_mse={mse:.4f}",
                      flush=True)

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"  EPGGS Evaluation Results")
    print(f"{'='*55}")
    print(f"  Pose error (cm):      {np.mean(metrics['pose_err_cm']):.2f} ± {np.std(metrics['pose_err_cm']):.2f}")
    if metrics['depth_absrel']:
        print(f"  Depth AbsRel:          {np.mean(metrics['depth_absrel']):.4f}")
    print(f"  Intensity MSE:         {np.mean(metrics['intensity_mse']):.4f}")
    print(f"{'='*55}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/dataset/EV3DS')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--num_views', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=50)
    args = parser.parse_args()
    evaluate(args)
