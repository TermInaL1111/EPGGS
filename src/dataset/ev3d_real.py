"""
Ev3D-S Dataset for REALM-format (5ch voxel grid).

Loads the actual Ev3D-S data format:
  EV3DS/Event/{scene}/XXXX.npy  — events (4, N): [x, y, t, p]
  EV3DS/Scenes/{scene}/XXXX.png — grayscale images
  EV3DS/Scenes/{scene}/XXXX.npz — depth_map, optical_flow
  EV3DS/Poses/r_XXXX.txt        — per-scene 4×4 poses (4 lines)

Converts events to 5-bin voxel grid matching REALM format.
"""
import os, glob
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from torch.utils.data import Dataset


class Ev3DDatasetReal(Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = 448,
        num_views: int = 3,
        num_bins: int = 5,
    ):
        self.root = root
        self.image_size = image_size
        self.num_views = num_views
        self.num_bins = num_bins

        # Scene list
        scenes = sorted(os.listdir(os.path.join(root, 'Scenes')))
        # Simple train/val split by scene (first 80 train, last 20 val)
        if split == 'train':
            self.scenes = scenes[:80]
        elif split == 'val':
            self.scenes = scenes[80:]
        else:
            self.scenes = scenes

        # Build sample list: (scene, frame_idx)
        self.samples = []
        for scene in self.scenes:
            png_dir = os.path.join(root, 'Scenes', scene)
            pngs = sorted([f for f in os.listdir(png_dir) if f.endswith('.png') and not f.startswith('._')])
            for png in pngs:
                try:
                    fid = int(png.replace('.png', ''))
                except ValueError:
                    continue
                self.samples.append((scene, fid))

        # Pre-load poses — Ev3D-S: 201 pose files, one per frame index (same for all scenes)
        self.all_poses = {}
        pose_dir = os.path.join(root, 'Poses')
        for pf in sorted(os.listdir(pose_dir)):
            if not pf.endswith('.txt'):
                continue
            # r_0001.txt → frame index 0, r_0201.txt → frame index 200
            try:
                fid = int(pf.replace('.txt', '').replace('r_', '')) - 1
            except ValueError:
                continue
            with open(os.path.join(pose_dir, pf)) as f:
                lines = f.readlines()
            mat = np.array([list(map(float, l.strip().split())) for l in lines])
            if mat.shape == (4, 4):
                self.all_poses[fid] = torch.from_numpy(mat).float()

        print(f"  Loaded {len(self.all_poses)} pose matrices (frame-indexed)")

        # Camera intrinsics (Ev3D-S uses DAVIS 346 camera)
        # Default: fx=fy=888.8889, cx=319.5, cy=239.5
        self.K = torch.tensor(
            [[888.8889, 0, 319.5],
             [0, 888.8889, 239.5],
             [0, 0, 1]], dtype=torch.float32
        )

        print(f"Ev3DDatasetReal [{split}]: {len(self.scenes)} scenes, {len(self.samples)} views")

    def _events_to_voxel(self, events_xy, events_tp, H, W):
        """
        Convert events to 5-bin voxel grid (REALM format).
        events_xy: (2, N) — x, y pixel coordinates
        events_tp: (2, N) — timestamps, polarities
        Returns: (5, H, W) voxel grid
        """
        x = events_xy[0].astype(np.int64)
        y = events_xy[1].astype(np.int64)
        t = events_tp[0]
        p = events_tp[1]

        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]

        if len(x) == 0:
            return np.zeros((self.num_bins, H, W), dtype=np.float32)

        t_min, t_max = t.min(), t.max()
        if t_max - t_min < 1e-6:
            t_max = t_min + 1.0

        bin_edges = np.linspace(t_min, t_max, self.num_bins + 1)
        voxel = np.zeros((self.num_bins, H, W), dtype=np.float32)

        for b in range(self.num_bins):
            mask = (t >= bin_edges[b]) & (t < bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            bx, by, bp = x[mask], y[mask], p[mask]
            t_norm = (t[mask] - bin_edges[b]) / (bin_edges[b + 1] - bin_edges[b] + 1e-8)
            # Bilinear interpolation weight in time
            weight = np.maximum(0, 1 - np.abs(2 * t_norm - 1))
            np.add.at(voxel[b], (by, bx), bp * weight)

        # Normalize (REALM does z-score normalization on nonzero entries)
        nonzero = voxel != 0
        if nonzero.sum() > 1:
            mean = voxel[nonzero].mean()
            std = voxel[nonzero].std()
            voxel[nonzero] = (voxel[nonzero] - mean) / max(std, 1e-6)

        return voxel

    def __len__(self):
        return max(1, len(self.samples) - self.num_views)

    def __getitem__(self, idx):
        H, W = 480, 640  # DAVIS346 original resolution
        target_size = self.image_size

        # Select V consecutive views
        voxels = []
        grays = []
        depths = []
        poses_list = []

        for v in range(self.num_views):
            scene, fid = self.samples[min(idx + v, len(self.samples) - 1)]

            # ── Events → 5ch voxel ──
            ev_path = os.path.join(self.root, 'Event', scene, f'{fid:04d}.npy')
            events = np.load(ev_path)  # (4, N) — [x, y, t, p]
            voxel = self._events_to_voxel(
                events[:2], events[2:], H, W
            )
            voxel_t = torch.from_numpy(voxel).float()
            voxel_t = F.interpolate(
                voxel_t.unsqueeze(0), size=(target_size, target_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            voxels.append(voxel_t)

            # ── GT grayscale ──
            gray_path = os.path.join(self.root, 'Scenes', scene, f'{fid:04d}.png')
            if os.path.exists(gray_path):
                gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            else:
                gray = np.zeros((H, W), dtype=np.uint8)
            gray = cv2.resize(gray, (target_size, target_size))
            gray_t = torch.from_numpy(gray).float().unsqueeze(0) / 255.0
            grays.append(gray_t)

            # ── GT depth ──
            depth_path = os.path.join(self.root, 'Scenes', scene, f'{fid:04d}.npz')
            if os.path.exists(depth_path):
                depth_data = np.load(depth_path, allow_pickle=True)
                depth = depth_data['depth_map']  # (480, 640)
            else:
                depth = np.zeros((H, W), dtype=np.float32)
            depth_t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
            depth_t = F.interpolate(depth_t, size=(target_size, target_size),
                                    mode='bilinear', align_corners=False).squeeze(0)
            depths.append(depth_t)

            # ── GT pose (frame-indexed, same for all scenes) ──
            if fid in self.all_poses:
                pose = self.all_poses[fid]
            else:
                pose = torch.eye(4)
            poses_list.append(pose)

        return {
            'event_voxel': torch.stack(voxels, dim=0),   # (V, 5, H, W)
            'gt_gray':     torch.stack(grays, dim=0),     # (V, 1, H, W)
            'gt_depth':    torch.stack(depths, dim=0),    # (V, 1, H, W)
            'gt_pose':     torch.stack(poses_list, dim=0), # (V, 4, 4)
            'K':           self.K,
        }


def test_dataset():
    ds = Ev3DDatasetReal('/root/dataset/EV3DS', split='train', num_views=3)
    print(f'Dataset: {len(ds)} samples')
    sample = ds[0]
    for k, v in sample.items():
        print(f'  {k}: {v.shape}')
    print('Dataset test passed!')

if __name__ == '__main__':
    test_dataset()
