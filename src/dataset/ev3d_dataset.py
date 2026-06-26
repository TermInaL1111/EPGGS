"""
Ev3D-S Dataset loader for EPGGS.

Loads event data in Ev3D-S format:
    - 201 views per object
    - Each view has: event .npy, GT grayscale, GT depth, GT pose, mask
"""
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class Ev3DDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 336,
        num_views: int = 3,
        split: str = 'train',
        num_bins: int = 5,
        dt_ms: float = 50.0,
    ):
        """
        Args:
            root: path to Ev3D-S root (contains train/ and test/)
            image_size: resize target (square)
            num_views: number of views per scene sample
            split: 'train' or 'test'
            num_bins: temporal bins for voxel grid (EvGGS uses 5)
            dt_ms: time window for event accumulation (ms)
        """
        self.root = root
        self.image_size = image_size
        self.num_views = num_views
        self.num_bins = num_bins
        self.dt_ms = dt_ms

        # Find all object directories
        split_dir = os.path.join(root, split)
        self.object_dirs = sorted(glob.glob(os.path.join(split_dir, '*')))
        print(f"Found {len(self.object_dirs)} objects in {split} split")

        # Pre-load object metadata
        self.scenes = []
        for obj_dir in self.object_dirs:
            event_dir = os.path.join(obj_dir, 'events')
            gray_dir = os.path.join(obj_dir, 'images')
            depth_dir = os.path.join(obj_dir, 'depth')
            pose_file = os.path.join(obj_dir, 'poses.txt')
            calib_file = os.path.join(obj_dir, 'calib.txt')

            if not os.path.exists(event_dir):
                continue

            # Load camera intrinsics
            K = self._load_calib(calib_file) if os.path.exists(calib_file) else None

            # Load all poses
            poses = self._load_poses(pose_file) if os.path.exists(pose_file) else None

            # List event files and corresponding views
            event_files = sorted(glob.glob(os.path.join(event_dir, '*.npy')))
            for evt_file in event_files:
                fname = os.path.splitext(os.path.basename(evt_file))[0]
                gray_file = os.path.join(gray_dir, f"{fname}.png")
                depth_file = os.path.join(depth_dir, f"{fname}.npy")
                if os.path.exists(gray_file):
                    self.scenes.append({
                        'event_file': evt_file,
                        'gray_file': gray_file,
                        'depth_file': depth_file if os.path.exists(depth_file) else None,
                        'poses': poses,
                        'K': K,
                        'frame_idx': int(fname.replace('frame', '')),
                    })

        print(f"Total views: {len(self.scenes)}")

    def _load_calib(self, path):
        """Load camera intrinsics K (3,3)."""
        with open(path, 'r') as f:
            fx, fy, cx, cy = map(float, f.readline().split()[:4])
        return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

    def _load_poses(self, path):
        """Load camera poses per view."""
        poses = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 17:
                    continue
                ts = float(parts[0])
                mat = np.array(parts[1:17], dtype=np.float32).reshape(4, 4)
                poses[ts] = torch.from_numpy(mat)
        return poses

    def _events_to_voxel(self, events, H, W):
        """Convert event stream to voxel grid (EvGGS-style, B=5 bins)."""
        if len(events) == 0:
            return np.zeros((self.num_bins, H, W), dtype=np.float32)

        t_min = events[:, 0].min()
        t_max = events[:, 0].max()
        if t_max - t_min < 1e-6:
            t_max = t_min + self.dt_ms / 1000.0

        bin_edges = np.linspace(t_min, t_max, self.num_bins + 1)
        voxel = np.zeros((self.num_bins, H, W), dtype=np.float32)

        for b in range(self.num_bins):
            mask = (events[:, 0] >= bin_edges[b]) & (events[:, 0] < bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            bin_events = events[mask]
            for e in bin_events:
                x, y, p = int(e[1]), int(e[2]), e[3]
                if 0 <= x < W and 0 <= y < H:
                    t_norm = (e[0] - bin_edges[b]) / (bin_edges[b + 1] - bin_edges[b] + 1e-8)
                    voxel[b, y, x] += p * max(0, 1 - abs(2 * t_norm - 1))

        return voxel

    def _events_to_frame(self, events, H, W):
        """Accumulate events into 3-channel frame (positive/negative/all)."""
        frame = np.zeros((3, H, W), dtype=np.float32)
        for e in events:
            x, y, p = int(e[1]), int(e[2]), e[3]
            if 0 <= x < W and 0 <= y < H:
                if p > 0:
                    frame[0, y, x] += 1  # positive
                else:
                    frame[1, y, x] += 1  # negative
                frame[2, y, x] += 1  # all

        # Normalize
        for c in range(3):
            mx = frame[c].max()
            if mx > 0:
                frame[c] /= mx
        return frame

    def __len__(self):
        return max(1, len(self.scenes) - self.num_views)

    def __getitem__(self, idx):
        """
        Returns a batch of V consecutive views from the same object.

        Returns dict with:
            'event_voxel': (V, 8, H, W) — 3ch frame + 5ch voxel
            'gt_gray':     (V, 1, H, W)
            'gt_depth':    (V, 1, H, W)
            'gt_pose':     (V, 4, 4)
            'K':           (3, 3)
        """
        H, W = 480, 640  # default Ev3D-S resolution
        to_size = self.image_size

        # Select V consecutive views
        views = []
        for v in range(self.num_views):
            scene = self.scenes[min(idx + v, len(self.scenes) - 1)]
            views.append(scene)

        # Load data per view
        event_voxels = []
        gt_grays = []
        gt_depths = []
        gt_poses = []

        for view in views:
            # Events → voxel + frame
            events = np.load(view['event_file'])
            voxel = self._events_to_voxel(events, H, W)
            frame = self._events_to_frame(events, H, W)

            # Resize
            voxel_t = torch.from_numpy(voxel).float()
            frame_t = torch.from_numpy(frame).float()
            voxel_t = F.interpolate(voxel_t.unsqueeze(0), size=(to_size, to_size),
                                     mode='bilinear', align_corners=False).squeeze(0)
            frame_t = F.interpolate(frame_t.unsqueeze(0), size=(to_size, to_size),
                                     mode='bilinear', align_corners=False).squeeze(0)

            # Concatenate: 3ch frame + 5ch voxel = 8ch
            event_voxel = torch.cat([frame_t, voxel_t], dim=0)  # (8, H, W)
            event_voxels.append(event_voxel)

            # GT grayscale
            gray = cv2.imread(view['gray_file'], cv2.IMREAD_GRAYSCALE)
            gray = cv2.resize(gray, (to_size, to_size))
            gray_t = torch.from_numpy(gray).float().unsqueeze(0) / 255.0  # (1, H, W)
            gt_grays.append(gray_t)

            # GT depth (if available)
            if view['depth_file'] and os.path.exists(view['depth_file']):
                depth = np.load(view['depth_file'])
                depth_t = torch.from_numpy(depth).float()
                depth_t = F.interpolate(depth_t.unsqueeze(0).unsqueeze(0),
                                         size=(to_size, to_size),
                                         mode='bilinear').squeeze()
            else:
                depth_t = torch.zeros(1, to_size, to_size)
            gt_depths.append(depth_t)

            # GT pose
            if view['poses'] is not None and view['frame_idx'] in view['poses']:
                gt_poses.append(view['poses'][view['frame_idx']])
            else:
                gt_poses.append(torch.eye(4))

        return {
            'event_voxel': torch.stack(event_voxels, dim=0),  # (V, 8, H, W)
            'gt_gray':     torch.stack(gt_grays, dim=0),       # (V, 1, H, W)
            'gt_depth':    torch.stack(gt_depths, dim=0),      # (V, 1, H, W)
            'gt_pose':     torch.stack(gt_poses, dim=0),       # (V, 4, 4)
            'K':           views[0]['K'] if views[0]['K'] is not None else torch.eye(3),
        }


# Torch functional import for resize
import torch.nn.functional as F
