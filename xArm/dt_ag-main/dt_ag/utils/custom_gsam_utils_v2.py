#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
from sklearn.cluster import DBSCAN
import open3d as o3d
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert
import pytorch3d.ops as torch3d_ops
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
from sam2.build_sam import build_sam2_camera_predictor

__all__ = ["GroundedSAM2", "default_gsam_config"]


def default_gsam_config() -> Dict:
    return {
        # Grounding‑DINO
        "gdino_cfg": "/home/alex/Documents/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "gdino_ckpt": "/home/alex/Documents/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth",
        "queries": ["red strawberry", "robot"],
        "box_thresh": 0.35,
        "text_thresh": 0.3,

        # SAM‑2
        "sam2_cfg": "//home/alex/Documents/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2_ckpt": "/home/alex/Documents/segment-anything-2-real-time/checkpoints/sam2.1_hiera_base_plus.pt",

        # Camera intrinsics
        "zed_intri": {
            "fx": 1069.73,
            "fy": 1069.73,
            "cx": 1135.86,
            "cy": 680.69
        },
        "T_base_zed": np.array([
            [-0.4949434,  0.2645439, -0.8276760, 0.590],
            [ 0.8685291,  0.1218584, -0.4804246, 0.520],
            [-0.0262341, -0.9566436, -0.2900771, 0.310],
            [ 0.000,      0.000,      0.000,      1.000],
        ], dtype=np.float32), # Radians

        # Point‑cloud parameters
        "pts_per_pcd": 1024,
        "fruit_fraction": 1/3,
    }

class GroundedSAM2:
    """
    Wrapper for Grounding-DINO + SAM-2 tracking and point-cloud generation.
    Set use_weights=True to generate weighted point-clouds.
    """
    def __init__(self, cfg: Dict, device: Optional[str] = None, tmp_dir: str = "__tmp_gsam2_frames", use_weights: bool = False):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_weights = use_weights

        print(f"[GSAM-UTILS] Loading models on {self.device}…")
        self.gdino = load_model(
            model_config_path=cfg["gdino_cfg"],
            model_checkpoint_path=cfg["gdino_ckpt"],
            device=self.device,
        )
        self.predictor = build_sam2_camera_predictor(cfg["sam2_cfg"], cfg["sam2_ckpt"])
        Path(tmp_dir).mkdir(exist_ok=True)
        self.tmp_dir = Path(tmp_dir)
        self._init_state = None
        self.DEPTH_GRAD_THRESH = 0.04

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounding box for point cloud cropping"""
        bounds_cfg = self.cfg.get("bounds", {"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 2.0]})
        return (
            np.array(bounds_cfg["min"], dtype=np.float32),
            np.array(bounds_cfg["max"], dtype=np.float32)
        )

    def detect_boxes(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        tf = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        pil = Image.fromarray(image)
        img_t, _ = tf(pil, None)
        H, W = image.shape[:2]
        boxes = []
        for q in self.cfg["queries"]:
            b, _, _ = predict(self.gdino, img_t, q, self.cfg["box_thresh"], self.cfg["text_thresh"])
            if b.numel():
                boxes.append(b)
        if not boxes:
            return np.empty((0, 4)), False
        B = torch.cat(boxes, 0) * torch.tensor([W, H, W, H], device=b.device)
        boxes = box_convert(B, 'cxcywh', 'xyxy').cpu().numpy()
        boxes[:, 0] = np.clip(boxes[:, 0], 0, W)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, W)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, H)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, H)
        return boxes, True

    def init_track(self, image: np.ndarray, boxes: np.ndarray) -> Dict[int, np.ndarray]:
        if boxes.size == 0:
            return {0: np.zeros(image.shape[:2], np.uint8)}
        self.predictor.load_first_frame(image)
        masks = {}
        for i, box in enumerate(boxes):
            _, _, logit = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=i,
                bbox=box.astype(np.float32),
                clear_old_points=True,
                normalize_coords=True,
            )
            masks[i] = (logit[-1, 0].cpu().numpy() > 0).astype(np.uint8) * 255
        return masks

    def propagate(self, image: np.ndarray) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        ids, logits = self.predictor.track(image)
        if self.use_weights:
            return {
                int(id_): (logits[i, 0].cpu().numpy() > 0).astype(np.uint8) * 255
                for i, id_ in enumerate(ids)
            }
        mask = np.zeros(image.shape[:2], np.uint8)
        for i in range(logits.shape[0]):
            mask |= (logits[i, 0].cpu().numpy() > 0).astype(np.uint8) * 255
        return mask

    def _make_pcd(self, mask: Union[np.ndarray, Dict[int, np.ndarray]], depth: np.ndarray, rgb_bgr: np.ndarray) -> np.ndarray:
        if self.use_weights and isinstance(mask, dict):
            return self._weighted_pcd(rgb_bgr, depth, mask)
        return self._simple_pcd(mask, depth, rgb_bgr)

    def _simple_pcd(self, mask: np.ndarray, depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        fx, fy, cx, cy = [self.cfg['zed_intri'][k] for k in ('fx', 'fy', 'cx', 'cy')]
        pts, _ = self._pcd_from_mask(mask, depth, rgb, fx, fy, cx, cy)
        # pts = self._filter_noise(pts)
        pts = self._fps_downsample(pts, self.cfg['pts_per_pcd'])
        return pts

    def _weighted_pcd(self, rgb: np.ndarray, depth: np.ndarray, masks: Dict[int, np.ndarray]) -> np.ndarray:
        frac  = float(self.cfg.get('fruit_fraction', 1/3))
        total = self.cfg['pts_per_pcd']
        fruit_n = max(1, int(total * frac))

        # ── 1. collect fruit (object-0) mask ───────────────────────────────
        fruit = masks.get(0)
        if isinstance(fruit, dict):               # flatten if nested
            fruit = np.bitwise_or.reduce(list(fruit.values()))
        if fruit is None:
            fruit = np.zeros(depth.shape, np.uint8)

        pts_f, _ = self._pcd_from_mask(fruit, depth, rgb)

        # ── 2. collect *all* other objects into a single mask ─────────────
        other_masks = []
        for k, m in masks.items():
            if k == 0:
                continue
            if isinstance(m, dict):
                other_masks.extend(m.values())
            else:
                other_masks.append(m)

        if other_masks:
            others = np.bitwise_or.reduce(other_masks)
        else:                                      # nothing else detected
            others = np.zeros(depth.shape, np.uint8)

        pts_r, _ = self._pcd_from_mask(others, depth, rgb)
        pts_f = self._filter_statistical(pts_f, nb=20, std_ratio=1.2)
        pts_r = self._filter_statistical(pts_r, nb=20, std_ratio=1.2)
        fsel = self._fps_downsample(pts_f, fruit_n)
        rsel = self._fps_downsample(pts_r, total - fruit_n)
        pts = np.vstack([fsel, rsel])
        if pts.shape[0] < total:
            pad = total - pts.shape[0]
            pts = np.pad(pts, ((0, pad), (0, 0)), 'constant')
        return pts

    def _pcd_from_mask(self, mask: np.ndarray, depth: np.ndarray, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fx, fy, cx, cy = [self.cfg['zed_intri'][k] for k in ('fx', 'fy', 'cx', 'cy')]
        if mask.ndim > 2:
            mask = mask.squeeze(0)
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), cv2.INTER_NEAREST)

        v, u = np.where(mask > 0)
        Z = depth[v, u]
        if not hasattr(self, "_grad_buf") or self._grad_buf.shape != depth.shape:
            sobx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
            soby = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
            self._grad_buf = np.hypot(sobx, soby)

        grad = self._grad_buf[v, u]
        valid = (np.isfinite(Z) & (Z > 1e-6))
        valid &= (grad < self.DEPTH_GRAD_THRESH)
        u, v, Z = u[valid], v[valid], Z[valid]
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], 1)
        base = (self.cfg['T_base_zed'] @ pts_cam.T).T[:, :3]
        cols = rgb[v, u][:, ::-1] / 255.0
        pts = np.hstack([base, cols])
        return pts, cols

    def _fps_downsample(self, pts: np.ndarray, k: int) -> np.ndarray:
        if pts.shape[0] <= k:
            return pts
        t = torch.from_numpy(pts[:, :3]).unsqueeze(0).cuda()
        _, idx = torch3d_ops.sample_farthest_points(t, K=k)
        return pts[idx.squeeze(0).cpu().numpy()]
    
    def _filter_noise(self, pts: np.ndarray) -> np.ndarray:
        """Apply all filtering techniques in sequence"""
        if pts.shape[0] == 0:
            return pts
        
        original_count = pts.shape[0]
        
        # # Stage 1: Distance-based filtering
        # pts = self._filter_by_distance_from_camera(pts, max_distance=2.0)
        # print(f"After distance filter: {pts.shape[0]}/{original_count} points")
        
        # # Stage 2: Bounds filtering
        # pts = self._filter_by_bounds(pts)
        # print(f"After bounds filter: {pts.shape[0]}/{original_count} points")
        
        # Stage 3: Statistical filtering (more aggressive)
        pts = self._filter_statistical(pts, nb=15, std_ratio=0.6)
        print(f"After statistical filter: {pts.shape[0]}/{original_count} points")
        
        # Stage 4: Cluster-based filtering (keep only largest cluster)
        if pts.shape[0] > 50:  # Only if we have enough points
            pts = self._filter_by_cluster_proximity(pts, eps=0.08, min_samples=15)
            print(f"After cluster filter: {pts.shape[0]}/{original_count} points")
        
        # # Stage 5: Percentile-based outlier removal
        # pts = self._filter_by_percentile_distance(pts, percentile=88)
        # print(f"After percentile filter: {pts.shape[0]}/{original_count} points")
        
        return pts
    
    def _filter_statistical(self, pts: np.ndarray, nb=10, std_ratio=1) -> np.ndarray:
        """
        Drop points whose average distance to `nb` neighbours is > `std_ratio`
        x global std.  NB: colours are preserved.
        """
        if pts.shape[0] < nb + 1:
            return pts
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts[:, :3])
        _, keep = p.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std_ratio)
        return pts[keep]
    
    def _filter_by_distance_from_camera(self, pts: np.ndarray, max_distance: float = 2.0) -> np.ndarray:
        """Remove points beyond a certain distance from camera"""
        if pts.shape[0] == 0:
            return pts
        distances = np.linalg.norm(pts[:, :3], axis=1)
        keep_mask = distances <= max_distance
        return pts[keep_mask]

    def _filter_by_cluster_proximity(self, pts: np.ndarray, eps: float = 0.1, min_samples: int = 10) -> np.ndarray:
        """Keep only points in the largest cluster using DBSCAN"""
        if pts.shape[0] < min_samples:
            return pts
        
        # Cluster based on 3D positions
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts[:, :3])
        labels = clustering.labels_
        
        # Find the largest cluster (excluding noise points labeled as -1)
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique_labels) == 0:
            return pts  # No clusters found, return original
        
        largest_cluster_label = unique_labels[np.argmax(counts)]
        keep_mask = labels == largest_cluster_label
        return pts[keep_mask]

    def _filter_by_percentile_distance(self, pts: np.ndarray, percentile: float = 95) -> np.ndarray:
        """Remove points beyond the Nth percentile of distances from centroid"""
        if pts.shape[0] == 0:
            return pts
        
        centroid = np.mean(pts[:, :3], axis=0)
        distances = np.linalg.norm(pts[:, :3] - centroid, axis=1)
        threshold = np.percentile(distances, percentile)
        keep_mask = distances <= threshold
        return pts[keep_mask]

    def _filter_by_bounds(self, pts: np.ndarray) -> np.ndarray:
        """Remove points outside the configured bounds"""
        if pts.shape[0] == 0:
            return pts
        
        min_bounds, max_bounds = self.bounds
        keep_mask = np.all(
            (pts[:, :3] >= min_bounds) & (pts[:, :3] <= max_bounds), 
            axis=1
        )
        return pts[keep_mask]

    def reset(self) -> None:
        self.predictor = build_sam2_camera_predictor(self.cfg['sam2_cfg'], self.cfg['sam2_ckpt'])
        for f in self.tmp_dir.glob('*.jpg'):
            f.unlink()

    def cleanup(self) -> None:
        for f in self.tmp_dir.glob('*.jpg'):
            f.unlink()
        try:
            self.tmp_dir.rmdir()
        except:
            pass