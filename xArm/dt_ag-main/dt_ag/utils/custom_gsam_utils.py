#!/usr/bin/env python3
from __future__ import annotations
import cv2, numpy as np, torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from sam2.build_sam import build_sam2_camera_predictor
from torchvision.ops import box_convert
import pytorch3d.ops as torch3d_ops
import time
__all__ = ["GroundedSAM2", "default_gsam_config"]

# ---------------------------------------------------------------------------  CONFIG --
def default_gsam_config() -> Dict:
    """
    Returns a single dictionary holding every path / hyper-param required by GroundedSAM2.
    """
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
        "zed_intri": {"fx": 1069.73, "fy": 1069.73, "cx": 1135.86, "cy": 680.69},

        "T_base_zed": np.array([
            [-0.4949434,  0.2645439, -0.8276760, 0.590],
            [0.8685291,  0.1218584, -0.4804246, 0.520],
            [-0.0262341, -0.9566436, -0.2900771, 0.310],
            [0.000,      0.000,      0.000,      1.000],
        ], dtype=np.float32),

        # Point‑cloud size
        "pts_per_pcd": 1024,
        "fruit_fraction": 1 / 3,
    }

class GroundedSAM2:
    """Light-weight wrapper around Grounding-DINO + SAM-2 video predictor."""

    def __init__(self, *, cfg: Dict, device: Optional[str] = None, tmp_dir="__tmp_gsam2_frames"):
        # pull values out of config dict
        self.cfg = cfg               
        self.text_queries = cfg["queries"]
        self.box_threshold = cfg["box_thresh"]
        self.text_threshold = cfg["text_thresh"]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[GSAM-UTILS] Loading models on {self.device} …")
        self.gdino = load_model(model_config_path = cfg["gdino_cfg"], model_checkpoint_path = cfg["gdino_ckpt"], device = self.device,)
        self.camera_predictor = build_sam2_camera_predictor(cfg["sam2_cfg"], cfg["sam2_ckpt"])

        print("[GSAM-UTILS] Models loaded.")

        self.inference_state = None
        self.frame_buffer: list[np.ndarray] = []
        self.current_frame_idx = 0
        self.tmp_dir = Path(tmp_dir); self.tmp_dir.mkdir(exist_ok=True)
        self.MIN_BOUNDS = np.array([-0.6, -0.2, -0.1])  # min values for x, y, z
        self.MAX_BOUNDS = np.array([0.4, 0.6, 0.7])  # max values for x, y, z
        
    def detect_boxes(self, image: np.ndarray, text_queries: List[str], box_threshold: float, text_threshold: float):
        start_time = time.monotonic()
        # Process the image like gdino load_image does
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Convert numpy array to PIL Image
        image_source = Image.fromarray(image)
        image_transformed, _ = transform(image_source, None)
        
        # Get image dimensions
        H, W = image.shape[:2]
        
        # Initialize an empty list to collect all boxes
        all_boxes = []
        
        # Iterate through each caption in text_queries
        for caption in text_queries:
            boxes_detected, logits, phrases = predict(
                model=self.gdino,
                image=image_transformed,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            if boxes_detected.numel():
                    all_boxes.append(boxes_detected)
        
        time_taken = time.monotonic() - start_time

        if all_boxes:
            # Concatenate boxes, rescale them to image dimensions, and convert to xyxy format
            B = torch.cat(all_boxes, 0) * torch.tensor([W, H, W, H], device=all_boxes[0].device)
            boxes_np = box_convert(B, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            return boxes_np, True, time_taken
        else:
            print("[GSAM-UTILS] No boxes detected.")
            return None, False, time_taken
        
    def initialize_tracking(self, rgb_image: np.ndarray, boxes: np.ndarray) -> Dict[int, np.ndarray]:

        start_time = time.monotonic()
        
        masks: Dict[int, np.ndarray] = {}
        if boxes is None or len(boxes) == 0:
            return {0: np.zeros(rgb_image.shape[:2], dtype=np.uint8)}
        
        # 1) Load the very first frame into the predictor
        self.camera_predictor.load_first_frame(rgb_image)
        
        # 2) For each detected box, add it as a new prompt and grab its mask
        all_masks = []
        for obj_id, box in enumerate(boxes):
            _, obj_ids, mask_logits = self.camera_predictor.add_new_prompt(
                frame_idx=0,
                obj_id=obj_id,
                bbox=box.astype(np.float32),
                clear_old_points=True,
                normalize_coords=True,
            )
            # Take only the newly added mask
            new_mask = mask_logits[-1, 0].cpu().numpy().astype(np.uint8) * 255
            all_masks.append(new_mask)
        
        # 3) Fuse and return
        if all_masks:
            masks[0] = self.combine_masks(all_masks)
        else:
            masks[0] = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        
        time_taken = time.monotonic() - start_time
        return masks, time_taken
    
    def initialize_tracking_weighted(self, rgb_image, boxes):
        """
        Returns
        -------
        Dict[int, np.ndarray]
            key 0 : strawberry mask
            key 1…: robot masks (one per extra box)
        """
        start_time = time.monotonic()
        masks: Dict[int, np.ndarray] = {}
        if boxes is None or len(boxes) == 0:
            return {0: np.zeros(rgb_image.shape[:2], np.uint8)}

        self.camera_predictor.load_first_frame(rgb_image)

        for obj_id, box in enumerate(boxes):
            _, _, mask_logits = self.camera_predictor.add_new_prompt(
                frame_idx=0, obj_id=obj_id, bbox=box.astype(np.float32),
                clear_old_points=True, normalize_coords=True,
            )
            # keep the object-specific mask *separate*
            masks[obj_id] = (mask_logits[-1, 0].cpu().numpy() > 0).astype(np.uint8) * 255

        time_taken = time.monotonic() - start_time

        return masks, time_taken
    
    def propagate_mask(self, rgb_image: np.ndarray) -> np.ndarray:
        # Track mode + mixed-precision (on GPU)
        start_time = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            obj_ids, mask_logits = self.camera_predictor.track(rgb_image)

        all_masks = []
        # mask_logits: Tensor[num_objs, 1, H, W]
        for i in range(mask_logits.shape[0]):
            # squeeze out channel dim
            raw = mask_logits[i, 0]  
            binary = (raw > 0).cpu().numpy().astype(np.uint8) * 255
            all_masks.append(binary)

        # fuse or return empty
        if all_masks:
            masks = self.combine_masks(all_masks)
            time_taken = time.monotonic() - start_time
            return masks, time_taken
        else:
            # no objects → blank mask at full resolution
            return np.zeros(rgb_image.shape[:2], dtype=np.uint8), time_taken
        
    def propagate_masks_weighted(self, rgb_image: np.ndarray):
        """
        Track all objects and return a dict {obj_id: binary_mask}.
        obj_id 0 = strawberry (first query), ≥1 = robot links / others.
        """
        start = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            obj_ids, mask_logits = self.camera_predictor.track(rgb_image)

        masks = {}
        for k in range(mask_logits.shape[0]):
            raw = mask_logits[k, 0]
            masks[int(obj_ids[k])] = (raw > 0).cpu().numpy().astype(np.uint8) * 255
        return masks, time.monotonic() - start

    def create_pcd(self, mask, depth, rgb_bgr, pts_per_pcd: int = 1024) -> np.ndarray:
        """
        Convert a depth map and RGB image to a 3D point cloud, using a mask to select pixels.
        Uses Farthest Point Sampling for downsampling when needed.
        
        Args:
            mask: Binary mask indicating which pixels to include
            depth: Depth image
            rgb_bgr: RGB image (in BGR format)
            pts_per_pcd: Number of points in output point cloud
            
        Returns:
            Point cloud as numpy array of shape (pts_per_pcd, 6) - XYZ coordinates and RGB colors
        """
        start_time = time.monotonic()
        # Extract intrinsics
        fx, fy, cx, cy = self.cfg["zed_intri"]["fx"], self.cfg["zed_intri"]["fy"], self.cfg["zed_intri"]["cx"], self.cfg["zed_intri"]["cy"]
        
        # First, ensure mask and depth are properly formed
        if mask is None or depth is None:
            print("[GSAM] Warning: Received None mask or depth")
            return np.zeros((pts_per_pcd, 6), dtype=np.float32)
        
        # Check depth dimensions
        if depth.size == 0 or depth.ndim != 2:
            print(f"[GSAM] Warning: Invalid depth dimensions: {depth.shape}")
            return np.zeros((pts_per_pcd, 6), dtype=np.float32)
        
        # Ensure mask is 2D (might be 3D with singleton dimension)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.ndim > 2:
            print(f"[GSAM] Warning: Unexpected mask dimensions: {mask.shape}, squeezing...")
            mask = np.squeeze(mask)  # Try to squeeze to 2D
                
        # Ensure mask size matches depth
        if mask.shape != depth.shape:
            try:
                print(f"[GSAM] Resizing mask from {mask.shape} to {depth.shape}")
                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                print(f"[GSAM] Error resizing mask: {e}")
                return np.zeros((pts_per_pcd, 6), dtype=np.float32)

        # Find valid pixels in mask
        v, u = np.where(mask > 0)
        if len(v) == 0:
            print("[GSAM] No non-zero pixels in mask")
            return np.zeros((pts_per_pcd, 6), dtype=np.float32)
            
        Z = depth[v, u]
        valid = np.isfinite(Z) & (Z > 1e-6)
        
        if not valid.any():
            # Return empty point cloud if no valid points
            print("[GSAM] No valid depth values in masked region")
            return np.zeros((pts_per_pcd, 6), dtype=np.float32)
        
        # Extract valid points
        u, v, Z = u[valid], v[valid], Z[valid]
        
        # Convert to 3D coordinates
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], 1)
        
        # Transform to base frame
        pts_base = (T @ pts_cam.T).T[:, :3]
        
        # Extract RGB values
        rgb = rgb_bgr[v, u][:, ::-1] / 255.0  # BGR->RGB, normalize

        # Combine position and colour
        points = pts_base.astype(np.float32)            # (N,3)
        colours = rgb.astype(np.float32)                 # (N,3)  0-1

        # Crop points within bounds
        points, colours = self.crop_points_within_bounds(points, colours)
            
        if len(points) > pts_per_pcd:      
            pts_t = torch.from_numpy(points).to("cuda")   # (N,3) 
            pts_t = pts_t.unsqueeze(0)                    # (1,N,3) as expected by PyTorch3D

            # iterative FPS on the GPU
            _, fps_idx = torch3d_ops.sample_farthest_points(pts_t, K=pts_per_pcd)
            fps_idx = fps_idx.squeeze(0).cpu().numpy()    # bring back only the indices

            points = points[fps_idx]
            colours = colours[fps_idx]

        if len(points) < pts_per_pcd:
            pad = pts_per_pcd - len(points)
            points = np.pad(points, ((0, pad), (0, 0)), 'constant')
            colours = np.pad(colours, ((0, pad), (0, 0)), 'constant')

        pcd_out = np.hstack([points, colours])            # (K,6)
        return pcd_out, time.monotonic() - start_time
    
    def weighted_pcd(self, rgb_bgr: np.ndarray, depth: np.ndarray, masks: Dict[int, np.ndarray], max_pts_num: int = 2048) -> np.ndarray:
        """
        Creates a weighted point cloud from a depth map and RGB image, using a mask to select pixels.
        """
        # ----- build raw clouds
        fruit_pts, fruit_rgb = self._compute_point_cloud(rgb_bgr, depth, [masks.get(0, np.zeros_like(depth, np.uint8))])
        robot_masks = [m for k, m in masks.items() if k != 0]
        robot_pts, robot_rgb = self._compute_point_cloud(rgb_bgr, depth, robot_masks)

        # ----- workspace crop (re-use your member fn)
        fruit_pts, fruit_rgb = self.crop_points_within_bounds(fruit_pts, fruit_rgb)
        robot_pts, robot_rgb = self.crop_points_within_bounds(robot_pts, robot_rgb)

        # ----- FPS down-sample with weighting
        frac = float(self.cfg.get("fruit_fraction", 1 / 3))
        frac = np.clip(frac, 0.0, 1.0)                 # safety
        n_fruit = max(1, int(max_pts_num * frac))
        fruit_pts, fruit_rgb = self._downsample_with_fps(fruit_pts, fruit_rgb, n_fruit)
        robot_pts, robot_rgb = self._downsample_with_fps(robot_pts, robot_rgb, max_pts_num - n_fruit)

        # ----- concatenate (strawberry first) and zero-pad if needed
        pts  = np.concatenate([fruit_pts, robot_pts ], axis=0)
        rgb = np.concatenate([fruit_rgb, robot_rgb], axis=0)
        pad  = max_pts_num - len(pts)
        if pad:
            pts = np.pad(pts, ((0, pad), (0, 0)), "constant")
            rgb = np.pad(rgb, ((0, pad), (0, 0)), "constant")

        return np.hstack([pts, rgb]).astype(np.float32)   # (K,6)
    
    def _downsample_with_fps(self, points, colours, k):
        """Pure GPU FPS courtesy of PyTorch3D; returns k points (or all if <= k)."""
        if len(points) <= k:
            return points, colours
        pts_t = torch.from_numpy(points).unsqueeze(0).to("cuda")  # (1,N,3)
        _, idx = torch3d_ops.sample_farthest_points(pts_t, K=k)
        idx = idx.squeeze(0).cpu().numpy()
        return points[idx], colours[idx]
    
    def _compute_point_cloud(self, rgb_bgr: np.ndarray, depth: np.ndarray, masks: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            masks: list of binary uint8 masks; pixels are selected if *any* mask is non-zero
        Returns:
            points - (N,3) float32 XYZ in base frame
            colours - (N,3) float32 RGB in [0,1]
        """
        if not masks:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)

        # fuse the mask list but *without* losing their identity for later weighting
        fused = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            fused |= (m > 0).astype(np.uint8)

        fx, fy, cx, cy = self.cfg["zed_intri"]["fx"], self.cfg["zed_intri"]["fy"], self.cfg["zed_intri"]["cx"], self.cfg["zed_intri"]["cy"]
        v, u = np.where(fused > 0)
        if len(v) == 0:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)

        Z = depth[v, u]
        valid = np.isfinite(Z) & (Z > 1e-6)
        u, v, Z = u[valid], v[valid], Z[valid]

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        pts_cam  = np.stack([X, Y, Z, np.ones_like(Z)], 1)         # (N,4)
        pts_base = (self.cfg["T_base_cam"] @ pts_cam.T).T[:, :3].astype(np.float32)  # (N,3)
        rgb = rgb_bgr[v, u][:, ::-1].astype(np.float32) / 255  # (N,3)

        return pts_base, rgb
    
    def combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        if not masks:
            return np.zeros((0,0), dtype=np.uint8)
        m = np.zeros_like(masks[0], dtype=bool)
        for x in masks:
            m |= x.astype(bool)
        return (m * 255).astype(np.uint8)
    
    def crop_points_within_bounds(self, points, colors, min_bounds=None, max_bounds=None):
        """
        Crop points within a bounding box.
        
        Args:
            points: numpy array of shape (N,3) containing XYZ coordinates
            colors: numpy array of shape (N,3) containing RGB colors
            min_bounds: minimum bounds for XYZ coordinates, defaults to self.MIN_BOUNDS
            max_bounds: maximum bounds for XYZ coordinates, defaults to self.MAX_BOUNDS
            
        Returns:
            tuple of (filtered_points, filtered_colors)
        """
        if min_bounds is None:
            min_bounds = self.MIN_BOUNDS
        if max_bounds is None:
            max_bounds = self.MAX_BOUNDS
            
        # Create a mask for points within the bounding box
        mask = np.all((points >= min_bounds) & (points <= max_bounds), axis=1)

        # Apply the mask to filter points and corresponding colors
        points = points[mask]
        colors = colors[mask]
        return points, colors
    
    def fuse_masks(self, masks_dict: dict[int, np.ndarray]) -> np.ndarray:
        """
        OR-fuse a {id:mask} dict into one binary mask for visual debug.
        """
        if not masks_dict:
            return np.zeros((0, 0), np.uint8)
        fused = np.zeros_like(next(iter(masks_dict.values())), np.uint8)
        for m in masks_dict.values():
            fused |= (m > 0).astype(np.uint8) * 255
        return fused

    # UTIL
    def reset(self) -> None:
        """Clears state but keeps models in memory for re-use."""
        self.inference_state = None
        self.frame_buffer.clear()
        self.current_frame_idx = 0
        for f in self.tmp_dir.glob("*.jpg"):
            f.unlink(missing_ok=True)

    def cleanup(self) -> None:
        """Full clean-up for process shutdown."""
        self.reset()
        try:
            self.tmp_dir.rmdir()
        except OSError:
            pass  # directory not empty - ignore