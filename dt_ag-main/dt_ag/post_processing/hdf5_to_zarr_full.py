#!/usr/bin/env python3
"""
Dynamic HDF5 → Zarr converter with optional GDINO + SAM-2 processing
===================================================================
• Works with episodes recorded by either collector variant:
      – rs_side_rgb, zed_rgb, zed_depth (+/- dt_left, dt_right)
• An episode is considered complete only w.r.t. its own
  grp.attrs["original_keys"] (+ optional 'action').
• Stores first/last frame PNGs to a *_debug folder as before.
"""

# ────────────────────────────────────────────────────────────────
#  Imports
# ────────────────────────────────────────────────────────────────
import os, glob, traceback
from pathlib import Path
from typing  import Dict, List, Tuple, Optional

import cv2, h5py, zarr, numpy as np
from PIL        import Image
from natsort    import natsorted
from rich.console import Console
from tqdm       import tqdm

# Optional extras ------------------------------------------------------------
try:
    from inference.custom_gsam_utils_v2 import GroundedSAM2 as GSamUtil, default_gsam_config
    GSAM_AVAILABLE = True
except ImportError:
    GSAM_AVAILABLE = False
    print("Warning: GSAM not available - mask/PCD generation disabled.")

try:
    from rotation_transformer import RotationTransformer
    rotation_transformer = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')
    ROTATION_TRANSFORMER_AVAILABLE = True
except ImportError:
    ROTATION_TRANSFORMER_AVAILABLE = False
    print("Warning: rotation_transformer not available - pose kept as quaternion.")

DEBUGGING  = False       # only convert episode_0000 when True
USE_GSAM   = False and GSAM_AVAILABLE
CONSOLE    = Console()

# ---------------------------------------------------------------------------
#  Dataset type helpers
# ---------------------------------------------------------------------------
IMAGE_KEYS = {
    'rgb', 'color', 'image', 'rs_front_rgb', 'rs_side_rgb', 'zed_rgb', 'dt_left', 'dt_right'
}
DEPTH_KEYS = {'depth', 'zed_depth'}
POSE_KEYS  = {'pose', 'last_pose'}

def hdf5_to_dict(p: str) -> Dict[str, np.ndarray]:
    with h5py.File(p, 'r') as f:
        return {k: f[k][()] for k in f.keys()}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_image_data(key: str, arr: np.ndarray) -> bool:
    if any(k in key.lower() for k in IMAGE_KEYS):
        return True
    if arr.ndim in (3,4) and arr.dtype == np.uint8:
        if arr.ndim == 4 and (arr.shape[-1] == 3 or arr.shape[1] == 3):
            return True
    return False

def is_depth_data(key: str, arr: np.ndarray) -> bool:
    return (key.lower() in DEPTH_KEYS or any(k in key.lower() for k in DEPTH_KEYS)) \
           and arr.dtype in (np.uint16, np.float32, np.float64)

def save_depth_png(depth: np.ndarray, out_file: Path):
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if depth.max() == depth.min():
        norm = np.zeros_like(depth, np.uint8)
    else:
        norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colour = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(out_file), colour)

def save_first_last_frames(data: Dict[str, np.ndarray], ep_dir: Path):
    ensure_dir(ep_dir)
    T = next(iter(data.values())).shape[0]
    for key, arr in data.items():
        first, last = arr[0], arr[T-1]
        for img, tag in [(first,'first'), (last,'last')]:
            if is_image_data(key, arr):
                img_hwc = img.transpose(1,2,0) if (img.ndim==3 and img.shape[0]==3) else img
                cv2.imwrite(str(ep_dir/f"{key}_{tag}.png"),
                            cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR) if img_hwc.shape[-1]==3 else img_hwc)
            elif is_depth_data(key, arr):
                save_depth_png(img, ep_dir/f"{key}_{tag}.png")

# ---------------------------------------------------------------------------
#  Episode completeness (data-driven)
# ---------------------------------------------------------------------------
def episode_complete(g: zarr.hierarchy.Group) -> bool:
    if 'length' not in g.attrs or 'original_keys' not in g.attrs:
        return False
    expected = set(g.attrs['original_keys'])
    if 'action' in g:
        expected.add('action')
    missing = [k for k in expected if k not in g]
    if missing:
        CONSOLE.log(f"[yellow]{g.name} missing {missing}")
        return False
    return True

def already_done(root: zarr.hierarchy.Group) -> set:
    done = {ep for ep in root.keys()
            if ep.startswith('episode_') and episode_complete(root[ep])}
    if done:
        CONSOLE.log(f"[green]Already done: {len(done)} episode(s)")
    return done

# ---------------------------------------------------------------------------
#  Pose helper
# ---------------------------------------------------------------------------
def combine_pose_gripper(pose: np.ndarray, grip: np.ndarray) -> np.ndarray:
    grip = grip.reshape(-1,1) if grip.ndim == 1 else grip
    if ROTATION_TRANSFORMER_AVAILABLE:
        xyz  = pose[:, :3]
        rot6 = rotation_transformer.forward(pose[:, 3:])
        return np.concatenate([xyz, rot6, grip], 1).astype(np.float32)
    else:
        raise ValueError("Rotation transformer not available")

# ---------------------------------------------------------------------------
#  Optional GSAM-2 wrapper
# ---------------------------------------------------------------------------
class GSAM2:
    def __init__(self):
        self.cfg  = default_gsam_config()
        self.util = GSamUtil(cfg=self.cfg, use_weights=True)

    def process(self, frames: np.ndarray, depths: Optional[np.ndarray], dump: Path):
        ensure_dir(dump)
        T = len(frames)
        boxes, ok = self.util.detect_boxes(frames[0])
        if not ok:
            return []

        masks0 = self.util.init_track(frames[0], boxes)
        for oid, m in masks0.items():
            Image.fromarray(m).save(dump / f"frame_0000_obj{oid}_mask.png")

        masks = {0: masks0}
        for t in range(1, T):
            mp = self.util.propagate(frames[t])
            masks[t] = mp
            for oid, m in mp.items():
                Image.fromarray(m).save(dump / f"frame_{t:04d}_obj{oid}_mask.png")

        if depths is None:
            CONSOLE.log("[yellow]No depth → skipping PCD")
            return []

        pcds = [self.util._make_pcd(masks[t], depths[t], frames[t]) for t in range(T)]
        return pcds

# ---------------------------------------------------------------------------
#  Main conversion
# ---------------------------------------------------------------------------
def main():
    H5_DIR   = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/2d_strawberry_baseline/4_cam_baseline")
    ZARR_DIR = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/2d_strawberry_baseline/4_cam_baseline_zarr")
    ZARR_DIR.mkdir(parents=True, exist_ok=True)

    DEBUG_DIR = ZARR_DIR.parent / (ZARR_DIR.name.replace("_zarr", "_debug"))
    ensure_dir(DEBUG_DIR)

    h5_files = natsorted(glob.glob(str(H5_DIR / "*.hdf5")))
    if not h5_files:
        CONSOLE.log(f"[red]No HDF5 files in {H5_DIR}")
        return

    root = zarr.open(ZARR_DIR, mode='a')
    done = already_done(root)
    todo = [(i,f) for i,f in enumerate(h5_files) if f"episode_{i:04d}" not in done]
    if DEBUGGING:
        todo = todo[:1]
    if not todo:
        CONSOLE.log("[green]Everything up-to-date!")
        return

    gsam = GSAM2() if USE_GSAM else None
    if USE_GSAM:
        CONSOLE.log("[blue]GSAM-2 enabled")

    for idx, path in tqdm(todo, desc="Episodes"):
        ep = f"episode_{idx:04d}"
        CONSOLE.log(f"[cyan]→ {ep}")

        try:
            data = hdf5_to_dict(path)
        except Exception as e:
            CONSOLE.log(f"[red]Failed to load {path}: {e}")
            continue

        T = next(iter(data.values())).shape[0]
        ep_dir = DEBUG_DIR / ep
        save_first_last_frames(data, ep_dir)

        # pose / action ------------------------------------------------------
        pose_arr = action_arr = None
        if "pose" in data and "gripper" in data:
            pose_arr = combine_pose_gripper(data["pose"], data["gripper"])
            if "last_pose" in data:
                action_arr = combine_pose_gripper(data["last_pose"], data["gripper"])

        # GSAM ---------------------------------------------------------------
        pcds_arr = None
        if gsam:
            rgb_key   = next((k for k in ("zed_rgb", "rs_front_rgb", "rs_side_rgb") if k in data), None)
            depth_key = "zed_depth" if rgb_key and "zed" in rgb_key and "zed_depth" in data else None
            if rgb_key:
                pcds = gsam.process(data[rgb_key], data.get(depth_key), ep_dir/"masks")
                if pcds:
                    pcds_arr = np.stack(pcds, axis=0)

        # write to Zarr ------------------------------------------------------
        if ep in root:
            del root[ep]
        g = root.create_group(ep)

        for key, arr in data.items():
            if key in ("pose", "gripper", "last_pose") and pose_arr is not None:
                continue
            chunks = (1,*arr.shape[1:]) if is_image_data(key,arr) else None
            g.array(key, arr, chunks=chunks, dtype=arr.dtype)

        if pose_arr is not None:
            g.array("pose", pose_arr, dtype=np.float32)
            fmt = "x,y,z,6d,grip" if ROTATION_TRANSFORMER_AVAILABLE else "x,y,z,quat,grip"
            g.attrs["pose_format"] = fmt
        if action_arr is not None:
            g.array("action", action_arr, dtype=np.float32)
        if pcds_arr is not None:
            g.array("point_cloud", pcds_arr, dtype=np.float32, chunks=(1,*pcds_arr.shape[1:]))

        g.attrs["length"]        = T
        g.attrs["original_keys"] = list(data.keys())

        CONSOLE.log(f"[green]✓ {ep} saved  ({len(data)} datasets)")

    CONSOLE.log(f"[green]Finished - processed {len(todo)} new episode(s)")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
