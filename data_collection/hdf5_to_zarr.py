#!/usr/bin/env python3
"""
Convert HDF5 episodes (from hdf5_data_collector.py) to per-episode zarr format
expected by XArmImageDataset2D.

Source HDF5 format:
    episode_N.hdf5
    ├── action/pose:        (T, 7) float32  [x,y,z,qx,qy,qz,qw]
    ├── action/gripper:     (T,)   float32
    ├── observation/pose:   (T, 7) float32
    ├── observation/gripper:(T,)   float32
    ├── images/rs_front:    (T, 3, H, W) uint8 CHW
    ├── images/rs_wrist:    (T, 3, H, W) uint8 CHW
    └── images/rs_head:     (T, 3, H, W) uint8 CHW

Target zarr format:
    output.zarr/
    └── episode_NNNN/
        ├── rs_front_rgb: (T, H, W, 3) uint8 HWC
        ├── rs_wrist_rgb: (T, H, W, 3) uint8 HWC
        ├── rs_head_rgb:  (T, H, W, 3) uint8 HWC
        ├── pose:         (T, 10) float32  [x,y,z,rot6d(6),gripper]
        ├── action:       (T, 10) float32  [x,y,z,rot6d(6),gripper]
        └── attrs: {length, pose_format, original_keys}

Usage:
    python3 hdf5_to_zarr.py <input_dir> <output.zarr> [--max-episodes N]
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import zarr
from natsort import natsorted

# Add dt_ag to path for RotationTransformer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dt_ag-main", "dt_ag"))
from rotation_transformer import RotationTransformer

# Matches the existing pipeline (hdf5_to_zarr_full.py line 35, inference_utils.py line 184)
rot_tf = RotationTransformer(from_rep="quaternion", to_rep="rotation_6d")


def quat_pose_gripper_to_10d(pose_7d: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,qx,qy,qz,qw] + gripper -> [x,y,z,rot6d(6),gripper].

    Uses PyTorch3D row convention (first 2 rows of rotation matrix),
    matching data creation (hdf5_to_zarr_full.py) and inference (inference_utils.py).
    """
    xyz = pose_7d[:, :3]  # (T, 3)
    quats = pose_7d[:, 3:]  # (T, 4)
    rot6d = rot_tf.forward(quats)  # (T, 6)
    grip = gripper.reshape(-1, 1)  # (T, 1)
    return np.concatenate([xyz, rot6d, grip], axis=1).astype(np.float32)  # (T, 10)


def convert_episode(h5_path: str, zarr_root: zarr.Group, episode_name: str) -> int:
    """Convert a single HDF5 episode to a zarr group. Returns number of frames."""
    with h5py.File(h5_path, "r") as f:
        T = f.attrs.get("num_frames", f["action/pose"].shape[0])

        # --- Pose and action: quaternion -> 10D ---
        obs_pose = f["observation/pose"][()]
        obs_grip = f["observation/gripper"][()]
        act_pose = f["action/pose"][()]
        act_grip = f["action/gripper"][()]

        pose_10d = quat_pose_gripper_to_10d(obs_pose, obs_grip)
        action_10d = quat_pose_gripper_to_10d(act_pose, act_grip)

        # --- Images: CHW (T,3,H,W) -> HWC (T,H,W,3) ---
        cam_map = {
            "images/rs_front": "rs_front_rgb",
            "images/rs_wrist": "rs_wrist_rgb",
            "images/rs_head": "rs_head_rgb",
        }
        images = {}
        for h5_key, zarr_key in cam_map.items():
            if h5_key in f:
                img_chw = f[h5_key][()]  # (T, 3, H, W)
                img_hwc = np.moveaxis(img_chw, 1, -1)  # (T, H, W, 3)
                images[zarr_key] = img_hwc

        # Track original HDF5 keys
        original_keys = []

        def _collect_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                original_keys.append(name)

        f.visititems(_collect_keys)

    # --- Write zarr group ---
    if episode_name in zarr_root:
        del zarr_root[episode_name]
    g = zarr_root.create_group(episode_name)

    # Images with per-frame chunking
    for zarr_key, img_arr in images.items():
        g.array(zarr_key, img_arr, chunks=(1, *img_arr.shape[1:]), dtype=img_arr.dtype)

    # Pose and action (no chunking)
    g.array("pose", pose_10d, dtype=np.float32)
    g.array("action", action_10d, dtype=np.float32)

    # Attributes
    g.attrs["length"] = int(T)
    g.attrs["pose_format"] = "x,y,z,6d,grip"
    g.attrs["original_keys"] = original_keys

    return int(T)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 episodes to per-episode zarr format for XArmImageDataset2D"
    )
    parser.add_argument("input_dir", help="Directory containing episode_*.hdf5 files")
    parser.add_argument("output_zarr", help="Output zarr directory path")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to convert")
    args = parser.parse_args()

    # Find and sort HDF5 files
    h5_files = natsorted(glob.glob(os.path.join(args.input_dir, "episode_*.hdf5")))
    if not h5_files:
        print(f"Error: No episode_*.hdf5 files found in {args.input_dir}")
        sys.exit(1)

    if args.max_episodes is not None:
        h5_files = h5_files[: args.max_episodes]

    print(f"Found {len(h5_files)} episode(s) in {args.input_dir}")

    # Open zarr store in append mode
    root = zarr.open(args.output_zarr, mode="a")

    total_frames = 0
    for idx, h5_path in enumerate(h5_files):
        ep_name = f"episode_{idx:04d}"
        print(f"  [{idx + 1}/{len(h5_files)}] {os.path.basename(h5_path)} -> {ep_name} ...", end=" ")
        T = convert_episode(h5_path, root, ep_name)
        total_frames += T
        print(f"({T} frames)")

    # --- Summary ---
    print(f"\nDone! Wrote {len(h5_files)} episode(s) to {args.output_zarr}")
    print(f"  Total frames: {total_frames}")

    # Print structure of written zarr
    root = zarr.open(args.output_zarr, mode="r")
    for ep_key in sorted(root.group_keys()):
        grp = root[ep_key]
        arrays = {k: grp[k].shape for k in grp.array_keys()}
        print(f"  {ep_key}: {arrays}  (length={grp.attrs.get('length', '?')})")


if __name__ == "__main__":
    main()
