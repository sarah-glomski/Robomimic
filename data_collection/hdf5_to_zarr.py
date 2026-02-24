#!/usr/bin/env python3
"""
Convert HDF5 episodes (from hdf5_data_collector.py) to UMI-style flat
concatenated zarr format compatible with ReplayBuffer / SequenceSampler.

Source HDF5 format:
    episode_N.hdf5
    ├── action/pose:        (T, 7) float32  [x,y,z,qx,qy,qz,qw]
    ├── action/gripper:     (T,)   float32
    ├── observation/pose:   (T, 7) float32
    ├── observation/gripper:(T,)   float32
    ├── images/rs_front:    (T, 3, H, W) uint8 CHW
    ├── images/rs_wrist:    (T, 3, H, W) uint8 CHW
    └── images/rs_head:     (T, 3, H, W) uint8 CHW

Target zarr format (UMI-style flat concatenation):
    output.zarr/
    ├── data/
    │   ├── rs_front_rgb: (N, H, W, 3) uint8    # N = total frames across all episodes
    │   ├── rs_wrist_rgb: (N, H, W, 3) uint8
    │   ├── rs_head_rgb:  (N, H, W, 3) uint8
    │   ├── pose:         (N, 10)      float32  [x,y,z,rot6d(6),gripper]
    │   └── action:       (N, 10)      float32  [x,y,z,rot6d(6),gripper]
    └── meta/
        └── episode_ends: (num_episodes,) int64  # cumulative end indices

    episode_ends stores the cumulative frame count at the end of each episode.
    For example, if episode 0 has 100 frames and episode 1 has 150 frames,
    episode_ends = [100, 250].

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


def load_episode(h5_path: str) -> dict:
    """Load a single HDF5 episode and return arrays as a dict.

    Returns dict with keys:
        pose:         (T, 10) float32
        action:       (T, 10) float32
        rs_front_rgb: (T, H, W, 3) uint8  (if present)
        rs_wrist_rgb: (T, H, W, 3) uint8  (if present)
        rs_head_rgb:  (T, H, W, 3) uint8  (if present)
    """
    data = {}
    with h5py.File(h5_path, "r") as f:
        # --- Pose and action: quaternion -> 10D ---
        obs_pose = f["observation/pose"][()]
        obs_grip = f["observation/gripper"][()]
        act_pose = f["action/pose"][()]
        act_grip = f["action/gripper"][()]

        data["pose"] = quat_pose_gripper_to_10d(obs_pose, obs_grip)
        data["action"] = quat_pose_gripper_to_10d(act_pose, act_grip)

        # --- Images: CHW (T,3,H,W) -> HWC (T,H,W,3) ---
        cam_map = {
            "images/rs_front": "rs_front_rgb",
            "images/rs_wrist": "rs_wrist_rgb",
            "images/rs_head": "rs_head_rgb",
        }
        for h5_key, zarr_key in cam_map.items():
            if h5_key in f:
                img_chw = f[h5_key][()]  # (T, 3, H, W)
                data[zarr_key] = np.moveaxis(img_chw, 1, -1)  # (T, H, W, 3)

    return data


def create_zarr_structure(root: zarr.Group, first_episode_data: dict):
    """Create data/ and meta/ groups with zero-length arrays matching the first episode."""
    data_group = root.require_group("data", overwrite=False)
    meta_group = root.require_group("meta", overwrite=False)

    for key, value in first_episode_data.items():
        shape = (0,) + value.shape[1:]
        if value.ndim >= 3:
            # Image arrays: chunk per frame
            chunks = (1,) + value.shape[1:]
        else:
            # Pose/action: chunk 1000 rows
            chunks = (1000,) + value.shape[1:]
        data_group.zeros(key, shape=shape, chunks=chunks, dtype=value.dtype)

    meta_group.zeros("episode_ends", shape=(0,), dtype=np.int64, compressor=None)


def append_episode(root: zarr.Group, episode_data: dict):
    """Append one episode's data to the flat zarr store.

    Mirrors the ReplayBuffer.add_episode() resize-append pattern.
    """
    data_group = root["data"]
    meta_group = root["meta"]
    episode_ends = meta_group["episode_ends"]

    # Current total length
    curr_len = int(episode_ends[-1]) if episode_ends.shape[0] > 0 else 0

    # Episode length (from first array)
    T = None
    for value in episode_data.values():
        if T is None:
            T = value.shape[0]
        else:
            assert T == value.shape[0], "All arrays in episode must have same length"

    new_len = curr_len + T

    # Resize and copy each array
    for key, value in episode_data.items():
        arr = data_group[key]
        assert value.shape[1:] == arr.shape[1:], (
            f"Shape mismatch for {key}: expected {arr.shape[1:]}, got {value.shape[1:]}"
        )
        new_shape = (new_len,) + value.shape[1:]
        arr.resize(new_shape)
        arr[curr_len:new_len] = value

    # Append to episode_ends
    episode_ends.resize(episode_ends.shape[0] + 1)
    episode_ends[-1] = new_len


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 episodes to UMI-style flat zarr format"
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

    # Open zarr store fresh
    root = zarr.open(args.output_zarr, mode="w")

    total_frames = 0
    for idx, h5_path in enumerate(h5_files):
        print(f"  [{idx + 1}/{len(h5_files)}] {os.path.basename(h5_path)} ...", end=" ")
        ep_data = load_episode(h5_path)
        T = next(iter(ep_data.values())).shape[0]

        if idx == 0:
            create_zarr_structure(root, ep_data)

        append_episode(root, ep_data)
        total_frames += T
        print(f"({T} frames)")

    # --- Summary ---
    print(f"\nDone! Wrote {len(h5_files)} episode(s) to {args.output_zarr}")
    print(f"  Total frames: {total_frames}")

    root = zarr.open(args.output_zarr, mode="r")
    episode_ends = root["meta/episode_ends"][:]
    print(f"  episode_ends: {episode_ends}")
    for key in sorted(root["data"].array_keys()):
        arr = root["data"][key]
        print(f"  data/{key}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
