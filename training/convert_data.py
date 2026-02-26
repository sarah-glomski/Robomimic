"""
Convert HDF5 teleop episodes to per-episode zarr for XArmImageDataset2D.

Reads all HDF5 episodes from demo_data/Collection{1,2,3}/, converts
quaternion poses to 10D [x,y,z,rot6d(6),gripper], resizes images
from 360x640 to 224x224, and writes a zarr store with per-episode groups.

Usage:
    python convert_data.py --output xarm_teleop.zarr
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr

# Add UMI codebase so we can import RotationTransformer
_UMI_ROOT = os.path.join(os.path.dirname(__file__), "..", "dt_ag-main")
sys.path.insert(0, os.path.join(_UMI_ROOT, "universal_manipulation_interface"))
sys.path.insert(0, _UMI_ROOT)

from dt_ag.rotation_transformer import RotationTransformer


# ---------------------------------------------------------------------------

DEMO_ROOT = os.path.join(os.path.dirname(__file__), "..", "data_collection", "demo_data")
COLLECTIONS = ["Collection4", "Collection5", "Collection6", "Collection7"]
IMG_SIZE = 224  # ViT input size
CAMERA_KEYS = ["rs_front", "rs_wrist", "rs_head"]


def quat_pose_to_10d(pose_7: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Convert (T,7) quaternion pose + (T,) gripper to (T,10) [xyz, rot6d, grip].

    HDF5 stores pose as [x, y, z, qx, qy, qz, qw] (scipy / ROS convention).
    RotationTransformer expects the same quaternion layout that pytorch3d uses
    which is (w, x, y, z).  We re-order before calling forward().
    """
    T = pose_7.shape[0]
    xyz = pose_7[:, :3]  # (T, 3)
    quat_xyzw = pose_7[:, 3:7]  # (T, 4)  — x y z w

    # Convert to wxyz for pytorch3d
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=1)

    rot_tf = RotationTransformer(from_rep="quaternion", to_rep="rotation_6d")
    rot6d = rot_tf.forward(quat_wxyz)  # (T, 6)
    if isinstance(rot6d, np.ndarray) is False:
        rot6d = rot6d.numpy()

    grip = gripper[:, None]  # (T, 1)
    return np.concatenate([xyz, rot6d, grip], axis=1).astype(np.float32)  # (T, 10)


def resize_images(imgs_chw: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize (T, C, H, W) uint8 images to (T, size, size, 3) HWC uint8.

    The dataset class expects HWC storage and converts to CHW internally.
    """
    T, C, H, W = imgs_chw.shape
    out = np.empty((T, size, size, C), dtype=np.uint8)
    for t in range(T):
        frame = np.transpose(imgs_chw[t], (1, 2, 0))  # CHW -> HWC
        out[t] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return out


def collect_episodes() -> list:
    """Return sorted list of (collection_name, episode_path) tuples."""
    episodes = []
    for coll in COLLECTIONS:
        coll_dir = os.path.join(DEMO_ROOT, coll)
        if not os.path.isdir(coll_dir):
            print(f"Warning: {coll_dir} not found, skipping")
            continue
        for fname in sorted(os.listdir(coll_dir)):
            if fname.endswith(".hdf5"):
                episodes.append((coll, os.path.join(coll_dir, fname)))
    return episodes


def convert(output_path: str):
    episodes = collect_episodes()
    print(f"Found {len(episodes)} episodes across {COLLECTIONS}")

    store = zarr.open(output_path, mode="w")

    for ep_idx, (coll, hdf5_path) in enumerate(episodes):
        ep_name = f"episode_{ep_idx}"
        print(f"[{ep_idx+1}/{len(episodes)}] {coll}/{os.path.basename(hdf5_path)} -> {ep_name}")

        with h5py.File(hdf5_path, "r") as f:
            T = f["observation/pose"].shape[0]

            # --- Pose / action in 10-D -------------------------------------------
            obs_pose = f["observation/pose"][:]  # (T, 7)
            obs_grip = f["observation/gripper"][:]  # (T,)
            pose_10d = quat_pose_to_10d(obs_pose, obs_grip)

            act_pose = f["action/pose"][:]  # (T, 7)
            act_grip = f["action/gripper"][:]  # (T,)
            action_10d = quat_pose_to_10d(act_pose, act_grip)

            # --- Images ----------------------------------------------------------
            grp = store.create_group(ep_name)
            for cam_key in CAMERA_KEYS:
                src_key = f"images/{cam_key}"
                raw = f[src_key][:]  # (T, 3, 360, 640) uint8
                resized = resize_images(raw, IMG_SIZE)  # (T, 224, 224, 3) HWC
                zarr_key = f"{cam_key}_rgb"
                grp.create_dataset(zarr_key, data=resized, chunks=(1, IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
                print(f"  {zarr_key}: {resized.shape}")

            # --- Pose & action arrays --------------------------------------------
            grp.create_dataset("pose", data=pose_10d, chunks=(T, 10), dtype="float32")
            grp.create_dataset("action", data=action_10d, chunks=(T, 10), dtype="float32")
            print(f"  pose: {pose_10d.shape}  action: {action_10d.shape}")

    print(f"\nDone. Zarr store written to {output_path}")
    print(f"Total episodes: {len(episodes)}")

    # Quick verification
    z = zarr.open(output_path, mode="r")
    for ep in sorted(z.group_keys()):
        g = z[ep]
        shapes = {k: g[k].shape for k in g.array_keys()}
        print(f"  {ep}: {shapes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 episodes to zarr")
    parser.add_argument("--output", type=str, default="xarm_teleop.zarr",
                        help="Output zarr path")
    args = parser.parse_args()
    convert(args.output)
