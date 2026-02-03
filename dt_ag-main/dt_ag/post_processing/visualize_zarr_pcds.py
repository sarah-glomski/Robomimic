#!/usr/bin/env python3
"""
Visualize Zarr Episode Point Clouds (Interactive Open3D version)
===============================================================

Opens the hard-coded Zarr store, selects an episode, and displays **each**
`zed_pcd` frame in its own Open3D window with a coordinate frame overlay.
Press 'q' to close the current window and move to the next frame.

Usage:
    python visualize_zarr_episode_o3d.py <episode_index>
"""

import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import zarr

# ──────────────────────────────────────────────────────────────
#  HARD-CODED PARAMETERS
# ──────────────────────────────────────────────────────────────
ZARR_STORE = (
    "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/"
    "3d_strawberry_baseline/new_setup_100_baseline_zarr"
)
WINDOW_W, WINDOW_H = 1000, 800
AXIS_SIZE = 0.1  # length of coordinate axes


def load_episode(store, idx: int) -> np.ndarray:
    episodes = sorted(store.group_keys())
    if not episodes:
        raise RuntimeError(f"No episodes found in {ZARR_STORE}")
    if not (0 <= idx < len(episodes)):
        raise ValueError(f"episode_index must be 0–{len(episodes)-1}")

    ep_name = episodes[idx]
    print(f"Episode '{ep_name}'  ({idx}/{len(episodes)-1})")

    try:
        pcd_ds = store[ep_name]["zed_pcd"]
    except KeyError:
        raise KeyError(f"'zed_pcd' not found under {ep_name}")

    return pcd_ds[:], ep_name


def show_frame(pc: np.ndarray, title: str) -> None:
    """Open one Open3D window with the given point cloud & coord-frame.
    Wait for user to press 'q' to close and continue."""
    
    # build PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

    if pc.shape[1] >= 6:  # RGB present
        cols = pc[:, 3:6]
        # ensure colors are 0–1
        if cols.max() > 1.0:
            cols = cols / 255.0
        pcd.colors = o3d.utility.Vector3dVector(cols)

    # coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE)

    vis = o3d.visualization.Visualizer()
    vis.create_window(title, WINDOW_W, WINDOW_H)
    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 2.0

    print(f"  → Press 'q' to close this window and move to next frame")
    
    # Keep the window open until user presses 'q'
    while True:
        vis.poll_events()
        vis.update_renderer()
        
        # Check if window was closed manually
        if not vis.poll_events():
            break
            
        # Small sleep to prevent high CPU usage
        time.sleep(0.01)
    
    vis.destroy_window()


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <episode_index>")
        sys.exit(1)

    try:
        episode_idx = int(sys.argv[1])
    except ValueError:
        print("Error: episode_index must be an integer")
        sys.exit(1)

    store = zarr.open(ZARR_STORE, mode="r")
    pcd_frames, ep_name = load_episode(store, episode_idx)
    num_frames = pcd_frames.shape[0]
    print(f"Loaded {num_frames} frames, each shape {pcd_frames.shape[1:]}")
    print("Instructions: Use mouse to rotate/pan/zoom. Press 'q' to move to next frame.")

    for i, pc in enumerate(pcd_frames):
        title = f"{ep_name} — frame {i+1}/{num_frames}"
        print(f"\nShowing {title}")
        show_frame(pc, title)

    print("\nFinished displaying all frames.")


if __name__ == "__main__":
    main()