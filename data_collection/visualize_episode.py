#!/usr/bin/env python3
"""
Visualize a single HDF5 episode collected by hdf5_data_collector.py.

Displays:
  - Top rows: Camera images (front, wrist, head) at N evenly-spaced timesteps
  - Bottom plots: Action/observation pose XYZ, gripper command/state over time

Usage:
    python3 visualize_episode.py <episode.hdf5> [--num-steps 10]
"""

import argparse
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_episode(path: str) -> dict:
    """Load all datasets and attrs from an HDF5 episode file."""
    data = {}
    attrs = {}
    with h5py.File(path, "r") as f:
        attrs["num_frames"] = f.attrs.get("num_frames", None)
        attrs["collection_rate_hz"] = f.attrs.get("collection_rate_hz", None)
        attrs["episode_index"] = f.attrs.get("episode_index", None)

        data["action_pose"] = f["action/pose"][()]
        data["action_gripper"] = f["action/gripper"][()]
        data["observation_pose"] = f["observation/pose"][()]
        data["observation_gripper"] = f["observation/gripper"][()]

        # Images: (T, 3, H, W) CHW uint8
        cam_keys = ["rs_front", "rs_wrist", "rs_head"]
        for key in cam_keys:
            ds_path = f"images/{key}"
            if ds_path in f:
                data[key] = f[ds_path][()]

    return data, attrs


def main():
    parser = argparse.ArgumentParser(description="Visualize an HDF5 episode")
    parser.add_argument("episode", help="Path to episode_N.hdf5 file")
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Number of timesteps to show (default: 10)"
    )
    args = parser.parse_args()

    data, attrs = load_episode(args.episode)

    T = data["action_pose"].shape[0]
    num_steps = min(args.num_steps, T)
    step_indices = np.linspace(0, T - 1, num_steps, dtype=int)

    # --- Print summary ---
    print(f"Episode: {args.episode}")
    print(f"  num_frames:        {attrs['num_frames']}")
    print(f"  collection_rate:   {attrs['collection_rate_hz']} Hz")
    print(f"  episode_index:     {attrs['episode_index']}")
    print(f"  action/pose:       {data['action_pose'].shape}  range: [{data['action_pose'].min():.4f}, {data['action_pose'].max():.4f}]")
    print(f"  observation/pose:  {data['observation_pose'].shape}  range: [{data['observation_pose'].min():.4f}, {data['observation_pose'].max():.4f}]")
    print(f"  action/gripper:    {data['action_gripper'].shape}  range: [{data['action_gripper'].min():.4f}, {data['action_gripper'].max():.4f}]")
    print(f"  obs/gripper:       {data['observation_gripper'].shape}  range: [{data['observation_gripper'].min():.4f}, {data['observation_gripper'].max():.4f}]")
    cam_keys = ["rs_front", "rs_wrist", "rs_head"]
    available_cams = [k for k in cam_keys if k in data]
    for key in available_cams:
        print(f"  images/{key}:       {data[key].shape}  dtype: {data[key].dtype}")

    num_cams = len(available_cams)
    if num_cams == 0:
        print("Warning: No camera images found in episode.")

    # --- Build figure ---
    # Layout: top rows for camera images, bottom row for 3 plots
    num_plot_rows = num_cams + 1  # camera rows + 1 plot row
    fig = plt.figure(figsize=(2.5 * num_steps, 2.5 * num_plot_rows))

    # Use gridspec: camera rows get equal height, plot row gets more
    gs = fig.add_gridspec(
        num_plot_rows, num_steps,
        height_ratios=[1] * num_cams + [1.5],
        hspace=0.3, wspace=0.05,
    )

    # --- Camera image grid ---
    for row_idx, cam_key in enumerate(available_cams):
        imgs = data[cam_key]  # (T, 3, H, W) CHW
        for col_idx, t in enumerate(step_indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            # CHW -> HWC for display
            img = np.moveaxis(imgs[t], 0, -1)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(cam_key, fontsize=9)
            if row_idx == 0:
                ax.set_title(f"t={t}", fontsize=8)

    # --- Bottom plots ---
    timesteps = np.arange(T)

    # Action pose XYZ
    ax1 = fig.add_subplot(gs[num_cams, :num_steps // 3])
    ax1.plot(timesteps, data["action_pose"][:, 0], label="x")
    ax1.plot(timesteps, data["action_pose"][:, 1], label="y")
    ax1.plot(timesteps, data["action_pose"][:, 2], label="z")
    ax1.set_title("Action Pose XYZ", fontsize=9)
    ax1.set_xlabel("timestep", fontsize=8)
    ax1.legend(fontsize=7)
    ax1.tick_params(labelsize=7)

    # Observation pose XYZ
    mid_start = num_steps // 3
    mid_end = 2 * num_steps // 3
    ax2 = fig.add_subplot(gs[num_cams, mid_start:mid_end])
    ax2.plot(timesteps, data["observation_pose"][:, 0], label="x")
    ax2.plot(timesteps, data["observation_pose"][:, 1], label="y")
    ax2.plot(timesteps, data["observation_pose"][:, 2], label="z")
    ax2.set_title("Observation Pose XYZ", fontsize=9)
    ax2.set_xlabel("timestep", fontsize=8)
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=7)

    # Gripper
    ax3 = fig.add_subplot(gs[num_cams, mid_end:])
    ax3.plot(timesteps, data["action_gripper"], label="action grip")
    ax3.plot(timesteps, data["observation_gripper"], label="obs grip", linestyle="--")
    ax3.set_title("Gripper", fontsize=9)
    ax3.set_xlabel("timestep", fontsize=8)
    ax3.legend(fontsize=7)
    ax3.tick_params(labelsize=7)

    fig.suptitle(args.episode, fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
