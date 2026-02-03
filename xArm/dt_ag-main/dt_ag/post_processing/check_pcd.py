#!/usr/bin/env python3
"""
Visualize point cloud from the first frame of a zarr episode
"""

import numpy as np
import zarr
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
ZARR_PATH = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/3d_strawberry_baseline/new_setup_100_baseline_zarr"
EPISODE_NAME = "episode_0000"  # Change this to the episode you want to visualize

# Camera intrinsics - ZED
ZED_FX = 1069.73
ZED_FY = 1069.73
ZED_CX = 1135.86
ZED_CY = 680.69

# ZED to xArm base transform
T_BASE_ZED = np.array([
    [-0.4949434,  0.2645439, -0.8276760, 0.590],
    [ 0.8685291,  0.1218584, -0.4804246, 0.520],
    [-0.0262341, -0.9566436, -0.2900771, 0.310],
    [ 0.000,      0.000,      0.000,      1.000],
], dtype=np.float32)  # Radians

## Identity transform for testing
# T_BASE_ZED = np.eye(4, dtype=np.float32)

def depth_to_point_cloud(depth, rgb, fx, fy, cx, cy, T_base_cam):
    """
    Convert depth image to point cloud in xArm base frame
    
    Args:
        depth: (H, W) depth image in meters
        rgb: (H, W, 3) RGB image
        fx, fy, cx, cy: camera intrinsics
        T_base_cam: 4x4 transformation matrix from camera to base
    
    Returns:
        point cloud as (N, 6) array with XYZRGB
    """
    h, w = depth.shape
    
    # Create pixel coordinate grid
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    # Get valid depth points
    valid_mask = (depth > 0) & np.isfinite(depth)
    
    if not np.any(valid_mask):
        return np.empty((0, 6), dtype=np.float32)
    
    # Extract valid coordinates
    u_valid = u_grid[valid_mask]
    v_valid = v_grid[valid_mask]
    Z = depth[valid_mask]
    
    # Back-project to 3D
    X = (u_valid - cx) * Z / fx
    Y = (v_valid - cy) * Z / fy
    
    # Stack into homogeneous coordinates
    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)  # (N, 4)
    
    # Transform to base frame
    pts_base = (T_base_cam @ pts_cam.T).T[:, :3]  # (N, 3)
    
    # Get colors
    colors = rgb[valid_mask] / 255.0  # Normalize to [0, 1]
    
    # Combine position and color
    points = np.concatenate([pts_base, colors], axis=1).astype(np.float32)
    
    return points


def visualize_pointcloud(points):
    """
    Visualize point cloud using Open3D
    
    Args:
        points: (N, 6) array with XYZRGB
    """
    if points.shape[0] == 0:
        print("No valid points to visualize!")
        return
        
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:])
    
    # Create a coordinate frame at the origin (xArm base)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    
    # Visualize
    print(f"Visualizing {points.shape[0]} points...")
    print("Controls:")
    print("  - Mouse: Rotate")
    print("  - Ctrl + Mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Q: Close window")
    
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name="Point Cloud in xArm Base Frame",
        width=1280,
        height=720
    )


def visualize_rgbd_images(rgb, depth):
    """
    Visualize RGB and depth images side by side
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # RGB image
    ax1.imshow(rgb)
    ax1.set_title("RGB Image")
    ax1.axis('off')
    
    # Depth image
    depth_vis = depth.copy()
    valid_mask = (depth_vis > 0) & np.isfinite(depth_vis)
    if np.any(valid_mask):
        vmin, vmax = np.percentile(depth_vis[valid_mask], [1, 99])
        depth_vis = np.clip(depth_vis, vmin, vmax)
    
    im = ax2.imshow(depth_vis, cmap='viridis')
    ax2.set_title("Depth Image")
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, label='Depth (m)')
    
    plt.tight_layout()
    plt.show()


def main():
    # Load zarr data
    zarr_path = Path(ZARR_PATH)
    episode_path = zarr_path / EPISODE_NAME
    
    if not episode_path.exists():
        print(f"Episode {EPISODE_NAME} not found at {zarr_path}")
        return
    
    print(f"Loading data from {episode_path}")
    
    # Open zarr group
    episode = zarr.open(str(episode_path), mode='r')
    
    # Load first frame data
    # RGB data: (T, 2, H, W, 3) - index 1 is ZED
    rgb_data = episode['zed_rgb'][0]  # First frame, ZED camera
    
    # Depth data: (T, 2, H, W) - index 1 is ZED
    depth_data = episode['zed_depth'][0]  # First frame, ZED camera

    # import pdb; pdb.set_trace()
    
    # Convert from BGR to RGB if needed
    if rgb_data.shape[-1] == 3:
        rgb_data = rgb_data[..., ::-1]  # BGR to RGB
    
    print(f"RGB shape: {rgb_data.shape}")
    print(f"Depth shape: {depth_data.shape}")
    # print(f"Depth range: {depth_data.min():.3f} to {depth_data.max():.3f} meters")
    
    # Visualize RGB and depth images
    visualize_rgbd_images(rgb_data, depth_data)
    
    # Generate point cloud
    points = depth_to_point_cloud(
        depth_data, rgb_data,
        ZED_FX, ZED_FY, ZED_CX, ZED_CY,
        T_BASE_ZED
    )
    
    print(f"Generated {points.shape[0]} points")
    
    # Add comparison with stored point cloud if available
    if 'point_cloud' in episode:
        stored_pc = episode['point_cloud'][0, 1]  # First frame, ZED camera
        print(f"Stored point cloud shape: {stored_pc.shape}")
        
        # Visualize both for comparison
        print("\nVisualizing generated point cloud...")
        visualize_pointcloud(points)
        
        print("\nVisualizing stored point cloud...")
        visualize_pointcloud(stored_pc)
    else:
        # Visualize generated point cloud
        visualize_pointcloud(points)


if __name__ == "__main__":
    main()