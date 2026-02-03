#!/usr/bin/env python3
"""
Point Cloud Visualization Script
================================

This script loads and visualizes .npy point cloud files using Open3D.
It displays the point cloud with proper RGB values and a coordinate frame.
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path


def visualize_pcd(pcd_file: str, point_size: float = 2.0):
    """
    Visualize a point cloud from a .npy file.
    
    Args:
        pcd_file: Path to the .npy file containing point cloud data
        point_size: Size of points in visualization
    """
    # Load the point cloud data
    print(f"Loading point cloud from: {pcd_file}")
    pc_data = np.load(pcd_file)
    
    if pc_data.size == 0:
        print("Error: Empty point cloud data")
        return
    
    print(f"Point cloud shape: {pc_data.shape}")
    
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Check data format and extract points and colors
    if pc_data.shape[1] >= 6:  # XYZ + RGB format
        # Extract XYZ coordinates
        points = pc_data[:, :3]
        
        # Extract RGB values (assuming they're in the range [0,1])
        colors = pc_data[:, 3:6]
        
        # Ensure colors are in range [0,1]
        if np.max(colors) > 1.0:
            colors = colors / 255.0
            
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Just XYZ coordinates
        pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
        # Use default color
        pcd.paint_uniform_color([0.5, 0.5, 1.0])
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud: {os.path.basename(pcd_file)}", width=1024, height=768)
    
    # Add geometries
    vis.add_geometry(pcd)
    
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coordinate_frame)
    
    # Configure renderer
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = point_size
    
    # Set initial viewpoint (optional)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    print("Visualizing point cloud. Press 'q' or 'ESC' to exit.")
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud from .npy file")
    parser.add_argument("file", nargs="?", help="Path to .npy file containing point cloud data")
    parser.add_argument("--point-size", type=float, default=2.0, help="Size of points in visualization")
    args = parser.parse_args()
    
    # If no file is provided, look for .npy files in the current directory
    if args.file is None:
        npy_files = list(Path('.').glob('*.npy'))
        if not npy_files:
            print("Error: No .npy files found in the current directory")
            return
        
        if len(npy_files) == 1:
            pcd_file = str(npy_files[0])
        else:
            print("Multiple .npy files found. Please select one:")
            for i, file in enumerate(npy_files):
                print(f"[{i}] {file}")
            
            try:
                selection = int(input("Enter file number: "))
                pcd_file = str(npy_files[selection])
            except (ValueError, IndexError):
                print("Invalid selection")
                return
    else:
        pcd_file = args.file
        
    visualize_pcd(pcd_file, args.point_size)


if __name__ == "__main__":
    main()
