#!/usr/bin/env python3
"""
Zarr Dataset Inspector

Inspects a converted Zarr dataset to show the actual dimensions and data types
of each array so you can configure your task config properly.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table

CONSOLE = Console()

def visualize_first_frames(episode_group, episode_name: str) -> None:
    """
    Visualize the first frame of each image array in the episode.
    
    Args:
        episode_group: Zarr episode group containing the arrays
        episode_name: Name of the episode for display purposes
    """
    # Find all image arrays
    image_arrays = []
    for array_name in episode_group.keys():
        array = episode_group[array_name]
        # Check if this looks like an image array (3D or 4D with reasonable dimensions)
        if len(array.shape) >= 3:
            # Common image array patterns
            if any(keyword in array_name.lower() for keyword in ['rgb', 'color', 'image', 'depth']):
                image_arrays.append(array_name)
            # Also check for arrays that have image-like dimensions
            elif len(array.shape) == 4 and array.shape[-1] in [1, 3, 4]:  # (T, H, W, C)
                image_arrays.append(array_name)
            elif len(array.shape) == 4 and array.shape[1] in [1, 3, 4]:   # (T, C, H, W)
                image_arrays.append(array_name)
            elif len(array.shape) == 3 and min(array.shape[1:]) > 50:     # (T, H, W) - likely depth
                image_arrays.append(array_name)
    
    if not image_arrays:
        CONSOLE.log("[yellow]No image arrays found to visualize")
        return
    
    CONSOLE.log(f"[green]Found {len(image_arrays)} image arrays to visualize: {image_arrays}")
    
    # Set up the plot
    n_images = len(image_arrays)
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    if n_images == 1:
        axes = [axes]  # Make it iterable
    
    fig.suptitle(f'First Frame Visualization - {episode_name}', fontsize=16)
    
    for idx, array_name in enumerate(image_arrays):
        array = episode_group[array_name]
        ax = axes[idx]
        
        try:
            # Get the first frame
            first_frame = array[0]  # Shape: (H, W, C) or (C, H, W) or (H, W)
            
            # Handle different array formats
            if len(first_frame.shape) == 3:
                if first_frame.shape[-1] in [1, 3, 4]:  # (H, W, C) format
                    display_image = first_frame
                    if first_frame.shape[-1] == 1:  # Grayscale
                        display_image = first_frame.squeeze(-1)
                elif first_frame.shape[0] in [1, 3, 4]:  # (C, H, W) format
                    if first_frame.shape[0] == 1:  # Grayscale
                        display_image = first_frame[0]
                    else:  # RGB
                        display_image = first_frame.transpose(1, 2, 0)  # Convert to (H, W, C)
                else:
                    # Assume it's some other 3D format, take first slice
                    display_image = first_frame[:, :, 0]
            elif len(first_frame.shape) == 2:  # (H, W) - likely depth or grayscale
                display_image = first_frame
            else:
                CONSOLE.log(f"[yellow]Cannot visualize {array_name} with shape {first_frame.shape}")
                continue
            
            # Handle different data types and ranges
            if 'depth' in array_name.lower():
                # For depth images, use a colormap and handle potential inf/nan values
                display_image = np.nan_to_num(display_image, nan=0, posinf=0, neginf=0)
                im = ax.imshow(display_image, cmap='viridis')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                # For RGB/color images
                if display_image.dtype == np.uint8:
                    # Already in 0-255 range
                    im = ax.imshow(display_image)
                elif display_image.max() <= 1.0:
                    # Assuming 0-1 range
                    im = ax.imshow(display_image)
                else:
                    # Scale to 0-255 if needed
                    display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min())
                    im = ax.imshow(display_image)
            
            # Set title and info
            shape_str = f"{array.shape}"
            dtype_str = f"{array.dtype}"
            ax.set_title(f'{array_name}\nShape: {shape_str}\nDType: {dtype_str}', fontsize=10)
            ax.axis('off')
            
            # Add value range info
            if len(first_frame.shape) <= 3:
                min_val = np.min(first_frame)
                max_val = np.max(first_frame)
                mean_val = np.mean(first_frame)
                ax.text(0.02, 0.98, f'Range: [{min_val:.2f}, {max_val:.2f}]\nMean: {mean_val:.2f}', 
                       transform=ax.transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
            
        except Exception as e:
            CONSOLE.log(f"[red]Error visualizing {array_name}: {e}")
            ax.text(0.5, 0.5, f'Error visualizing\n{array_name}\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(array_name)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = Path("first_frame_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    CONSOLE.log(f"[green]Visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()

def inspect_zarr_dataset(zarr_path: Path, episode_idx: int = 0, visualize: bool = True) -> None:
    """
    Inspect a single episode from the Zarr dataset to understand data shapes.
    
    Args:
        zarr_path: Path to the Zarr dataset
        episode_idx: Which episode to inspect (default: 0)
        visualize: Whether to create visualizations (default: True)
    """
    
    if not zarr_path.exists():
        CONSOLE.log(f"[red]Zarr path does not exist: {zarr_path}")
        return
    
    try:
        root = zarr.open(zarr_path, mode="r")
        
        # List all available episodes
        episodes = [key for key in root.keys() if key.startswith("episode_")]
        episodes.sort()
        
        if not episodes:
            CONSOLE.log("[red]No episodes found in Zarr dataset!")
            return
        
        CONSOLE.log(f"[green]Found {len(episodes)} episodes in dataset")
        CONSOLE.log(f"[blue]Episodes: {episodes[0]} to {episodes[-1]}")
        
        # Select episode to inspect
        if episode_idx >= len(episodes):
            CONSOLE.log(f"[yellow]Episode index {episode_idx} out of range, using episode 0")
            episode_idx = 0
        
        episode_name = episodes[episode_idx]
        episode_group = root[episode_name]
        
        CONSOLE.log(f"\n[bold blue]Inspecting {episode_name}:")
        
        # Create a table for the results
        table = Table(title=f"Data Shapes in {episode_name}")
        table.add_column("Array Name", style="cyan", no_wrap=True)
        table.add_column("Shape", style="magenta")
        table.add_column("Data Type", style="green")
        table.add_column("Min Value", style="yellow")
        table.add_column("Max Value", style="yellow")
        table.add_column("Notes", style="white")
        
        # Inspect each array in the episode
        for array_name in sorted(episode_group.keys()):
            array = episode_group[array_name]
            
            # Get basic info
            shape_str = str(array.shape)
            dtype_str = str(array.dtype)
            
            # Calculate min/max for small arrays or sample for large ones
            try:
                if array.size < 1000000:  # For reasonably sized arrays
                    min_val = np.min(array[:])
                    max_val = np.max(array[:])
                else:  # For very large arrays, sample
                    sample = array[:10] if len(array.shape) > 0 else array[:]
                    min_val = np.min(sample)
                    max_val = np.max(sample)
                min_str = f"{min_val:.3f}" if isinstance(min_val, (float, np.floating)) else str(min_val)
                max_str = f"{max_val:.3f}" if isinstance(max_val, (float, np.floating)) else str(max_val)
            except Exception as e:
                min_str = "N/A"
                max_str = "N/A"
            
            # Add notes based on array name and properties
            notes = ""
            if array_name in ["rs_rgb", "zed_rgb", "rs_color_images"]:
                notes = "RGB images"
            elif array_name in ["rs_depth", "zed_depth"]:
                notes = "Depth images"
            elif array_name == "zed_pcd":
                notes = "Point clouds (T, N_points, 6) - [x,y,z,r,g,b]"
            elif array_name in ["agent_pos", "pose"]:
                notes = "Robot poses (T, 7) - [x,y,z,qw,qx,qy,qz]"
            elif array_name == "action":
                notes = "Actions (T, 7) - pose deltas"
            
            table.add_row(array_name, shape_str, dtype_str, min_str, max_str, notes)
        
        CONSOLE.print(table)
        
        # Print episode attributes
        if hasattr(episode_group, 'attrs') and episode_group.attrs:
            CONSOLE.log(f"\n[bold blue]Episode Attributes:")
            for attr_name, attr_value in episode_group.attrs.items():
                CONSOLE.log(f"  {attr_name}: {attr_value}")
        
        # Visualize first frames if requested
        if visualize:
            CONSOLE.log(f"\n[bold green]Creating visualizations...")
            visualize_first_frames(episode_group, episode_name)
        
        # Generate suggested config
        generate_config_suggestion(episode_group)
        
    except Exception as e:
        CONSOLE.log(f"[red]Error inspecting Zarr dataset: {e}")
        import traceback
        traceback.print_exc()

def generate_config_suggestion(episode_group) -> None:
    """Generate a suggested configuration based on the actual data shapes."""
    
    CONSOLE.log(f"\n[bold green]Suggested Configuration Updates:")
    
    # Check RGB shapes - updated to handle rs_color_images
    rgb_arrays = ["rs_rgb", "zed_rgb", "rs_color_images"]
    rgb_found = False
    for rgb_name in rgb_arrays:
        if rgb_name in episode_group:
            rgb_shape = episode_group[rgb_name].shape
            if len(rgb_shape) == 4:  # (T, H, W, C) or (T, C, H, W)
                if rgb_shape[-1] in [1, 3, 4]:  # (T, H, W, C)
                    h, w, c = rgb_shape[1], rgb_shape[2], rgb_shape[3]
                elif rgb_shape[1] in [1, 3, 4]:  # (T, C, H, W)
                    c, h, w = rgb_shape[1], rgb_shape[2], rgb_shape[3]
                else:
                    continue
                CONSOLE.log(f"RGB images ({rgb_name}): {h}x{w}x{c}")
                CONSOLE.log(f"  Current config: [2, 84, 84, 3]")
                CONSOLE.log(f"  Suggested:      [2, {h}, {w}, {c}]")
                rgb_found = True
                break
    
    # Check depth shapes
    depth_arrays = ["rs_depth", "zed_depth"]
    for depth_name in depth_arrays:
        if depth_name in episode_group:
            depth_shape = episode_group[depth_name].shape
            if len(depth_shape) == 3:  # (T, H, W)
                h, w = depth_shape[1], depth_shape[2]
                CONSOLE.log(f"Depth images ({depth_name}): {h}x{w}")
                CONSOLE.log(f"  Current config: [2, 84, 84]")
                CONSOLE.log(f"  Suggested:      [2, {h}, {w}]")
                break
    
    # Check point cloud shape
    if "zed_pcd" in episode_group:
        pcd_shape = episode_group["zed_pcd"].shape
        if len(pcd_shape) == 3:  # (T, N_points, features)
            n_points, features = pcd_shape[1], pcd_shape[2]
            CONSOLE.log(f"Point clouds: {n_points} points x {features} features")
            CONSOLE.log(f"  Current config: [1024, 6]")
            CONSOLE.log(f"  Suggested:      [{n_points}, {features}]")
    
    # Check agent_pos/pose shape
    pos_arrays = ["agent_pos", "pose"]
    for pos_name in pos_arrays:
        if pos_name in episode_group:
            pos_shape = episode_group[pos_name].shape
            if len(pos_shape) == 2:  # (T, features)
                features = pos_shape[1]
                CONSOLE.log(f"Agent positions ({pos_name}): {features} features")
                CONSOLE.log(f"  Current config: [7]")
                CONSOLE.log(f"  Suggested:      [{features}]")
                break
    
    # Check action shape
    if "action" in episode_group:
        action_shape = episode_group["action"].shape
        if len(action_shape) == 2:  # (T, features)
            features = action_shape[1]
            CONSOLE.log(f"Actions: {features} features")
            CONSOLE.log(f"  Current config: [7]")
            CONSOLE.log(f"  Suggested:      [{features}]")
    
    # Print complete suggested config
    CONSOLE.log(f"\n[bold cyan]Complete Suggested shape_meta:")
    print_suggested_yaml_config(episode_group)

def print_suggested_yaml_config(episode_group) -> None:
    """Print a complete YAML config suggestion."""
    
    config_lines = ["shape_meta:"]
    
    # RGB config - check for rs_color_images first, then fallback to rs_rgb
    rgb_arrays = ["rs_color_images", "rs_rgb"]
    for rgb_name in rgb_arrays:
        if rgb_name in episode_group:
            rgb_shape = episode_group[rgb_name].shape
            if len(rgb_shape) == 4:
                if rgb_shape[-1] in [1, 3, 4]:  # (T, H, W, C)
                    h, w, c = rgb_shape[1], rgb_shape[2], rgb_shape[3]
                elif rgb_shape[1] in [1, 3, 4]:  # (T, C, H, W)
                    c, h, w = rgb_shape[1], rgb_shape[2], rgb_shape[3]
                else:
                    continue
                config_lines.extend([
                    "  rgb:",
                    f"    shape: [2, {h}, {w}, {c}]",
                    "    dtype: uint8"
                ])
                break
    
    # Depth config
    depth_arrays = ["rs_depth", "zed_depth"]
    for depth_name in depth_arrays:
        if depth_name in episode_group:
            depth_shape = episode_group[depth_name].shape
            if len(depth_shape) == 3:
                h, w = depth_shape[1], depth_shape[2]
                config_lines.extend([
                    "  depth:",
                    f"    shape: [2, {h}, {w}]",
                    "    dtype: float32"
                ])
                break
    
    # Point cloud config
    if "zed_pcd" in episode_group:
        pcd_shape = episode_group["zed_pcd"].shape
        if len(pcd_shape) == 3:
            n_points, features = pcd_shape[1], pcd_shape[2]
            config_lines.extend([
                "  point_cloud:",
                f"    shape: [{n_points}, {features}]",
                "    dtype: float32"
            ])
    
    # Agent position config - check both pose and agent_pos
    pos_arrays = ["pose", "agent_pos"]
    for pos_name in pos_arrays:
        if pos_name in episode_group:
            pos_shape = episode_group[pos_name].shape
            if len(pos_shape) == 2:
                features = pos_shape[1]
                config_lines.extend([
                    "  agent_pos:",
                    f"    shape: [{features}]       # [x y z qw qx qy qz]",
                    "    dtype: float32"
                ])
                break
    
    # Action config
    if "action" in episode_group:
        action_shape = episode_group["action"].shape
        if len(action_shape) == 2:
            features = action_shape[1]
            config_lines.extend([
                "  action:",
                f"    shape: [{features}]",
                "    dtype: float32"
            ])
    
    # Print the config
    for line in config_lines:
        CONSOLE.log(f"[cyan]{line}")

def main():
    """Main function to run the inspector."""
    
    # Path to your Zarr dataset
    ZARR_PATH = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/2d_strawberry_baseline/10_hz_baseline_100_zarr_both_crop_no_jitter")
    
    # Inspect the first episode (you can change this)
    EPISODE_IDX = 0
    
    # Whether to create visualizations (set to False to skip)
    VISUALIZE = True
    
    CONSOLE.log(f"[bold blue]Inspecting Zarr Dataset")
    CONSOLE.log(f"Path: {ZARR_PATH}")
    CONSOLE.log(f"Episode: {EPISODE_IDX}")
    CONSOLE.log(f"Visualize: {VISUALIZE}")
    
    inspect_zarr_dataset(ZARR_PATH, EPISODE_IDX, VISUALIZE)

if __name__ == "__main__":
    main()