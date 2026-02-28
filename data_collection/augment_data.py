#!/usr/bin/env python3
"""
Data Augmentation Script for Teleoperation Episodes

Generates augmented episodes by combining the robot workspace (right side)
from one episode with the human workspace (left side) from another episode.
The front and head cameras capture both workspaces side-by-side, so cropping
at a configurable pixel column separates them. The wrist camera shows only
the robot's perspective and is kept entirely from the robot episode.

Episodes are grouped by end-state condition (horizontal / vertical), specified
by collection name. Only cross-collection pairs are generated (no pairing
within the same collection). An expansion ratio controls how many augmented
episodes are sampled relative to the original episode count.

Usage:
    # Preview crop boundaries
    python3 augment_data.py --preview \
        --collections Collection4 Collection5 Collection6 Collection7 \
        --front-crop-x 250 --head-crop-x 220

    # Dry run
    python3 augment_data.py --dry-run \
        --collections Collection4 Collection5 Collection6 Collection7 \
        --horizontal Collection4 Collection6 \
        --vertical Collection5 Collection7 \
        --front-crop-x 250 --head-crop-x 220 \
        --expansion-ratio 2.0

    # Full augmentation
    python3 augment_data.py \
        --collections Collection4 Collection5 Collection6 Collection7 \
        --horizontal Collection4 Collection6 \
        --vertical Collection5 Collection7 \
        --front-crop-x 250 --head-crop-x 220 \
        --expansion-ratio 2.0 \
        --blend-width 10 \
        --output-dir demo_data/Augmented
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np
from natsort import natsorted


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------

def discover_episodes(
    base_dir: str, collection_names: List[str]
) -> Tuple[List[str], Dict[int, str]]:
    """Build a globally-indexed list of episode HDF5 paths.

    Scans collection directories in natsorted order, matching how
    convert_data.py flattens episodes. The list index IS the global index.

    Returns:
        episode_paths: list of HDF5 file paths, index = global episode index
        collection_map: {global_index: collection_name}
    """
    collections = natsorted(collection_names)
    all_paths: List[str] = []
    collection_map: Dict[int, str] = {}

    print(f"Scanning {base_dir} for collections: {collections}")
    print(f"{'Collection':<20} {'Episodes':<10} {'Global range'}")
    print("-" * 50)

    for coll in collections:
        coll_dir = os.path.join(base_dir, coll)
        if not os.path.isdir(coll_dir):
            print(f"  WARNING: {coll_dir} not found, skipping")
            continue

        h5_files = natsorted(glob.glob(os.path.join(coll_dir, "episode_*.hdf5")))
        start_idx = len(all_paths)
        for i, path in enumerate(h5_files):
            collection_map[start_idx + i] = coll
        all_paths.extend(h5_files)
        end_idx = len(all_paths) - 1

        if h5_files:
            print(f"{coll:<20} {len(h5_files):<10} {start_idx}-{end_idx}")
        else:
            print(f"{coll:<20} {'0':<10} (empty)")

    print(f"\nTotal: {len(all_paths)} episodes\n")
    return all_paths, collection_map


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def load_episode_raw(path: str) -> dict:
    """Load all datasets from an HDF5 episode into memory.

    Returns dict with keys matching HDF5 paths:
        'action/pose'         (T, 7) float32
        'action/gripper'      (T,) float32
        'observation/pose'    (T, 7) float32
        'observation/gripper' (T,) float32
        'hand/pose'           (T, 7) float32
        'images/rs_front'     (T, 3, 360, 640) uint8 CHW
        'images/rs_wrist'     (T, 3, 360, 640) uint8 CHW
        'images/rs_head'      (T, 3, 360, 640) uint8 CHW
    """
    data = {}
    with h5py.File(path, "r") as f:
        for key in [
            "action/pose", "action/gripper",
            "observation/pose", "observation/gripper",
            "hand/pose",
            "images/rs_front", "images/rs_wrist", "images/rs_head",
        ]:
            if key in f:
                data[key] = f[key][()]
    return data


def save_augmented_episode(
    data: dict,
    output_path: str,
    episode_index: int,
    source_robot: str,
    source_human: str,
) -> None:
    """Save an augmented episode to HDF5 matching the collector format.

    Images use LZF compression. Pose/gripper have no compression.
    File attrs: num_frames, collection_rate_hz, episode_index, provenance.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    T = data["action/pose"].shape[0]

    with h5py.File(output_path, "w") as f:
        # Action group
        action_grp = f.create_group("action")
        action_grp.create_dataset("pose", data=data["action/pose"])
        action_grp.create_dataset("gripper", data=data["action/gripper"])

        # Observation group
        obs_grp = f.create_group("observation")
        obs_grp.create_dataset("pose", data=data["observation/pose"])
        obs_grp.create_dataset("gripper", data=data["observation/gripper"])

        # Hand group
        hand_grp = f.create_group("hand")
        hand_grp.create_dataset("pose", data=data["hand/pose"])

        # Images with LZF compression
        images_grp = f.create_group("images")
        images_grp.create_dataset("rs_front", data=data["images/rs_front"], compression="lzf")
        images_grp.create_dataset("rs_wrist", data=data["images/rs_wrist"], compression="lzf")
        images_grp.create_dataset("rs_head", data=data["images/rs_head"], compression="lzf")

        # Metadata
        f.attrs["num_frames"] = T
        f.attrs["collection_rate_hz"] = 30
        f.attrs["episode_index"] = episode_index
        f.attrs["augmented"] = True
        f.attrs["source_robot"] = source_robot
        f.attrs["source_human"] = source_human


# ---------------------------------------------------------------------------
# Temporal resampling
# ---------------------------------------------------------------------------

def resample_nearest(data: np.ndarray, target_length: int) -> np.ndarray:
    """Resample array along axis 0 to target_length via nearest-neighbor."""
    source_length = data.shape[0]
    if source_length == target_length:
        return data.copy()
    indices = np.round(np.linspace(0, source_length - 1, target_length)).astype(int)
    return data[indices]


# ---------------------------------------------------------------------------
# Image splicing
# ---------------------------------------------------------------------------

def splice_images(
    robot_images: np.ndarray,
    human_images: np.ndarray,
    crop_x: int,
    blend_width: int = 0,
) -> np.ndarray:
    """Splice robot (right) and human (left) halves of CHW images.

    Args:
        robot_images:  (T, 3, H, W) uint8 — source for columns [crop_x:].
        human_images:  (T, 3, H, W) uint8 — source for columns [:crop_x].
                       Must already be resampled to same T as robot_images.
        crop_x:        Column index for the splice boundary.
        blend_width:   Alpha-blend zone width centered on crop_x (0 = hard cut).
    """
    T, C, H, W = robot_images.shape
    out = np.empty_like(robot_images)

    # Copy human (left) and robot (right) halves
    out[:, :, :, :crop_x] = human_images[:, :, :, :crop_x]
    out[:, :, :, crop_x:] = robot_images[:, :, :, crop_x:]

    # Optional alpha blend at the seam
    if blend_width > 0:
        half = blend_width // 2
        x_start = max(0, crop_x - half)
        x_end = min(W, crop_x + half)
        zone_w = x_end - x_start
        if zone_w > 1:
            # Alpha ramp: 0 = fully human, 1 = fully robot
            alpha = np.linspace(0.0, 1.0, zone_w, dtype=np.float32)
            # Broadcast: (zone_w,) -> (1, 1, 1, zone_w) for CHW
            alpha = alpha[np.newaxis, np.newaxis, np.newaxis, :]
            blended = (
                (1.0 - alpha) * human_images[:, :, :, x_start:x_end].astype(np.float32)
                + alpha * robot_images[:, :, :, x_start:x_end].astype(np.float32)
            )
            out[:, :, :, x_start:x_end] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


# ---------------------------------------------------------------------------
# Augmented episode assembly
# ---------------------------------------------------------------------------

def create_augmented_episode(
    robot_ep: dict,
    human_ep: dict,
    front_crop_x: int,
    head_crop_x: int,
    blend_width: int = 0,
) -> dict:
    """Combine robot data from one episode with human visuals from another.

    Robot episode provides: action, observation, wrist camera, right side of
    front/head cameras. Human episode provides: left side of front/head cameras,
    hand pose. Output length matches robot episode.
    """
    T_robot = robot_ep["action/pose"].shape[0]

    # Resample human data to match robot length
    human_front = resample_nearest(human_ep["images/rs_front"], T_robot)
    human_head = resample_nearest(human_ep["images/rs_head"], T_robot)
    human_hand_pose = resample_nearest(human_ep["hand/pose"], T_robot)

    # Splice front and head cameras
    spliced_front = splice_images(
        robot_ep["images/rs_front"], human_front, front_crop_x, blend_width
    )
    spliced_head = splice_images(
        robot_ep["images/rs_head"], human_head, head_crop_x, blend_width
    )

    return {
        "action/pose": robot_ep["action/pose"].copy(),
        "action/gripper": robot_ep["action/gripper"].copy(),
        "observation/pose": robot_ep["observation/pose"].copy(),
        "observation/gripper": robot_ep["observation/gripper"].copy(),
        "hand/pose": human_hand_pose,
        "images/rs_front": spliced_front,
        "images/rs_wrist": robot_ep["images/rs_wrist"].copy(),
        "images/rs_head": spliced_head,
    }


def create_baseline_episode(
    episode: dict,
    front_crop_x: int,
    head_crop_x: int,
    blend_width: int = 0,
) -> dict:
    """Create a baseline episode with the human side frozen at frame 0.

    Simulates no human being present by replacing the human (left) side of
    front and head cameras with the first frame repeated for the entire
    duration. All robot data (action, observation, wrist camera, right side
    of front/head) is kept unchanged. Hand pose is frozen at frame 0.
    """
    T = episode["action/pose"].shape[0]

    # Create frozen human images: repeat frame 0 for all T frames
    frozen_front = np.broadcast_to(
        episode["images/rs_front"][0:1], (T,) + episode["images/rs_front"].shape[1:]
    ).copy()
    frozen_head = np.broadcast_to(
        episode["images/rs_head"][0:1], (T,) + episode["images/rs_head"].shape[1:]
    ).copy()

    # Frozen hand pose: repeat frame 0
    frozen_hand = np.broadcast_to(
        episode["hand/pose"][0:1], (T,) + episode["hand/pose"].shape[1:]
    ).copy()

    # Splice with frozen human side
    spliced_front = splice_images(
        episode["images/rs_front"], frozen_front, front_crop_x, blend_width
    )
    spliced_head = splice_images(
        episode["images/rs_head"], frozen_head, head_crop_x, blend_width
    )

    return {
        "action/pose": episode["action/pose"].copy(),
        "action/gripper": episode["action/gripper"].copy(),
        "observation/pose": episode["observation/pose"].copy(),
        "observation/gripper": episode["observation/gripper"].copy(),
        "hand/pose": frozen_hand,
        "images/rs_front": spliced_front,
        "images/rs_wrist": episode["images/rs_wrist"].copy(),
        "images/rs_head": spliced_head,
    }


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_cross_collection_pairs(
    group_collections: List[str],
    collection_map: Dict[int, str],
    episode_paths: List[str],
) -> List[Tuple[int, int]]:
    """Generate all cross-collection ordered pairs within a condition group.

    Only pairs episodes from different collections. For collections A and B
    with Na and Nb episodes: generates Na*Nb + Nb*Na = 2*Na*Nb ordered pairs.
    """
    # Group episode indices by collection
    coll_to_indices: Dict[str, List[int]] = {}
    for idx, coll in collection_map.items():
        if coll in group_collections:
            coll_to_indices.setdefault(coll, []).append(idx)

    # Sort indices within each collection for determinism
    for coll in coll_to_indices:
        coll_to_indices[coll].sort()

    # Generate cross-collection pairs
    colls = sorted(coll_to_indices.keys())
    pairs: List[Tuple[int, int]] = []
    for i, coll_a in enumerate(colls):
        for j, coll_b in enumerate(colls):
            if i == j:
                continue
            for robot_idx in coll_to_indices[coll_a]:
                for human_idx in coll_to_indices[coll_b]:
                    pairs.append((robot_idx, human_idx))

    return pairs


def sample_pairs(
    all_pairs: List[Tuple[int, int]],
    num_original_episodes: int,
    expansion_ratio: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """Sample pairs according to expansion ratio.

    num_augmented = num_original_episodes * expansion_ratio (rounded).
    If that exceeds the pair pool, uses all pairs.
    """
    num_augmented = round(num_original_episodes * expansion_ratio)
    if num_augmented >= len(all_pairs):
        print(f"  Expansion ratio requests {num_augmented} pairs, "
              f"but only {len(all_pairs)} cross-collection pairs exist. Using all.")
        # Shuffle for variety in ordering
        selected = list(all_pairs)
        rng.shuffle(selected)
        return selected

    indices = rng.choice(len(all_pairs), size=num_augmented, replace=False)
    return [all_pairs[i] for i in sorted(indices)]


# ---------------------------------------------------------------------------
# Preview mode
# ---------------------------------------------------------------------------

def preview_crop_boundaries(
    episode_paths: List[str],
    front_crop_x: int,
    head_crop_x: int,
    num_episodes: int = 3,
    num_frames: int = 3,
) -> None:
    """Display sample frames with crop boundaries drawn as red lines."""
    # Sample episodes evenly
    n = min(num_episodes, len(episode_paths))
    sample_indices = np.linspace(0, len(episode_paths) - 1, n, dtype=int)

    rows = []
    for ep_idx in sample_indices:
        path = episode_paths[ep_idx]
        with h5py.File(path, "r") as f:
            T = f["images/rs_front"].shape[0]
            frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

            row_imgs = []
            for fi in frame_indices:
                # Front camera
                front_chw = f["images/rs_front"][fi]  # (3, H, W)
                front_hwc = np.moveaxis(front_chw, 0, -1)  # (H, W, 3)
                front_bgr = cv2.cvtColor(front_hwc, cv2.COLOR_RGB2BGR)
                cv2.line(front_bgr, (front_crop_x, 0), (front_crop_x, front_bgr.shape[0] - 1), (0, 0, 255), 2)
                cv2.putText(front_bgr, f"ep{ep_idx} f{fi} FRONT", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Head camera
                head_chw = f["images/rs_head"][fi]
                head_hwc = np.moveaxis(head_chw, 0, -1)
                head_bgr = cv2.cvtColor(head_hwc, cv2.COLOR_RGB2BGR)
                cv2.line(head_bgr, (head_crop_x, 0), (head_crop_x, head_bgr.shape[0] - 1), (0, 0, 255), 2)
                cv2.putText(head_bgr, f"ep{ep_idx} f{fi} HEAD", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Stack front and head vertically
                row_imgs.append(np.vstack([front_bgr, head_bgr]))

            rows.append(np.hstack(row_imgs))

    grid = np.vstack(rows)

    # Resize if too large for display
    max_h = 1080
    if grid.shape[0] > max_h:
        scale = max_h / grid.shape[0]
        grid = cv2.resize(grid, None, fx=scale, fy=scale)

    cv2.imshow("Crop Preview (press any key to close)", grid)
    print("Showing crop preview. Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def print_dry_run(
    groups: Dict[str, List[Tuple[int, int]]],
    episode_paths: List[str],
    output_dir: str,
) -> None:
    """Print what pairs would be generated without processing."""
    total = 0
    ep_counter = 0

    for group_name, pairs in groups.items():
        print(f"\n--- {group_name.upper()} group: {len(pairs)} pairs ---")
        for robot_idx, human_idx in pairs:
            robot_base = os.path.basename(os.path.dirname(episode_paths[robot_idx]))
            robot_file = os.path.basename(episode_paths[robot_idx])
            human_base = os.path.basename(os.path.dirname(episode_paths[human_idx]))
            human_file = os.path.basename(episode_paths[human_idx])
            out_name = f"episode_{ep_counter}.hdf5"
            print(f"  {out_name}: robot={robot_base}/{robot_file} + human={human_base}/{human_file}")
            ep_counter += 1
        total += len(pairs)

    print(f"\nTotal augmented episodes: {total}")
    print(f"Output directory: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Augment episodes by splicing human/robot camera halves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--collections", nargs="+", required=True,
                        help="Collection directory names (e.g., Collection4 Collection5)")
    parser.add_argument("--horizontal", nargs="+", default=None,
                        help="Collection names for the horizontal condition group")
    parser.add_argument("--vertical", nargs="+", default=None,
                        help="Collection names for the vertical condition group")
    parser.add_argument("--front-crop-x", type=int, required=True,
                        help="Pixel column for front camera crop boundary (0-639)")
    parser.add_argument("--head-crop-x", type=int, required=True,
                        help="Pixel column for head camera crop boundary (0-639)")
    parser.add_argument("--expansion-ratio", type=float, default=1.0,
                        help="Augmented episodes per original episode in each group (default: 1.0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for augmented HDF5 files")
    parser.add_argument("--blend-width", type=int, default=0,
                        help="Pixel blending zone at seam (default: 0 = hard cut)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for pair sampling (default: 42)")
    parser.add_argument("--baseline", action="store_true",
                        help="Generate baseline episodes with human side frozen at frame 0")
    parser.add_argument("--preview", action="store_true",
                        help="Show sample frames with crop lines (visual check)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pairs without processing data")
    parser.add_argument("--base-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data"),
                        help="Base directory containing collection folders")
    args = parser.parse_args()

    # Discover episodes
    episode_paths, collection_map = discover_episodes(args.base_dir, args.collections)
    if not episode_paths:
        print("Error: No episodes found.")
        sys.exit(1)

    # Validate crop boundaries
    for name, val in [("front-crop-x", args.front_crop_x), ("head-crop-x", args.head_crop_x)]:
        if val < 1 or val > 638:
            print(f"Error: --{name} must be in [1, 638], got {val}")
            sys.exit(1)

    # Validate blend width
    if args.blend_width < 0:
        print("Error: --blend-width must be >= 0")
        sys.exit(1)

    # Validate expansion ratio
    if args.expansion_ratio <= 0:
        print("Error: --expansion-ratio must be > 0")
        sys.exit(1)

    # Preview mode
    if args.preview:
        preview_crop_boundaries(episode_paths, args.front_crop_x, args.head_crop_x)
        return

    # Parse condition groups
    if args.horizontal is None and args.vertical is None:
        print("Error: Provide at least one of --horizontal or --vertical")
        sys.exit(1)

    # Validate that group collections are a subset of --collections
    all_collections = set(collection_map.values())
    for flag, group_colls in [("--horizontal", args.horizontal), ("--vertical", args.vertical)]:
        if group_colls is not None:
            for coll in group_colls:
                if coll not in all_collections:
                    print(f"Error: {flag} references '{coll}' which is not in --collections "
                          f"or has no episodes. Available: {sorted(all_collections)}")
                    sys.exit(1)
            if not args.baseline and len(group_colls) < 2:
                print(f"Error: {flag} needs at least 2 collections for cross-collection pairing, "
                      f"got {len(group_colls)}")
                sys.exit(1)

    # -----------------------------------------------------------------------
    # Baseline mode: freeze human side at frame 0 for each episode
    # -----------------------------------------------------------------------
    if args.baseline:
        # Collect all episode indices in the specified groups
        baseline_indices: List[int] = []
        for group_name, group_colls in [("horizontal", args.horizontal), ("vertical", args.vertical)]:
            if group_colls is None:
                continue
            group_eps = sorted(
                idx for idx, coll in collection_map.items() if coll in group_colls
            )
            print(f"{group_name.upper()} group: {len(group_eps)} episodes for baseline")
            baseline_indices.extend(group_eps)

        # Deduplicate and sort
        baseline_indices = sorted(set(baseline_indices))
        print(f"\nTotal baseline episodes to generate: {len(baseline_indices)}")

        # Dry run for baseline
        if args.dry_run:
            output_dir = args.output_dir if args.output_dir else "(not specified)"
            print(f"Output directory: {output_dir}\n")
            for i, idx in enumerate(baseline_indices):
                ep_base = os.path.basename(os.path.dirname(episode_paths[idx]))
                ep_file = os.path.basename(episode_paths[idx])
                print(f"  episode_{i}.hdf5: source={ep_base}/{ep_file} (human frozen at frame 0)")
            return

        # Full run requires output dir
        if args.output_dir is None:
            print("Error: --output-dir is required for baseline (or use --dry-run)")
            sys.exit(1)

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        existing = glob.glob(os.path.join(output_dir, "episode_*.hdf5"))
        if existing:
            print(f"Warning: {output_dir} already contains {len(existing)} episode files. "
                  "They may be overwritten.")

        total = len(baseline_indices)
        for i, idx in enumerate(baseline_indices):
            ep_name = os.path.join(
                os.path.basename(os.path.dirname(episode_paths[idx])),
                os.path.basename(episode_paths[idx]),
            )
            print(f"[{i+1}/{total}] source={ep_name} (frozen human)")

            episode = load_episode_raw(episode_paths[idx])
            T = episode["action/pose"].shape[0]
            print(f"  T={T}")

            baseline_ep = create_baseline_episode(
                episode, args.front_crop_x, args.head_crop_x, args.blend_width,
            )

            out_path = os.path.join(output_dir, f"episode_{i}.hdf5")
            save_augmented_episode(
                baseline_ep, out_path, i,
                episode_paths[idx], "frozen_frame_0",
            )
            print(f"  -> {out_path}")

            del episode, baseline_ep

        print(f"\nDone. {total} baseline episodes written to {output_dir}")
        return

    # -----------------------------------------------------------------------
    # Standard augmentation mode: cross-collection pairing
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    groups: Dict[str, List[Tuple[int, int]]] = {}

    for group_name, group_colls in [("horizontal", args.horizontal), ("vertical", args.vertical)]:
        if group_colls is None:
            continue

        # Count original episodes in this group
        num_original = sum(
            1 for idx, coll in collection_map.items() if coll in group_colls
        )

        # Generate all cross-collection pairs
        all_pairs = generate_cross_collection_pairs(group_colls, collection_map, episode_paths)
        print(f"{group_name.upper()} group: {len(group_colls)} collections, "
              f"{num_original} original episodes, {len(all_pairs)} cross-collection pairs available")

        # Sample according to expansion ratio
        selected = sample_pairs(all_pairs, num_original, args.expansion_ratio, rng)
        print(f"  -> sampling {len(selected)} pairs (expansion ratio {args.expansion_ratio}x)\n")
        groups[group_name] = selected

    # Dry run
    if args.dry_run:
        if args.output_dir is None:
            args.output_dir = "(not specified)"
        print_dry_run(groups, episode_paths, args.output_dir)
        return

    # Full run requires output dir
    if args.output_dir is None:
        print("Error: --output-dir is required for augmentation (or use --dry-run / --preview)")
        sys.exit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing files
    existing = glob.glob(os.path.join(output_dir, "episode_*.hdf5"))
    if existing:
        print(f"Warning: {output_dir} already contains {len(existing)} episode files. "
              "They may be overwritten.")

    # Process all pairs
    total_pairs = sum(len(p) for p in groups.values())
    augmented_idx = 0

    for group_name, pairs in groups.items():
        print(f"\n=== {group_name.upper()} group: {len(pairs)} pairs ===\n")

        for robot_idx, human_idx in pairs:
            augmented_idx_total = augmented_idx + 1
            robot_name = os.path.join(
                os.path.basename(os.path.dirname(episode_paths[robot_idx])),
                os.path.basename(episode_paths[robot_idx]),
            )
            human_name = os.path.join(
                os.path.basename(os.path.dirname(episode_paths[human_idx])),
                os.path.basename(episode_paths[human_idx]),
            )
            print(f"[{augmented_idx_total}/{total_pairs}] "
                  f"robot={robot_name} + human={human_name}")

            # Load episodes
            robot_ep = load_episode_raw(episode_paths[robot_idx])
            human_ep = load_episode_raw(episode_paths[human_idx])

            T_r = robot_ep["action/pose"].shape[0]
            T_h = human_ep["action/pose"].shape[0]
            print(f"  T_robot={T_r}, T_human={T_h} -> output T={T_r}")

            # Create augmented episode
            augmented = create_augmented_episode(
                robot_ep, human_ep,
                args.front_crop_x, args.head_crop_x,
                args.blend_width,
            )

            # Save
            out_path = os.path.join(output_dir, f"episode_{augmented_idx}.hdf5")
            save_augmented_episode(
                augmented, out_path, augmented_idx,
                episode_paths[robot_idx], episode_paths[human_idx],
            )
            print(f"  -> {out_path}")

            augmented_idx += 1

            # Free memory
            del robot_ep, human_ep, augmented

    print(f"\nDone. {augmented_idx} augmented episodes written to {output_dir}")


if __name__ == "__main__":
    main()
