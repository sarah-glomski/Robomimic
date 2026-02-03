#!/usr/bin/env python3
"""
Universal HDF5 Episode Visualiser
=================================
• Detects images saved either channel-last (H,W,C) or channel-first (C,H,W)
  and their time-series variants (N, …).
• Supports colour (RGB / RGBA), grayscale and depth (uint16 or float32 metres).
• Keyboard controls:
      n / →   next frame
      p / ←   previous frame
      q / Esc quit current episode
• Requires: h5py, numpy, opencv-python
"""

import os, glob, math, sys
import cv2
import h5py
import numpy as np

# ─── Display layout ────────────────────────────────────────────────────────────
WIN_W, WIN_H = 640, 480      # size of each OpenCV window

# ─── Dataset helpers ───────────────────────────────────────────────────────────
def looks_like_image(shape):
    """
    Return True if the ndarray shape *could* be an image or stack of images in
    either channel-last or channel-first order.
    Accepted layouts:
        (H,W)                                – single gray/depth
        (H,W,C)  where C∈{1,3,4}             – single image, channel-last
        (C,H,W)  where C∈{1,3,4}             – single image, channel-first
        (N,H,W)                              – sequence gray/depth
        (N,H,W,C) where C∈{1,3,4}            – sequence, channel-last
        (N,C,H,W) where C∈{1,3,4}            – sequence, channel-first
    """
    if len(shape) == 2:
        return True
    if len(shape) == 3:
        return shape[2] in (1, 3, 4) or shape[0] in (1, 3, 4)
    if len(shape) == 4:
        return (shape[3] in (1,3,4)) or (shape[1] in (1,3,4))
    return False


def to_hwc(img):
    """Convert an image in any accepted layout to H×W×C (C∈{1,3})."""
    if img.ndim == 2:                  # gray / depth
        return img[..., None]          # add dummy channel
    if img.ndim == 3:
        if img.shape[0] in (1,3,4):    # CHW
            return img.transpose(1,2,0)
        return img                     # already HWC
    raise ValueError(f"Unsupported ndim {img.ndim}")


def depth_viz(depth):
    """
    Convert depth (H×W×1, uint16 mm or float32 m) to a colour map for display.
    Invalid / zero pixels are shown black.
    """
    depth = depth.squeeze()
    if depth.dtype == np.uint16:
        valid = depth > 0
        depth_m = depth.astype(np.float32) / 1000.0
    else:
        valid = np.isfinite(depth) & (depth > 0)
        depth_m = depth

    if not np.any(valid):
        return np.zeros((WIN_H, WIN_W, 3), np.uint8)

    d_min, d_max = depth_m[valid].min(), depth_m[valid].max()
    if abs(d_max - d_min) < 1e-6:
        d_max = d_min + 1e-3
    norm = np.zeros_like(depth_m, np.float32)
    norm[valid] = (depth_m[valid] - d_min) / (d_max - d_min) * 255
    colour = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    colour[~valid] = 0
    return cv2.resize(colour, (WIN_W, WIN_H))


def visualise(img):
    """Return a BGR image suitable for cv2.imshow()."""
    img = to_hwc(img)
    H, W, C = img.shape

    # depth?
    if C == 1 and (img.dtype == np.uint16 or np.issubdtype(img.dtype, np.floating)):
        return depth_viz(img)

    # grayscale
    if C == 1:
        view = img.astype(np.float32)
        if view.max() <= 1.0:       # normalise [0,1]→[0,255]
            view *= 255
        view = cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return cv2.resize(view, (WIN_W, WIN_H))

    # colour (assume RGB / RGBA)
    if C == 4:                      # drop alpha
        img = img[..., :3]
    if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    view = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(view, (WIN_W, WIN_H))


# ─── Main visualiser logic ─────────────────────────────────────────────────────
def gather_datasets(h5):
    """Return a dict {name: (ds, sequence_length)} for every image-like dataset."""
    out = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and looks_like_image(obj.shape):
            if len(obj.shape) == 4:          # time series
                length = obj.shape[0]
            elif len(obj.shape) == 3 and obj.shape[0] not in (1,3,4):
                length = obj.shape[0]        # gray/depth sequence (N,H,W)
            else:
                length = 1
            out[name] = (obj, length)
    h5.visititems(visitor)
    return out


def show_episode(path):
    print(f"\n=== {os.path.basename(path)} ===")
    with h5py.File(path, "r") as h5:
        datasets = gather_datasets(h5)
        if not datasets:
            print("No image-like datasets found.")
            return

        max_len = max(L for _, L in datasets.values())
        print(f"Found {len(datasets)} datasets, longest length {max_len} frames.")
        positions = grid_positions(len(datasets))

        frame = 0
        while True:
            for idx, (name, (ds, length)) in enumerate(datasets.items()):
                if frame >= length:
                    continue
                # retrieve frame, handling layout
                if len(ds.shape) == 4 and ds.shape[0] == length:      # N,H,W,C or N,C,H,W
                    img = ds[frame]
                elif len(ds.shape) == 3 and ds.shape[0] == length:    # N,H,W gray/depth
                    img = ds[frame]
                else:
                    img = ds[()]                                      # single image
                view = visualise(img)
                win = f"{name}"
                cv2.imshow(win, view)
                cv2.moveWindow(win, *positions[idx])

            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27):        # q or Esc → quit episode
                break
            if key in (ord('n'), ord(' '), 83, 0x27):   # n, space, → arrow
                frame = min(frame + 1, max_len - 1)
            elif key in (ord('p'), 81, 0x25):           # p, ← arrow
                frame = max(frame - 1, 0)

        cv2.destroyAllWindows()


def grid_positions(n):
    """Return {idx: (x,y)} so windows tile without overlap."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    pos = {}
    for i in range(n):
        r, c = divmod(i, cols)
        pos[i] = (c * WIN_W, r * WIN_H)
    return pos


def main(root="demo_data"):
    files = sorted(glob.glob(os.path.join(root, "*.hdf5")))
    if not files:
        print(f"No HDF5 files in {root}")
        sys.exit(0)

    for f in files:
        show_episode(f)


if __name__ == "__main__":
    main()
