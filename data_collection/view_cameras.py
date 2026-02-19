#!/usr/bin/env python3
"""Simple script to view live feeds from all 3 RealSense cameras."""

import pyrealsense2 as rs
import numpy as np
import cv2

CAMERAS = {
    "rs_front": "244222071219",
    "rs_wrist": "317222072257",
    "rs_head":  "845112071112",
}

WIDTH, HEIGHT, FPS = 640, 360, 30


def main():
    pipelines = {}

    for name, serial in CAMERAS.items():
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

        pipe = rs.pipeline()
        try:
            pipe.start(cfg)
            pipelines[name] = pipe
            print(f"Started {name} (serial {serial})")
        except RuntimeError as e:
            print(f"Failed to start {name} (serial {serial}): {e}")

    if not pipelines:
        print("No cameras connected.")
        return

    print("Press 'q' to quit.")

    try:
        while True:
            frames = {}
            for name, pipe in pipelines.items():
                frameset = pipe.wait_for_frames(timeout_ms=1000)
                color_frame = frameset.get_color_frame()
                if color_frame:
                    frames[name] = np.asanyarray(color_frame.get_data())

            for name, image in frames.items():
                cv2.imshow(name, image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for pipe in pipelines.values():
            pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
