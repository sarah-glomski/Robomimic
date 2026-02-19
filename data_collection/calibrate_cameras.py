#!/usr/bin/env python3
"""
Camera-to-Robot Calibration Script

Calibrates the extrinsic transforms (rotation + translation) from each fixed
camera's frame to the robot base frame using a checkerboard placed flat on the
table in the robot's workspace.

Algorithm:
  1. User puts xArm in teach mode and touches two checkerboard corners with
     the gripper tip to establish the board pose in robot frame.
  2. Script detects checkerboard corners in each camera image.
  3. cv2.solvePnP computes each camera's extrinsics relative to the robot frame.
  4. Results saved to camera_calibration.json for use by other nodes.

Usage:
    # Cameras must be running (launch_data_collection.py or standalone realsense nodes)
    python3 calibrate_cameras.py --rows 7 --cols 9 --square-size 0.02
"""

import os
import sys
import json
import time
import argparse
import threading
from datetime import datetime

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from xarm.wrapper import XArmAPI

# Camera configuration (must match launch_data_collection.py)
CAMERAS = {
    "head": {
        "serial": "845112071112",
        "color_topic": "/rs_head/rs_head/color/image_raw",
        "info_topic": "/rs_head/rs_head/color/camera_info",
    },
    "front": {
        "serial": "244222071219",
        "color_topic": "/rs_front/rs_front/color/image_raw",
        "info_topic": "/rs_front/rs_front/color/camera_info",
    },
    "wrist": {
        "serial": "317222072257",
        "color_topic": "/rs_wrist/rs_wrist/color/image_raw",
        "info_topic": "/rs_wrist/rs_wrist/color/camera_info",
    },
}

XARM_IP = "192.168.1.219"
TCP_OFFSET_Z_MM = 172.0  # Actual flange-to-gripper-tip distance


class CalibrationNode(Node):
    """ROS2 node that subscribes to camera image and camera_info topics."""

    def __init__(self):
        super().__init__('camera_calibration')
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self._latest_frames = {}   # cam_name -> np.ndarray (BGR)
        self._intrinsics = {}      # cam_name -> dict

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        for cam_name, cfg in CAMERAS.items():
            self.create_subscription(
                CameraInfo, cfg["info_topic"],
                lambda msg, n=cam_name: self._camera_info_cb(msg, n),
                sensor_qos
            )
            self.create_subscription(
                Image, cfg["color_topic"],
                lambda msg, n=cam_name: self._image_cb(msg, n),
                sensor_qos
            )

    def _camera_info_cb(self, msg: CameraInfo, cam_name: str):
        with self._lock:
            if cam_name in self._intrinsics:
                return
            self._intrinsics[cam_name] = {
                "fx": msg.k[0], "fy": msg.k[4],
                "cx": msg.k[2], "cy": msg.k[5],
                "dist_coeffs": np.array(msg.d, dtype=np.float64),
            }
        self.get_logger().info(
            f"  {cam_name}: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} "
            f"cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}"
        )

    def _image_cb(self, msg: Image, cam_name: str):
        with self._lock:
            self._latest_frames[cam_name] = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )

    def get_frame(self, cam_name: str):
        with self._lock:
            frame = self._latest_frames.get(cam_name)
            return frame.copy() if frame is not None else None

    def get_intrinsics(self, cam_name: str):
        with self._lock:
            return self._intrinsics.get(cam_name)


def record_touch_point(arm: XArmAPI, label: str) -> np.ndarray:
    """Prompt user to position robot, press Enter, and record TCP position."""
    input(f"\n>>> Move gripper tip to {label}, then press ENTER...")
    code, pose = arm.get_position(is_radian=True)
    if code != 0:
        raise RuntimeError(f"Failed to read xArm position, error code: {code}")
    pos_m = np.array(pose[0:3]) / 1000.0
    print(f"    Recorded: [{pos_m[0]:.4f}, {pos_m[1]:.4f}, {pos_m[2]:.4f}] m")
    return pos_m


def compute_board_corners_robot_frame(
    origin: np.ndarray,
    row_end: np.ndarray,
    rows: int,
    cols: int,
    square_size: float,
) -> np.ndarray:
    """
    Compute all inner corner positions of the checkerboard in robot base frame.

    The board is flat on the table. Two touched corners define the row direction.
    Column direction = cross([0,0,1], row_dir) since the board is horizontal.

    Args:
        origin:      XYZ of first inner corner (row 0, col 0) in meters.
        row_end:     XYZ of last corner in first row (row 0, col cols-1).
        rows:        Number of inner corner rows.
        cols:        Number of inner corner columns.
        square_size: Square side length in meters.

    Returns:
        np.ndarray shape (rows*cols, 3): Corners in row-major order,
        matching cv2.findChessboardCorners output.
    """
    row_vec = row_end - origin
    row_dir = row_vec / np.linalg.norm(row_vec)

    z_up = np.array([0.0, 0.0, 1.0])
    col_dir = np.cross(z_up, row_dir)
    col_dir = col_dir / np.linalg.norm(col_dir)

    corners = np.zeros((rows * cols, 3), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            corners[i * cols + j] = (
                origin
                + j * square_size * row_dir
                + i * square_size * col_dir
            )
    return corners


def detect_checkerboard(frame: np.ndarray, rows: int, cols: int):
    """
    Detect checkerboard corners in a BGR image.

    Returns:
        corners_2d: np.ndarray (N, 1, 2) or None if not found.
        display_frame: BGR image with corners drawn.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)

    display = frame.copy()
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        cv2.drawChessboardCorners(display, (cols, rows), corners, found)
    else:
        # Show image dimensions to help debug
        h, w = frame.shape[:2]
        cv2.putText(display, f"NOT DETECTED ({w}x{h})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return (corners if found else None), display


def solve_camera_extrinsics(
    corners_3d_robot: np.ndarray,
    corners_2d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict:
    """
    Compute camera-to-robot extrinsics via cv2.solvePnP.

    solvePnP gives (rvec, tvec) such that:
        point_cam = R_robot2cam @ point_robot + t_robot2cam

    We invert to get camera-to-robot (what mediapipe_hand_tracker uses):
        R_cam2robot = R_robot2cam.T
        t_cam2robot = -R_robot2cam.T @ t_robot2cam
    """
    obj_pts = corners_3d_robot.astype(np.float64)
    img_pts = corners_2d.reshape(-1, 2).astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("solvePnP failed")

    R_robot2cam, _ = cv2.Rodrigues(rvec)
    t_robot2cam = tvec.flatten()

    R_cam2robot = R_robot2cam.T
    t_cam2robot = -R_robot2cam.T @ t_robot2cam

    # Reprojection error
    projected, _ = cv2.projectPoints(
        obj_pts, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - img_pts, axis=1)
    rms_error = np.sqrt(np.mean(errors ** 2))

    return {
        "rotation_cam_to_robot": R_cam2robot,
        "translation_cam_to_robot": t_cam2robot,
        "camera_position_robot_frame": t_cam2robot.copy(),
        "reprojection_error_px": float(rms_error),
    }


def rotation_matrix_to_quaternion_wxyz(R_mat: np.ndarray) -> list:
    """Convert 3x3 rotation matrix to [w, x, y, z] quaternion."""
    quat_xyzw = Rotation.from_matrix(R_mat).as_quat()  # scipy: [x, y, z, w]
    return [float(quat_xyzw[3]), float(quat_xyzw[0]),
            float(quat_xyzw[1]), float(quat_xyzw[2])]


def build_output(args, origin, row_end, cam_results, cam_intrinsics):
    """Assemble the calibration JSON structure."""
    output = {
        "calibration_date": datetime.now().isoformat(),
        "checkerboard": {
            "rows": args.rows,
            "cols": args.cols,
            "square_size_m": args.square_size,
            "origin_robot_frame": origin.tolist(),
            "row_end_robot_frame": row_end.tolist(),
        },
    }

    for cam_name in ["head", "front"]:
        if cam_name not in cam_results:
            continue
        r = cam_results[cam_name]
        intr = cam_intrinsics.get(cam_name, {})
        quat = rotation_matrix_to_quaternion_wxyz(r["rotation_cam_to_robot"])
        output[f"{cam_name}_camera"] = {
            "serial": CAMERAS[cam_name]["serial"],
            "rotation_cam_to_robot": r["rotation_cam_to_robot"].tolist(),
            "translation_cam_to_robot": r["translation_cam_to_robot"].tolist(),
            "camera_position_robot_frame": r["camera_position_robot_frame"].tolist(),
            "camera_quaternion_wxyz": quat,
            "intrinsics": {
                "fx": intr.get("fx", 0), "fy": intr.get("fy", 0),
                "cx": intr.get("cx", 0), "cy": intr.get("cy", 0),
            },
            "reprojection_error_px": r["reprojection_error_px"],
        }

    if "wrist" in cam_intrinsics:
        intr = cam_intrinsics["wrist"]
        output["wrist_camera"] = {
            "serial": CAMERAS["wrist"]["serial"],
            "intrinsics": {
                "fx": intr["fx"], "fy": intr["fy"],
                "cx": intr["cx"], "cy": intr["cy"],
            },
            "flange_to_camera_offset_mm": [0, 0, 0, 0, 0, 0],
            "note": "Update flange_to_camera_offset_mm with measured mount dimensions",
        }

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Camera-to-robot calibration using a checkerboard"
    )
    parser.add_argument("--rows", type=int, default=7,
                        help="Inner corner rows (default: 7)")
    parser.add_argument("--cols", type=int, default=9,
                        help="Inner corner columns (default: 9)")
    parser.add_argument("--square-size", type=float, default=0.02,
                        help="Square side length in meters (default: 0.02)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: camera_calibration.json)")
    parser.add_argument("--xarm-ip", type=str, default=XARM_IP,
                        help=f"xArm IP address (default: {XARM_IP})")
    parser.add_argument("--touch-points", type=str, nargs=2, metavar=("ORIGIN", "ROW_END"),
                        help="Reuse previously recorded touch points (x,y,z x,y,z in meters). "
                             "Example: --touch-points 0.4737,-0.0147,0.0761 0.4665,-0.1668,0.0767")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, "camera_calibration.json")

    print("=" * 60)
    print("Camera-to-Robot Calibration")
    print("=" * 60)
    print(f"Checkerboard: {args.rows}x{args.cols} inner corners, "
          f"{args.square_size*1000:.0f}mm squares")
    print()

    # --- Initialize ROS2 in background thread ---
    rclpy.init()
    node = CalibrationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    # --- Wait for camera intrinsics ---
    print("Waiting for camera intrinsics...")
    all_cameras = ["head", "front", "wrist"]
    cameras_to_calibrate = ["head", "front"]
    timeout = 15.0
    t0 = time.time()
    while time.time() - t0 < timeout:
        received = [c for c in all_cameras if node.get_intrinsics(c) is not None]
        if len(received) >= len(all_cameras):
            break
        time.sleep(0.5)

    for cam in all_cameras:
        if node.get_intrinsics(cam) is None:
            print(f"  WARNING: No camera_info for {cam} "
                  f"(topic: {CAMERAS[cam]['info_topic']})")

    cam_intrinsics = {}
    for cam in all_cameras:
        intr = node.get_intrinsics(cam)
        if intr is not None:
            cam_intrinsics[cam] = intr

    # --- Get touch points (from CLI or interactive) ---
    arm = None
    if args.touch_points is not None:
        # Reuse previously recorded touch points
        origin = np.array([float(v) for v in args.touch_points[0].split(",")], dtype=np.float64)
        row_end = np.array([float(v) for v in args.touch_points[1].split(",")], dtype=np.float64)
        print(f"\nUsing provided touch points:")
        print(f"  Origin:  [{origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}] m")
        print(f"  Row end: [{row_end[0]:.4f}, {row_end[1]:.4f}, {row_end[2]:.4f}] m")
    else:
        # Interactive: connect to xArm, enter teach mode
        print(f"\nConnecting to xArm at {args.xarm_ip}...")
        arm = XArmAPI(args.xarm_ip)
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_tcp_offset([0, 0, TCP_OFFSET_Z_MM, 0, 0, 0])
        time.sleep(0.1)
        arm.set_mode(2)  # Teach mode
        arm.set_state(0)
        time.sleep(0.5)
        print("xArm in TEACH MODE — you can freely move the arm by hand.")

        print("\nPlace the checkerboard FLAT on the table in view of the cameras.")
        print("You will touch TWO corners with the gripper tip:")
        print(f"  1) ORIGIN corner — first inner corner (row 0, col 0)")
        print(f"  2) ROW END corner — same row, opposite end (row 0, col {args.cols - 1})")

        origin = record_touch_point(arm, "ORIGIN corner (row 0, col 0)")
        row_end = record_touch_point(
            arm, f"ROW END corner (row 0, col {args.cols - 1})")

        # Switch back to position mode
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)

    # Validate touch distance
    touched_dist = np.linalg.norm(row_end - origin)
    expected_dist = (args.cols - 1) * args.square_size
    print(f"\n    Touched distance:  {touched_dist*1000:.1f} mm")
    print(f"    Expected distance: {expected_dist*1000:.1f} mm")
    diff_mm = abs(touched_dist - expected_dist) * 1000
    print(f"    Difference:        {diff_mm:.1f} mm")
    if diff_mm > 10:
        print("    WARNING: >10mm deviation — double-check corner positions.")

    print(f"\n    To reuse these touch points later, run with:")
    print(f"    --touch-points {origin[0]:.4f},{origin[1]:.4f},{origin[2]:.4f} "
          f"{row_end[0]:.4f},{row_end[1]:.4f},{row_end[2]:.4f}")

    # --- Compute all corners in robot frame ---
    corners_3d = compute_board_corners_robot_frame(
        origin, row_end, args.rows, args.cols, args.square_size
    )
    print(f"\nComputed {len(corners_3d)} corner positions in robot frame.")

    # Build all 4 possible corner orderings to resolve solvePnP planar ambiguity.
    # cross(z_up, row_dir) picks one column direction but the actual OpenCV corner
    # ordering depends on image orientation, so we must try row/col flips too.
    grid = corners_3d.reshape(args.rows, args.cols, 3)
    corners_3d_variants = [
        # ("original",  corners_3d),
        ("col_flip",  grid[:, ::-1, :].reshape(-1, 3).copy()),
        # ("row_flip",  grid[::-1, :, :].reshape(-1, 3).copy()),
        # ("both_flip", corners_3d[::-1].copy()),
    ]
    board_z = np.mean(corners_3d[:, 2])

    # --- Detect checkerboard and solve for each camera ---
    cam_results = {}
    while True:
        input("\n>>> Ensure checkerboard is visible to cameras, then press ENTER...")

        # Give cameras a moment to get a fresh frame
        time.sleep(0.3)

        for cam_name in cameras_to_calibrate:
            if cam_name in cam_results:
                continue  # Already calibrated on a previous attempt
            print(f"\n  {cam_name} camera:")
            frame = node.get_frame(cam_name)
            if frame is None:
                print(f"    No frame available — skipping.")
                continue

            intr = cam_intrinsics.get(cam_name)
            if intr is None:
                print(f"    No intrinsics available — skipping.")
                continue

            corners_2d, display = detect_checkerboard(frame, args.rows, args.cols)
            cv2.imshow(f"{cam_name} - checkerboard", display)
            cv2.waitKey(500)

            if corners_2d is None:
                print(f"    Checkerboard NOT detected.")
                continue

            print(f"    Checkerboard detected ({len(corners_2d)} corners)")

            camera_matrix = np.array([
                [intr["fx"], 0, intr["cx"]],
                [0, intr["fy"], intr["cy"]],
                [0, 0, 1],
            ], dtype=np.float64)

            # Try all 4 corner orderings to resolve planar ambiguity.
            # For a flat checkerboard, solvePnP can place the camera on the
            # wrong side of the board if the 3D-2D ordering doesn't match.
            # We pick the solution where the camera is above the board (z > board_z).
            candidates = []
            for label, pts_3d in corners_3d_variants:
                try:
                    r = solve_camera_extrinsics(
                        pts_3d, corners_2d, camera_matrix, intr["dist_coeffs"])
                    r["_label"] = label
                    candidates.append(r)
                except RuntimeError:
                    pass

            if not candidates:
                print(f"    solvePnP failed for all orderings — skipping.")
                continue

            # Print ALL candidates so user can verify against physical measurement
            valid = [c for c in candidates
                     if c["camera_position_robot_frame"][2] > board_z]
            print(f"    All solutions (camera above board):")
            for c in candidates:
                pos = c["camera_position_robot_frame"]
                above = "above" if pos[2] > board_z else "BELOW"
                print(f"      {c['_label']:>10s}: "
                      f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                      f"err={c['reprojection_error_px']:.2f}px  ({above})")

            if valid:
                result = min(valid, key=lambda c: c["reprojection_error_px"])
            else:
                result = min(candidates, key=lambda c: c["reprojection_error_px"])
                print(f"    WARNING: No solution has camera above board")

            print(f"    >>> Selected: {result['_label']}")
            del result["_label"]

            cam_results[cam_name] = result
            pos = result["camera_position_robot_frame"]
            err = result["reprojection_error_px"]
            print(f"    Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
            print(f"    Reprojection error: {err:.2f} px")

        # Check if all cameras are calibrated or offer retry
        remaining = [c for c in cameras_to_calibrate if c not in cam_results]
        if not remaining:
            break
        retry = input(f"\n  {len(remaining)} camera(s) not detected ({', '.join(remaining)}). "
                       f"Retry? [Y/n] ").strip().lower()
        if retry == 'n':
            break

    # --- Save calibration ---
    if len(cam_results) == 0:
        print("\nERROR: No cameras were calibrated. Exiting without saving.")
    else:
        output = build_output(args, origin, row_end, cam_results, cam_intrinsics)
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nCalibration saved to: {args.output}")

        # Print summary
        if "head" in cam_results:
            r = cam_results["head"]
            print("\n--- HEAD CAMERA ---")
            print(f"  Rotation matrix (cam→robot):")
            R = r["rotation_cam_to_robot"]
            for row in R:
                print(f"    [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
            t = r["translation_cam_to_robot"]
            print(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
            print(f"  Reprojection error: {r['reprojection_error_px']:.2f} px")

        if "front" in cam_results:
            r = cam_results["front"]
            print("\n--- FRONT CAMERA ---")
            print(f"  Rotation matrix (cam→robot):")
            R = r["rotation_cam_to_robot"]
            for row in R:
                print(f"    [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
            t = r["translation_cam_to_robot"]
            print(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
            print(f"  Reprojection error: {r['reprojection_error_px']:.2f} px")

    # --- Cleanup ---
    if len(cam_results) > 0:
        print("\nPress any key in the OpenCV window to close...")
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    if arm is not None:
        arm.disconnect()
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
