#!/usr/bin/env python3
"""
Hand to Action Transformation Utilities

Transforms MediaPipe hand landmarks into robot actions:
- Palm center position extraction
- Pinch gesture for gripper control
- Coordinate frame transformation (MediaPipe -> Robot)
- Low-pass filtering for smoothness
"""

import numpy as np
from scipy.spatial.transform import Rotation

# MediaPipe Hand Landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


class LowPassFilter:
    """
    Simple exponential low-pass filter for smoothing noisy signals.

    Args:
        alpha: Smoothing factor (0-1). Lower = smoother but more lag.
    """
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value = None

    def filter(self, new_value: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to new value."""
        if self.value is None:
            self.value = new_value.copy()
        else:
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value.copy()

    def reset(self):
        """Reset filter state."""
        self.value = None


class HandToActionTransformer:
    """
    Transform MediaPipe hand landmarks to robot actions.

    Coordinate Frames:
    - MediaPipe: x-right, y-down, z-away from camera (normalized 0-1 for x,y)
    - Robot: x-forward, y-left, z-up (in meters)

    Args:
        position_scale: Scale factor for hand position to robot workspace
        position_offset: Offset to center hand workspace in robot frame [x, y, z]
        pinch_threshold_close: Distance below which gripper closes
        pinch_threshold_open: Distance above which gripper opens
        filter_alpha: Low-pass filter smoothing factor
    """
    def __init__(
        self,
        position_scale: float = 0.3,
        position_offset: np.ndarray = None,
        pinch_threshold_close: float = 0.05,
        pinch_threshold_open: float = 0.12,
        filter_alpha: float = 0.3
    ):
        self.position_scale = position_scale
        self.position_offset = position_offset if position_offset is not None else np.array([0.3, 0.0, 0.2])
        self.pinch_threshold_close = pinch_threshold_close
        self.pinch_threshold_open = pinch_threshold_open

        # Smoothing filters
        self.position_filter = LowPassFilter(alpha=filter_alpha)
        self.gripper_filter = LowPassFilter(alpha=0.2)

        # Track previous gripper state for hysteresis
        self._prev_gripper = 0.0

    def landmarks_to_action(self, landmarks) -> dict:
        """
        Convert MediaPipe landmarks to robot action.

        Args:
            landmarks: MediaPipe hand landmarks (21 NormalizedLandmark objects)

        Returns:
            dict with keys:
                - position: np.ndarray (3,) - robot target position in meters
                - gripper: float - gripper command 0 (open) to 1 (closed)
                - raw_hand_position: np.ndarray (3,) - unfiltered hand position
        """
        # Extract palm center position in MediaPipe frame
        raw_hand_position = self.get_palm_position(landmarks)

        # Transform to robot frame
        robot_position = self.transform_to_robot_frame(raw_hand_position)

        # Apply smoothing
        robot_position = self.position_filter.filter(robot_position)

        # Get gripper from pinch gesture
        gripper = self.get_gripper_from_pinch(landmarks)

        return {
            'position': robot_position,
            'gripper': gripper,
            'raw_hand_position': raw_hand_position
        }

    def get_palm_position(self, landmarks) -> np.ndarray:
        """
        Get palm center from hand landmarks.
        Uses average of wrist and 4 MCP (knuckle) joints.

        Returns:
            np.ndarray (3,): Palm center in MediaPipe normalized coordinates
        """
        palm_indices = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

        positions = []
        for idx in palm_indices:
            lm = landmarks[idx]
            positions.append([lm.x, lm.y, lm.z])

        positions = np.array(positions)
        return positions.mean(axis=0)

    def transform_to_robot_frame(self, hand_position: np.ndarray) -> np.ndarray:
        """
        Transform hand position from MediaPipe frame to robot frame.

        MediaPipe frame: x-right, y-down, z-away (normalized 0-1 for x,y)
        Robot frame: x-forward, y-left, z-up (meters)

        Args:
            hand_position: Position in MediaPipe normalized coordinates

        Returns:
            Position in robot frame (meters)
        """
        # Center hand position around 0.5 (middle of frame)
        centered_x = hand_position[0] - 0.5  # -0.5 to 0.5
        centered_y = hand_position[1] - 0.5  # -0.5 to 0.5
        depth_z = hand_position[2]  # Relative depth from MediaPipe

        # Transform coordinates:
        # MediaPipe x (right) -> Robot y (left, so negate)
        # MediaPipe y (down) -> Robot z (up, so negate)
        # MediaPipe z (away) -> Robot x (forward, so negate for intuitive control)
        robot_position = np.array([
            -depth_z * self.position_scale * 2.0,     # Forward/back from depth
            -centered_x * self.position_scale * 2.0,  # Left/right
            -centered_y * self.position_scale * 2.0,  # Up/down
        ])

        # Add offset to center in robot workspace
        robot_position = robot_position + self.position_offset

        return robot_position

    def get_gripper_from_pinch(self, landmarks) -> float:
        """
        Get gripper command from pinch gesture (thumb to index finger distance).
        Uses hysteresis to prevent oscillation.

        Returns:
            float: Gripper command 0.0 (open) to 1.0 (closed)
        """
        # Get thumb and index tip positions
        thumb_tip = np.array([
            landmarks[THUMB_TIP].x,
            landmarks[THUMB_TIP].y,
            landmarks[THUMB_TIP].z
        ])
        index_tip = np.array([
            landmarks[INDEX_TIP].x,
            landmarks[INDEX_TIP].y,
            landmarks[INDEX_TIP].z
        ])

        # Calculate distance
        distance = np.linalg.norm(thumb_tip - index_tip)

        # Apply hysteresis for stability
        if distance < self.pinch_threshold_close:
            gripper = 1.0  # Closed
        elif distance > self.pinch_threshold_open:
            gripper = 0.0  # Open
        else:
            # Linear interpolation in the middle zone
            range_size = self.pinch_threshold_open - self.pinch_threshold_close
            gripper = 1.0 - (distance - self.pinch_threshold_close) / range_size

        # Apply smoothing
        gripper_arr = np.array([gripper])
        gripper = self.gripper_filter.filter(gripper_arr)[0]

        self._prev_gripper = gripper
        return float(gripper)

    def reset(self):
        """Reset all filters."""
        self.position_filter.reset()
        self.gripper_filter.reset()
        self._prev_gripper = 0.0


def get_hand_tracking_status(landmarks) -> bool:
    """
    Check if hand tracking is valid.

    Args:
        landmarks: MediaPipe hand landmarks or None

    Returns:
        bool: True if hand is being tracked
    """
    return landmarks is not None and len(landmarks) == 21
