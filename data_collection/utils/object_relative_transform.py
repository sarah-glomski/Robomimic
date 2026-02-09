#!/usr/bin/env python3
"""
Object-Relative Frame Transform

Transforms hand positions from absolute robot-frame coordinates
to object-relative coordinates, then maps them to a robot object frame.

Same-orientation formula:
    robot_target = robot_obj_pos + (hand_pos - human_obj_pos)

Generalized (with rotation):
    hand_rel = R_human_obj_inv @ (hand_pos - human_obj_pos)
    robot_target = robot_obj_pos + R_robot_obj @ hand_rel
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ObjectFrame:
    """Defines a reference frame for an object on the table."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 rotation matrix

    @classmethod
    def from_position(cls, x: float, y: float, z: float) -> 'ObjectFrame':
        """Create a frame with position only (identity rotation)."""
        return cls(position=np.array([x, y, z]))

    @classmethod
    def from_position_euler(cls, x: float, y: float, z: float,
                            roll: float = 0.0, pitch: float = 0.0,
                            yaw: float = 0.0) -> 'ObjectFrame':
        """Create a frame with position and euler angle rotation (radians)."""
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        return cls(position=np.array([x, y, z]), rotation=rot)


class ObjectRelativeTransform:
    """
    Computes robot target position using object-relative mapping.

    The hand position is expressed relative to a human_object_frame,
    then that relative offset is applied to a robot_object_frame.
    """

    def __init__(
        self,
        human_object_frame: ObjectFrame,
        robot_object_frame: ObjectFrame,
    ):
        self.human_object_frame = human_object_frame
        self.robot_object_frame = robot_object_frame

    def transform(self, hand_position: np.ndarray) -> np.ndarray:
        """
        Transform absolute hand position to robot target using object-relative mapping.

        Args:
            hand_position: [x, y, z] hand position in robot base frame (meters)

        Returns:
            [x, y, z] robot target position in robot base frame (meters)
        """
        h = self.human_object_frame
        r = self.robot_object_frame

        # Compute hand position relative to human object frame
        hand_relative = np.linalg.inv(h.rotation) @ (hand_position - h.position)

        # Apply relative offset to robot object frame
        robot_target = r.position + r.rotation @ hand_relative

        return robot_target
