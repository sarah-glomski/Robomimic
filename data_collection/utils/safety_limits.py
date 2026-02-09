#!/usr/bin/env python3
"""
Safety Limits for xArm Control

Provides workspace bounds, velocity limits, and position delta limits
for safe robot operation. Used by both simulator and real controller.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class WorkspaceBounds:
    """Defines the safe workspace boundaries in meters."""
    x_min: float = 0.10   # Forward from base
    x_max: float = 0.50
    y_min: float = -0.30  # Left/right
    y_max: float = 0.30
    z_min: float = 0.05   # Height (above table)
    z_max: float = 0.40


@dataclass
class VelocityLimits:
    """Defines maximum velocities for safe operation."""
    max_linear_velocity: float = 0.15      # m/s - max cartesian velocity
    max_angular_velocity: float = 0.5       # rad/s - max rotation speed
    max_gripper_velocity: float = 1.0       # units/s - gripper open/close speed


@dataclass
class DeltaLimits:
    """Defines maximum position change per control step."""
    max_position_delta: float = 0.010       # 10mm max movement per step
    max_orientation_delta: float = 0.05     # rad max rotation per step
    max_gripper_delta: float = 0.05         # max gripper change per step


class SafetyLimits:
    """
    Safety limits manager for xArm control.

    Provides methods to:
    - Clamp positions to workspace bounds
    - Limit velocity
    - Limit position deltas between timesteps
    - Check if position is within bounds
    """

    def __init__(
        self,
        workspace: Optional[WorkspaceBounds] = None,
        velocity: Optional[VelocityLimits] = None,
        delta: Optional[DeltaLimits] = None,
        soft_start_factor: float = 1.0
    ):
        """
        Initialize safety limits.

        Args:
            workspace: Workspace boundary limits
            velocity: Velocity limits
            delta: Per-step delta limits
            soft_start_factor: Multiplier for velocity/delta limits (0.1 = 10% speed)
        """
        self.workspace = workspace or WorkspaceBounds()
        self.velocity = velocity or VelocityLimits()
        self.delta = delta or DeltaLimits()
        self.soft_start_factor = np.clip(soft_start_factor, 0.01, 1.0)

        # Track previous position for delta limiting
        self._prev_position: Optional[np.ndarray] = None
        self._prev_gripper: Optional[float] = None

    def set_soft_start_factor(self, factor: float):
        """Set soft start factor (0.1 = 10% speed, 1.0 = full speed)."""
        self.soft_start_factor = np.clip(factor, 0.01, 1.0)

    def get_effective_max_velocity(self) -> float:
        """Get current max velocity with soft start applied."""
        return self.velocity.max_linear_velocity * self.soft_start_factor

    def get_effective_max_delta(self) -> float:
        """Get current max position delta with soft start applied."""
        return self.delta.max_position_delta * self.soft_start_factor

    def clamp_position(self, position: np.ndarray) -> np.ndarray:
        """
        Clamp position to workspace bounds.

        Args:
            position: [x, y, z] position in meters

        Returns:
            Clamped position within workspace bounds
        """
        clamped = np.array([
            np.clip(position[0], self.workspace.x_min, self.workspace.x_max),
            np.clip(position[1], self.workspace.y_min, self.workspace.y_max),
            np.clip(position[2], self.workspace.z_min, self.workspace.z_max),
        ])
        return clamped

    def is_within_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within workspace bounds."""
        return (
            self.workspace.x_min <= position[0] <= self.workspace.x_max and
            self.workspace.y_min <= position[1] <= self.workspace.y_max and
            self.workspace.z_min <= position[2] <= self.workspace.z_max
        )

    def limit_velocity(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Limit movement to respect velocity constraints.

        Args:
            current_pos: Current [x, y, z] position
            target_pos: Desired [x, y, z] position
            dt: Time step in seconds

        Returns:
            New target position respecting velocity limits
        """
        max_vel = self.get_effective_max_velocity()
        max_step = max_vel * dt

        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)

        if error_norm <= max_step:
            return target_pos.copy()
        else:
            # Move at max velocity toward target
            direction = error / error_norm
            return current_pos + direction * max_step

    def limit_delta(
        self,
        target_pos: np.ndarray,
        allow_reset: bool = False
    ) -> np.ndarray:
        """
        Limit position change from previous timestep.

        Args:
            target_pos: Desired [x, y, z] position
            allow_reset: If True, allows large jumps (for reset to home)

        Returns:
            Position limited by max delta from previous position
        """
        if self._prev_position is None or allow_reset:
            self._prev_position = target_pos.copy()
            return target_pos.copy()

        max_delta = self.get_effective_max_delta()

        delta = target_pos - self._prev_position
        delta_norm = np.linalg.norm(delta)

        if delta_norm <= max_delta:
            limited_pos = target_pos.copy()
        else:
            # Limit the step size
            direction = delta / delta_norm
            limited_pos = self._prev_position + direction * max_delta

        # Ensure within workspace
        limited_pos = self.clamp_position(limited_pos)
        self._prev_position = limited_pos.copy()

        return limited_pos

    def limit_gripper_delta(
        self,
        target_gripper: float,
        allow_reset: bool = False
    ) -> float:
        """
        Limit gripper change from previous timestep.

        Args:
            target_gripper: Desired gripper value [0, 1]
            allow_reset: If True, allows instant gripper changes

        Returns:
            Gripper value limited by max delta
        """
        target_gripper = np.clip(target_gripper, 0.0, 1.0)

        if self._prev_gripper is None or allow_reset:
            self._prev_gripper = target_gripper
            return target_gripper

        max_delta = self.delta.max_gripper_delta * self.soft_start_factor

        delta = target_gripper - self._prev_gripper

        if abs(delta) <= max_delta:
            limited_gripper = target_gripper
        else:
            limited_gripper = self._prev_gripper + np.sign(delta) * max_delta

        self._prev_gripper = limited_gripper
        return limited_gripper

    def reset_tracking(self):
        """Reset position tracking (call when resetting robot)."""
        self._prev_position = None
        self._prev_gripper = None

    def get_bounds_str(self) -> str:
        """Get human-readable workspace bounds string."""
        return (
            f"X: [{self.workspace.x_min}, {self.workspace.x_max}] m, "
            f"Y: [{self.workspace.y_min}, {self.workspace.y_max}] m, "
            f"Z: [{self.workspace.z_min}, {self.workspace.z_max}] m"
        )

    def get_bounds_array(self) -> np.ndarray:
        """Get workspace bounds as array [[x_min, x_max], [y_min, y_max], [z_min, z_max]]."""
        return np.array([
            [self.workspace.x_min, self.workspace.x_max],
            [self.workspace.y_min, self.workspace.y_max],
            [self.workspace.z_min, self.workspace.z_max],
        ])


def create_conservative_limits() -> SafetyLimits:
    """Create conservative safety limits for initial testing."""
    return SafetyLimits(
        workspace=WorkspaceBounds(
            x_min=0.15, x_max=0.45,  # Smaller workspace
            y_min=-0.25, y_max=0.25,
            z_min=0.08, z_max=0.35,
        ),
        velocity=VelocityLimits(
            max_linear_velocity=0.08,  # Slower
            max_angular_velocity=0.3,
            max_gripper_velocity=0.5,
        ),
        delta=DeltaLimits(
            max_position_delta=0.005,  # 5mm max per step
            max_orientation_delta=0.03,
            max_gripper_delta=0.03,
        ),
        soft_start_factor=0.5  # Start at 50% speed
    )


def create_standard_limits() -> SafetyLimits:
    """Create standard safety limits for normal operation."""
    return SafetyLimits()  # Uses defaults
