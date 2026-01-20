import numpy as np

class PixelToXArmYZ:
    def __init__(
        self,
        image_width,
        image_height,
        v_max=0.15,
        deadzone=0.05,
        smoothing=0.2,
    ):
        self.cx = image_width / 2
        self.cy = image_height / 2
        self.v_max = v_max
        self.deadzone = deadzone
        self.alpha = smoothing

        self.vy_prev = 0.0
        self.vz_prev = 0.0

    def _deadzone(self, v):
        return 0.0 if abs(v) < self.deadzone else v

    def step(self, x_px, y_px):
        # Normalize pixel displacement
        dx = (x_px - self.cx) / self.cx
        dy = (y_px - self.cy) / self.cy

        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)

        dx = self._deadzone(dx)
        dy = self._deadzone(dy)

        # Map to robot Y-Z velocities
        vy =  self.v_max * dx
        vz = -self.v_max * dy

        # Exponential smoothing (EMA)
        vy = self.alpha * vy + (1 - self.alpha) * self.vy_prev
        vz = self.alpha * vz + (1 - self.alpha) * self.vz_prev

        self.vy_prev = vy
        self.vz_prev = vz

        return vy, vz
