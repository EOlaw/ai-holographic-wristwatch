"""Micro-mirror beam steering for the AI Holographic Wristwatch projector.

Controls MEMS (Micro-Electro-Mechanical Systems) mirrors to steer laser beams
and adjust depth-of-field for sharp holographic projection at varying distances.
"""

import threading
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class FocusMode(Enum):
    AUTO = "auto"
    MANUAL = "manual"
    TRACKING = "tracking"


@dataclass
class BeamState:
    mode: FocusMode
    mirror_angle_x: float
    mirror_angle_y: float
    focal_distance_mm: float
    depth_of_field_mm: float
    is_locked: bool
    timestamp: float = field(default_factory=time.time)


class BeamFocusController:
    """MEMS mirror angle control for holographic beam steering and depth-of-field.

    Supports automatic focus tracking, manual override, and object-following
    tracking mode. Mirror angles are specified in degrees from optical axis.
    """

    ANGLE_RANGE_DEG = (-15.0, 15.0)
    FOCAL_RANGE_MM = (50.0, 500.0)
    DOF_DEFAULT_MM = 20.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._mode = FocusMode.AUTO
        self._angle_x = 0.0
        self._angle_y = 0.0
        self._focal_distance_mm = 200.0
        self._dof_mm = self.DOF_DEFAULT_MM
        self._locked = False
        logger.info("BeamFocusController initialized")

    def set_mode(self, mode: FocusMode) -> None:
        """Switch focus mode."""
        with self._lock:
            self._mode = mode
            logger.info("Beam focus mode -> %s", mode.value)

    def set_angle(self, angle_x: float, angle_y: float) -> bool:
        """Set MEMS mirror angles manually (degrees). Returns False if out of range."""
        lo, hi = self.ANGLE_RANGE_DEG
        if not (lo <= angle_x <= hi and lo <= angle_y <= hi):
            logger.warning("Angle (%.2f, %.2f) out of range [%.1f, %.1f]",
                           angle_x, angle_y, lo, hi)
            return False
        with self._lock:
            self._angle_x = angle_x
            self._angle_y = angle_y
            self._write_hardware_angles(angle_x, angle_y)
            logger.debug("Mirror angles set to (%.3f°, %.3f°)", angle_x, angle_y)
            return True

    def set_focal_distance(self, distance_mm: float) -> bool:
        """Set focal distance in mm. Returns False if outside supported range."""
        lo, hi = self.FOCAL_RANGE_MM
        if not (lo <= distance_mm <= hi):
            logger.warning("Focal distance %.1f mm out of range", distance_mm)
            return False
        with self._lock:
            self._focal_distance_mm = distance_mm
            self._dof_mm = self._calculate_dof(distance_mm)
            logger.debug("Focal distance set to %.1f mm, DoF %.1f mm",
                         distance_mm, self._dof_mm)
            return True

    def auto_focus(self, target_distance_mm: float) -> BeamState:
        """Run autofocus routine toward target distance and return resulting state."""
        with self._lock:
            if self._mode == FocusMode.AUTO:
                self.set_focal_distance(target_distance_mm)
                angle_x, angle_y = self._simulate_tracking_correction(target_distance_mm)
                self.set_angle(angle_x, angle_y)
                self._locked = True
            return self.get_state()

    def get_state(self) -> BeamState:
        """Return current beam and mirror state."""
        with self._lock:
            return BeamState(
                mode=self._mode,
                mirror_angle_x=self._angle_x,
                mirror_angle_y=self._angle_y,
                focal_distance_mm=self._focal_distance_mm,
                depth_of_field_mm=self._dof_mm,
                is_locked=self._locked,
            )

    def _calculate_dof(self, focal_mm: float) -> float:
        """Compute depth-of-field based on focal distance (simplified thin-lens model)."""
        f_number = 2.8
        circle_of_confusion_mm = 0.03
        dof = 2.0 * f_number * circle_of_confusion_mm * (focal_mm ** 2) / (focal_mm ** 2)
        return max(5.0, min(dof * (focal_mm / 100.0), 100.0))

    def _write_hardware_angles(self, x: float, y: float) -> None:
        """Simulate writing MEMS mirror angles to hardware DAC."""
        pass  # Hardware I2C/SPI write would occur here

    def _simulate_tracking_correction(self, distance_mm: float) -> Tuple[float, float]:
        """Simulate computed tracking correction angles."""
        noise_x = random.uniform(-0.1, 0.1)
        noise_y = random.uniform(-0.1, 0.1)
        return noise_x, noise_y


_GLOBAL_BEAM_FOCUS_CONTROLLER: Optional[BeamFocusController] = None
_GLOBAL_BEAM_FOCUS_CONTROLLER_LOCK = threading.Lock()


def get_beam_focus_controller() -> BeamFocusController:
    """Return the global BeamFocusController singleton."""
    global _GLOBAL_BEAM_FOCUS_CONTROLLER
    with _GLOBAL_BEAM_FOCUS_CONTROLLER_LOCK:
        if _GLOBAL_BEAM_FOCUS_CONTROLLER is None:
            _GLOBAL_BEAM_FOCUS_CONTROLLER = BeamFocusController()
    return _GLOBAL_BEAM_FOCUS_CONTROLLER


def run_beam_focusing_tests() -> None:
    """Self-test for BeamFocusController."""
    logger.info("Running BeamFocusController tests...")
    ctrl = BeamFocusController()

    ctrl.set_mode(FocusMode.MANUAL)
    assert ctrl._mode == FocusMode.MANUAL

    assert ctrl.set_angle(5.0, -3.0), "Valid angle set failed"
    assert not ctrl.set_angle(20.0, 0.0), "Out-of-range angle should fail"

    assert ctrl.set_focal_distance(150.0), "Valid focal distance failed"
    assert not ctrl.set_focal_distance(1000.0), "Out-of-range distance should fail"

    ctrl.set_mode(FocusMode.AUTO)
    state = ctrl.auto_focus(250.0)
    assert state.is_locked, "Autofocus should lock"
    assert state.focal_distance_mm == 250.0

    logger.info("BeamFocusController tests PASSED")


if __name__ == "__main__":
    run_beam_focusing_tests()
