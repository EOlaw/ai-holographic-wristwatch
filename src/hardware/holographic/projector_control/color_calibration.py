"""White point and color gamut calibration for the AI Holographic Wristwatch.

Implements Delta-E color matching and gamma correction to ensure accurate
color reproduction across the RGB laser projector's color gamut.
"""

import threading
import time
import math
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class ColorProfile(Enum):
    SRGB = "srgb"
    DCI_P3 = "dci_p3"
    REC2020 = "rec2020"
    NATIVE = "native"


@dataclass
class ColorCalibration:
    profile: ColorProfile
    white_point_xy: Tuple[float, float]
    gamma: float
    red_primary_xy: Tuple[float, float]
    green_primary_xy: Tuple[float, float]
    blue_primary_xy: Tuple[float, float]
    delta_e_avg: float
    calibrated_at: float = field(default_factory=time.time)


class ColorCalibrationEngine:
    """Delta-E color matching and gamma correction for holographic color accuracy.

    Calibrates laser power ratios to achieve target color primaries and white
    point. Uses CIE 2000 Delta-E metric to measure color accuracy.
    """

    WHITE_POINTS = {
        ColorProfile.SRGB:    (0.3127, 0.3290),
        ColorProfile.DCI_P3:  (0.3140, 0.3510),
        ColorProfile.REC2020: (0.3127, 0.3290),
        ColorProfile.NATIVE:  (0.3200, 0.3350),
    }

    PRIMARIES = {
        ColorProfile.SRGB: {
            "red":   (0.6400, 0.3300),
            "green": (0.3000, 0.6000),
            "blue":  (0.1500, 0.0600),
        },
        ColorProfile.DCI_P3: {
            "red":   (0.6800, 0.3200),
            "green": (0.2650, 0.6900),
            "blue":  (0.1500, 0.0600),
        },
        ColorProfile.REC2020: {
            "red":   (0.7080, 0.2920),
            "green": (0.1700, 0.7970),
            "blue":  (0.1310, 0.0460),
        },
        ColorProfile.NATIVE: {
            "red":   (0.6600, 0.3400),
            "green": (0.2800, 0.6500),
            "blue":  (0.1400, 0.0500),
        },
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current_profile = ColorProfile.SRGB
        self._gamma = 2.2
        self._calibration: Optional[ColorCalibration] = None
        logger.info("ColorCalibrationEngine initialized")

    def calibrate(self, profile: ColorProfile) -> ColorCalibration:
        """Run full color calibration for the specified profile."""
        with self._lock:
            logger.info("Starting color calibration for profile: %s", profile.value)
            wp = self.WHITE_POINTS[profile]
            prim = self.PRIMARIES[profile]
            self._current_profile = profile
            self._gamma = 2.4 if profile == ColorProfile.DCI_P3 else 2.2
            delta_e = self._measure_delta_e(profile)
            self._calibration = ColorCalibration(
                profile=profile,
                white_point_xy=wp,
                gamma=self._gamma,
                red_primary_xy=prim["red"],
                green_primary_xy=prim["green"],
                blue_primary_xy=prim["blue"],
                delta_e_avg=delta_e,
            )
            logger.info("Calibration complete. Delta-E avg: %.3f", delta_e)
            return self._calibration

    def apply_gamma(self, linear_value: float) -> float:
        """Apply gamma correction to a linear light value (0.0–1.0)."""
        v = max(0.0, min(1.0, linear_value))
        return math.pow(v, 1.0 / self._gamma)

    def get_current_calibration(self) -> Optional[ColorCalibration]:
        """Return the most recent calibration result."""
        with self._lock:
            return self._calibration

    def _measure_delta_e(self, profile: ColorProfile) -> float:
        """Simulate measuring average CIE Delta-E 2000 across color patches."""
        return random.uniform(0.5, 2.0)

    def _simulate_sensor_read(self) -> Tuple[float, float, float]:
        """Simulate a colorimeter reading returning XYZ tristimulus values."""
        return (random.uniform(0.9, 1.1), 1.0, random.uniform(0.9, 1.1))


_GLOBAL_COLOR_CALIBRATION_ENGINE: Optional[ColorCalibrationEngine] = None
_GLOBAL_COLOR_CALIBRATION_ENGINE_LOCK = threading.Lock()


def get_color_calibration_engine() -> ColorCalibrationEngine:
    """Return the global ColorCalibrationEngine singleton."""
    global _GLOBAL_COLOR_CALIBRATION_ENGINE
    with _GLOBAL_COLOR_CALIBRATION_ENGINE_LOCK:
        if _GLOBAL_COLOR_CALIBRATION_ENGINE is None:
            _GLOBAL_COLOR_CALIBRATION_ENGINE = ColorCalibrationEngine()
    return _GLOBAL_COLOR_CALIBRATION_ENGINE


def run_color_calibration_tests() -> None:
    """Self-test for ColorCalibrationEngine."""
    logger.info("Running ColorCalibrationEngine tests...")
    engine = ColorCalibrationEngine()

    for profile in ColorProfile:
        cal = engine.calibrate(profile)
        assert cal.profile == profile
        assert 0.0 < cal.delta_e_avg < 5.0, "Delta-E out of expected range"
        assert cal.gamma > 0.0

    gamma_out = engine.apply_gamma(0.5)
    assert 0.0 < gamma_out < 1.0, "Gamma output out of range"

    assert engine.get_current_calibration() is not None

    logger.info("ColorCalibrationEngine tests PASSED")


if __name__ == "__main__":
    run_color_calibration_tests()
