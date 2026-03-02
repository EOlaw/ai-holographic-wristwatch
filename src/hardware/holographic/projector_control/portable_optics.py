"""Miniaturized optics alignment for the AI Holographic Wristwatch projector.

Controls diffractive optical elements (DOEs) for beam shaping and performs
alignment verification to maintain holographic image quality in a wearable form.
"""

import threading
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class OpticsState(Enum):
    UNINITIALIZED = "uninitialized"
    ALIGNED = "aligned"
    MISALIGNED = "misaligned"
    CALIBRATING = "calibrating"
    FAULT = "fault"


@dataclass
class OpticsReading:
    state: OpticsState
    doe_position_um: Tuple[float, float, float]
    alignment_error_um: float
    beam_uniformity_pct: float
    diffraction_efficiency_pct: float
    timestamp: float = field(default_factory=time.time)


class PortableOptics:
    """Diffractive optical element control and alignment verification.

    Manages the translation and tilt of DOE elements using piezoelectric
    actuators. Alignment is verified by measuring a reference beam pattern.
    """

    ALIGNMENT_TOLERANCE_UM = 2.0
    DOE_TRAVEL_RANGE_UM = (-50.0, 50.0)

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = OpticsState.UNINITIALIZED
        self._doe_position = (0.0, 0.0, 0.0)
        self._alignment_error_um = 99.0
        logger.info("PortableOptics initialized")

    def initialize(self) -> bool:
        """Home DOE actuators and perform initial alignment."""
        with self._lock:
            self._state = OpticsState.CALIBRATING
            logger.info("Homing DOE actuators...")
            self._doe_position = (0.0, 0.0, 0.0)
            self._write_hardware_position(0.0, 0.0, 0.0)
            self._alignment_error_um = self._measure_alignment_error()
            if self._alignment_error_um <= self.ALIGNMENT_TOLERANCE_UM:
                self._state = OpticsState.ALIGNED
                logger.info("Optics aligned. Error: %.2f µm", self._alignment_error_um)
                return True
            self._state = OpticsState.MISALIGNED
            logger.warning("Alignment error %.2f µm exceeds tolerance", self._alignment_error_um)
            return False

    def correct_alignment(self) -> bool:
        """Run iterative alignment correction loop."""
        with self._lock:
            self._state = OpticsState.CALIBRATING
            for iteration in range(10):
                error = self._measure_alignment_error()
                if error <= self.ALIGNMENT_TOLERANCE_UM:
                    self._alignment_error_um = error
                    self._state = OpticsState.ALIGNED
                    logger.info("Aligned after %d iterations. Error: %.2f µm",
                                iteration + 1, error)
                    return True
                correction = self._calculate_correction(error)
                x, y, z = self._doe_position
                new_pos = (
                    x + correction[0],
                    y + correction[1],
                    z + correction[2],
                )
                lo, hi = self.DOE_TRAVEL_RANGE_UM
                new_pos = tuple(max(lo, min(hi, v)) for v in new_pos)
                self._doe_position = new_pos
                self._write_hardware_position(*new_pos)
            self._state = OpticsState.MISALIGNED
            logger.error("Alignment correction failed after 10 iterations")
            return False

    def read_state(self) -> OpticsReading:
        """Return current optics status and alignment metrics."""
        with self._lock:
            return OpticsReading(
                state=self._state,
                doe_position_um=self._doe_position,
                alignment_error_um=self._alignment_error_um,
                beam_uniformity_pct=self._simulate_beam_uniformity(),
                diffraction_efficiency_pct=self._simulate_diffraction_efficiency(),
            )

    def _write_hardware_position(self, x: float, y: float, z: float) -> None:
        """Simulate writing DOE position to piezo actuator controllers."""
        pass  # Hardware SPI/I2C write would occur here

    def _measure_alignment_error(self) -> float:
        """Simulate measuring alignment error from reference beam detector."""
        return random.uniform(0.1, 5.0)

    def _calculate_correction(self, error: float) -> Tuple[float, float, float]:
        """Compute correction vector from measured alignment error."""
        step = error * 0.3
        return (
            random.uniform(-step, step),
            random.uniform(-step, step),
            random.uniform(-step * 0.1, step * 0.1),
        )

    def _simulate_beam_uniformity(self) -> float:
        """Simulate beam uniformity measurement (percent of ideal)."""
        return random.uniform(85.0, 99.0)

    def _simulate_diffraction_efficiency(self) -> float:
        """Simulate diffractive efficiency measurement."""
        return random.uniform(70.0, 95.0)


_GLOBAL_PORTABLE_OPTICS: Optional[PortableOptics] = None
_GLOBAL_PORTABLE_OPTICS_LOCK = threading.Lock()


def get_portable_optics() -> PortableOptics:
    """Return the global PortableOptics singleton."""
    global _GLOBAL_PORTABLE_OPTICS
    with _GLOBAL_PORTABLE_OPTICS_LOCK:
        if _GLOBAL_PORTABLE_OPTICS is None:
            _GLOBAL_PORTABLE_OPTICS = PortableOptics()
    return _GLOBAL_PORTABLE_OPTICS


def run_portable_optics_tests() -> None:
    """Self-test for PortableOptics."""
    logger.info("Running PortableOptics tests...")
    optics = PortableOptics()

    assert optics._state == OpticsState.UNINITIALIZED
    optics.initialize()
    reading = optics.read_state()
    assert reading.state in (OpticsState.ALIGNED, OpticsState.MISALIGNED)
    assert 0.0 <= reading.beam_uniformity_pct <= 100.0
    assert 0.0 <= reading.diffraction_efficiency_pct <= 100.0

    optics.correct_alignment()
    reading2 = optics.read_state()
    assert reading2.state in (OpticsState.ALIGNED, OpticsState.MISALIGNED)

    logger.info("PortableOptics tests PASSED")


if __name__ == "__main__":
    run_portable_optics_tests()
