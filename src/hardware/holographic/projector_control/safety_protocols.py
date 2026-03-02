"""Laser safety compliance manager for the AI Holographic Wristwatch (IEC 60825).

Enforces eye exposure limits, proximity sensor interlocks, and provides
emergency shutdown capability to maintain laser safety classifications.
"""

import threading
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Callable

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class SafetyLevel(Enum):
    CLASS_1 = "class_1"
    CLASS_1M = "class_1m"
    CLASS_2 = "class_2"
    CLASS_3R = "class_3r"
    CLASS_3B = "class_3b"
    CLASS_4 = "class_4"


class HazardType(Enum):
    EYE_EXPOSURE = "eye_exposure"
    SKIN_EXPOSURE = "skin_exposure"
    PROXIMITY_VIOLATION = "proximity_violation"
    THERMAL_OVERLOAD = "thermal_overload"
    POWER_LIMIT_EXCEEDED = "power_limit_exceeded"


@dataclass
class SafetyReading:
    safety_level: SafetyLevel
    hazard_detected: bool
    hazard_type: Optional[HazardType]
    proximity_mm: float
    exposure_time_s: float
    cumulative_dose_mw_s: float
    interlock_active: bool
    timestamp: float = field(default_factory=time.time)


class SafetyProtocolManager:
    """IEC 60825 laser safety compliance — eye exposure limits and interlocks.

    Monitors proximity sensors, cumulative exposure dose, and power levels.
    Triggers emergency shutdown via registered callbacks when thresholds are
    exceeded. Designed for continuous operation in a wearable device context.
    """

    MIN_SAFE_PROXIMITY_MM = 100.0
    MAX_CONTINUOUS_EXPOSURE_S = 0.25
    MAX_DOSE_MW_S = 1.25
    OPERATING_SAFETY_LEVEL = SafetyLevel.CLASS_1

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._interlock_active = False
        self._cumulative_dose = 0.0
        self._exposure_start: Optional[float] = None
        self._shutdown_callbacks: List[Callable[[], None]] = []
        self._last_hazard: Optional[HazardType] = None
        logger.info("SafetyProtocolManager initialized (IEC 60825 Class 1)")

    def register_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Register a callable invoked on emergency shutdown."""
        with self._lock:
            self._shutdown_callbacks.append(callback)

    def check_safety(self) -> SafetyReading:
        """Run all safety checks and return current safety reading."""
        with self._lock:
            proximity = self._read_proximity_sensor()
            exposure_s = self._calculate_exposure_time()
            dose = self._cumulative_dose + (exposure_s * self._estimate_power_mw())

            hazard = None
            if proximity < self.MIN_SAFE_PROXIMITY_MM:
                hazard = HazardType.PROXIMITY_VIOLATION
            elif exposure_s > self.MAX_CONTINUOUS_EXPOSURE_S:
                hazard = HazardType.EYE_EXPOSURE
            elif dose > self.MAX_DOSE_MW_S:
                hazard = HazardType.EYE_EXPOSURE

            if hazard is not None and not self._interlock_active:
                self._last_hazard = hazard
                logger.warning("Safety hazard: %s — triggering shutdown", hazard.value)
                self._trigger_emergency_shutdown()

            return SafetyReading(
                safety_level=self.OPERATING_SAFETY_LEVEL,
                hazard_detected=hazard is not None,
                hazard_type=hazard,
                proximity_mm=proximity,
                exposure_time_s=exposure_s,
                cumulative_dose_mw_s=dose,
                interlock_active=self._interlock_active,
            )

    def reset_exposure_timer(self) -> None:
        """Reset the cumulative dose counter (call when laser beam is interrupted)."""
        with self._lock:
            self._cumulative_dose = 0.0
            self._exposure_start = None
            logger.debug("Exposure timer reset")

    def release_interlock(self) -> None:
        """Manually release the safety interlock after operator confirmation."""
        with self._lock:
            self._interlock_active = False
            self._last_hazard = None
            self.reset_exposure_timer()
            logger.info("Safety interlock manually released")

    def _trigger_emergency_shutdown(self) -> None:
        """Activate interlock and invoke all shutdown callbacks."""
        self._interlock_active = True
        for cb in self._shutdown_callbacks:
            try:
                cb()
            except Exception as exc:
                logger.error("Shutdown callback error: %s", exc)
        logger.critical("Emergency shutdown triggered")

    def _read_proximity_sensor(self) -> float:
        """Simulate reading proximity sensor distance in millimeters."""
        return random.uniform(150.0, 400.0)

    def _calculate_exposure_time(self) -> float:
        """Calculate continuous laser exposure time in seconds."""
        now = time.time()
        if self._exposure_start is None:
            self._exposure_start = now
            return 0.0
        return now - self._exposure_start

    def _estimate_power_mw(self) -> float:
        """Estimate combined laser power for dose calculation."""
        return random.uniform(0.1, 5.0)


_GLOBAL_SAFETY_PROTOCOL_MANAGER: Optional[SafetyProtocolManager] = None
_GLOBAL_SAFETY_PROTOCOL_MANAGER_LOCK = threading.Lock()


def get_safety_protocol_manager() -> SafetyProtocolManager:
    """Return the global SafetyProtocolManager singleton."""
    global _GLOBAL_SAFETY_PROTOCOL_MANAGER
    with _GLOBAL_SAFETY_PROTOCOL_MANAGER_LOCK:
        if _GLOBAL_SAFETY_PROTOCOL_MANAGER is None:
            _GLOBAL_SAFETY_PROTOCOL_MANAGER = SafetyProtocolManager()
    return _GLOBAL_SAFETY_PROTOCOL_MANAGER


def run_safety_protocols_tests() -> None:
    """Self-test for SafetyProtocolManager."""
    logger.info("Running SafetyProtocolManager tests...")
    manager = SafetyProtocolManager()

    shutdown_called = []
    manager.register_shutdown_callback(lambda: shutdown_called.append(True))

    reading = manager.check_safety()
    assert isinstance(reading.safety_level, SafetyLevel)
    assert reading.proximity_mm > 0.0

    manager.reset_exposure_timer()
    manager.release_interlock()
    assert not manager._interlock_active

    logger.info("SafetyProtocolManager tests PASSED")


if __name__ == "__main__":
    run_safety_protocols_tests()
