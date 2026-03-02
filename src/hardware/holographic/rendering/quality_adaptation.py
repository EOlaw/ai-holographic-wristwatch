"""Dynamic quality Level-of-Detail adaptation for the AI Holographic Wristwatch.

Adjusts holographic render quality based on battery level, thermal state, and
computational load to maintain smooth playback within power and thermal budgets.
"""

import threading
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class RenderQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class QualityState:
    quality: RenderQuality
    battery_pct: float
    temperature_c: float
    frame_budget_ms: float
    polygon_budget: int
    texture_resolution: int
    timestamp: float = field(default_factory=time.time)


_QUALITY_PARAMS = {
    RenderQuality.LOW:    {"frame_ms": 33.3, "polygons": 5_000,   "tex_res": 256},
    RenderQuality.MEDIUM: {"frame_ms": 16.7, "polygons": 20_000,  "tex_res": 512},
    RenderQuality.HIGH:   {"frame_ms": 11.1, "polygons": 80_000,  "tex_res": 1024},
    RenderQuality.ULTRA:  {"frame_ms": 8.3,  "polygons": 250_000, "tex_res": 2048},
}


class QualityAdaptationEngine:
    """Battery- and thermal-aware holographic render quality selector.

    Automatically downscales render quality when battery drops below thresholds
    or when the SoC thermal sensor exceeds safe operating temperatures. Quality
    can also be manually overridden and locked by the user.
    """

    BATTERY_THRESHOLDS = {
        RenderQuality.ULTRA:  75.0,
        RenderQuality.HIGH:   50.0,
        RenderQuality.MEDIUM: 25.0,
        RenderQuality.LOW:    0.0,
    }
    THERMAL_LIMIT_C = 70.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._quality = RenderQuality.HIGH
        self._locked = False
        logger.info("QualityAdaptationEngine initialized at %s quality", self._quality.value)

    def update(self) -> QualityState:
        """Sample system state and adapt quality; returns the resolved QualityState."""
        with self._lock:
            battery = self._read_battery_pct()
            temperature = self._read_temperature_c()

            if not self._locked:
                if temperature >= self.THERMAL_LIMIT_C:
                    self._quality = RenderQuality.LOW
                    logger.warning("Thermal throttle: quality -> LOW (%.1f°C)", temperature)
                else:
                    self._quality = self._select_quality_for_battery(battery)

            params = _QUALITY_PARAMS[self._quality]
            return QualityState(
                quality=self._quality,
                battery_pct=battery,
                temperature_c=temperature,
                frame_budget_ms=params["frame_ms"],
                polygon_budget=params["polygons"],
                texture_resolution=params["tex_res"],
            )

    def set_quality(self, quality: RenderQuality, lock: bool = False) -> None:
        """Manually set quality level. Pass lock=True to prevent auto-adaptation."""
        with self._lock:
            self._quality = quality
            self._locked = lock
            logger.info("Quality manually set to %s (locked=%s)", quality.value, lock)

    def unlock(self) -> None:
        """Release quality lock and re-enable automatic adaptation."""
        with self._lock:
            self._locked = False
            logger.info("Quality adaptation lock released")

    def _select_quality_for_battery(self, battery_pct: float) -> RenderQuality:
        """Choose the highest permissible quality for the given battery level."""
        for quality in (RenderQuality.ULTRA, RenderQuality.HIGH,
                        RenderQuality.MEDIUM, RenderQuality.LOW):
            if battery_pct >= self.BATTERY_THRESHOLDS[quality]:
                return quality
        return RenderQuality.LOW

    def _read_battery_pct(self) -> float:
        """Simulate reading battery percentage from power management IC."""
        return random.uniform(20.0, 100.0)

    def _read_temperature_c(self) -> float:
        """Simulate reading SoC temperature from thermal sensor."""
        return random.uniform(35.0, 68.0)


_GLOBAL_QUALITY_ADAPTATION_ENGINE: Optional[QualityAdaptationEngine] = None
_GLOBAL_QUALITY_ADAPTATION_ENGINE_LOCK = threading.Lock()


def get_quality_adaptation_engine() -> QualityAdaptationEngine:
    """Return the global QualityAdaptationEngine singleton."""
    global _GLOBAL_QUALITY_ADAPTATION_ENGINE
    with _GLOBAL_QUALITY_ADAPTATION_ENGINE_LOCK:
        if _GLOBAL_QUALITY_ADAPTATION_ENGINE is None:
            _GLOBAL_QUALITY_ADAPTATION_ENGINE = QualityAdaptationEngine()
    return _GLOBAL_QUALITY_ADAPTATION_ENGINE


def run_quality_adaptation_tests() -> None:
    """Self-test for QualityAdaptationEngine."""
    logger.info("Running QualityAdaptationEngine tests...")
    engine = QualityAdaptationEngine()

    state = engine.update()
    assert isinstance(state.quality, RenderQuality)
    assert state.frame_budget_ms > 0.0
    assert state.polygon_budget > 0

    engine.set_quality(RenderQuality.ULTRA, lock=True)
    state2 = engine.update()
    assert state2.quality == RenderQuality.ULTRA, "Lock should prevent adaptation"

    engine.unlock()
    engine.set_quality(RenderQuality.LOW)
    state3 = engine.update()
    assert state3.polygon_budget == _QUALITY_PARAMS[RenderQuality.LOW]["polygons"]

    logger.info("QualityAdaptationEngine tests PASSED")


if __name__ == "__main__":
    run_quality_adaptation_tests()
