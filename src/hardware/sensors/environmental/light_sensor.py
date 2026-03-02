"""
Ambient Light Sensor — AI Holographic Wristwatch

Measures ambient light conditions for:
- Display brightness auto-adjustment
- Circadian rhythm tracking and sleep quality assessment
- Color temperature and UV index measurement
- Day/night and indoor/outdoor context
- SensorInterface compliance
"""

from __future__ import annotations

import threading
import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Deque, Dict, List, Optional

from src.core.interfaces.sensor_interface import SensorInterface, SensorReading, SensorStatus, SensorInfo
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class LightEnvironment(Enum):
    PITCH_DARK    = "pitch_dark"      # < 1 lux
    VERY_DIM      = "very_dim"        # 1–10 lux
    DIM           = "dim"             # 10–50 lux
    INDOOR_LOW    = "indoor_low"      # 50–200 lux
    INDOOR_NORMAL = "indoor_normal"   # 200–500 lux
    INDOOR_BRIGHT = "indoor_bright"   # 500–1000 lux
    OUTDOOR_SHADE = "outdoor_shade"   # 1000–10000 lux
    OUTDOOR_SUN   = "outdoor_sun"     # 10000–100000 lux
    DIRECT_SUN    = "direct_sun"      # > 100000 lux


class CircadianPhase(Enum):
    MORNING   = "morning"    # 06:00–10:00
    DAY       = "day"        # 10:00–17:00
    EVENING   = "evening"    # 17:00–21:00
    NIGHT     = "night"      # 21:00–06:00


@dataclass
class LightReading(SensorReading):
    illuminance_lux: float = 0.0
    color_temp_k: float = 5500.0       # correlated color temperature
    uv_index: float = 0.0              # 0–11+ UV index scale
    infrared_raw: float = 0.0
    environment: LightEnvironment = LightEnvironment.INDOOR_NORMAL
    circadian_phase: CircadianPhase = CircadianPhase.DAY
    recommended_brightness_pct: float = 50.0
    is_daytime: bool = True
    melanopic_lux: float = 0.0         # circadian stimulus (blue-weighted)


def _lux_to_environment(lux: float) -> LightEnvironment:
    if lux < 1:      return LightEnvironment.PITCH_DARK
    if lux < 10:     return LightEnvironment.VERY_DIM
    if lux < 50:     return LightEnvironment.DIM
    if lux < 200:    return LightEnvironment.INDOOR_LOW
    if lux < 500:    return LightEnvironment.INDOOR_NORMAL
    if lux < 1000:   return LightEnvironment.INDOOR_BRIGHT
    if lux < 10000:  return LightEnvironment.OUTDOOR_SHADE
    if lux < 100000: return LightEnvironment.OUTDOOR_SUN
    return LightEnvironment.DIRECT_SUN


def _lux_to_brightness(lux: float) -> float:
    """Logarithmic mapping: low light → lower brightness for comfort."""
    import math
    if lux <= 0: return 5.0
    log_lux = math.log10(max(1.0, lux))
    return min(100.0, max(5.0, log_lux * 20.0))


def _melanopic_lux(illuminance: float, color_temp_k: float) -> float:
    """Approximate melanopic (circadian) lux from illuminance and CCT."""
    # Warmer light (< 3000K) has less circadian effect
    factor = min(1.5, max(0.1, (color_temp_k - 2700) / 3000))
    return illuminance * factor


_GLOBAL_LIGHT: Optional["LightSensor"] = None
_GLOBAL_LIGHT_LOCK = threading.Lock()


class LightSensor(SensorInterface):
    """Ambient light sensor (APDS-9960 or TSL2591) with UV index and CCT."""

    SENSOR_ID    = "environmental.light"
    SENSOR_TYPE  = "ambient_light"
    MODEL        = "TSL2591+APDS-9960"
    MANUFACTURER = "AMS"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._running = self._initialized = False
        self._error_count = self._read_count = 0
        self._last_reading: Optional[LightReading] = None
        self._history: Deque[LightReading] = deque(maxlen=200)

    def initialize(self) -> bool:
        with self._lock:
            self._initialized = self._running = True
            logger.info(f"LightSensor {self.MODEL} initialized")
            return True

    def read(self) -> Optional[LightReading]:
        if not self._initialized:
            return None
        with self._lock:
            hour = time.localtime().tm_hour
            # Simulate lux based on time of day
            if 6 <= hour < 9:
                base_lux = random.gauss(800, 100)
                cct = 5000.0
                phase = CircadianPhase.MORNING
            elif 9 <= hour < 17:
                base_lux = random.gauss(5000, 500)
                cct = 6500.0
                phase = CircadianPhase.DAY
            elif 17 <= hour < 21:
                base_lux = random.gauss(300, 50)
                cct = 3500.0
                phase = CircadianPhase.EVENING
            else:
                base_lux = random.gauss(20, 5)
                cct = 2700.0
                phase = CircadianPhase.NIGHT

            lux = max(0.0, base_lux)
            uv = max(0.0, (lux / 10000) * 8 * random.uniform(0.8, 1.2))

            reading = LightReading(
                sensor_id=self.SENSOR_ID,
                timestamp=time.time(),
                illuminance_lux=lux,
                color_temp_k=cct + random.gauss(0, 100),
                uv_index=uv,
                infrared_raw=lux * 0.3,
                environment=_lux_to_environment(lux),
                circadian_phase=phase,
                recommended_brightness_pct=_lux_to_brightness(lux),
                is_daytime=(6 <= hour < 20),
                melanopic_lux=_melanopic_lux(lux, cct),
                confidence=0.95,
            )
            self._last_reading = reading
            self._history.append(reading)
            self._read_count += 1
            return reading

    async def stream(self) -> AsyncIterator[LightReading]:
        import asyncio
        while self._running:
            r = self.read()
            if r: yield r
            await asyncio.sleep(1.0)

    def calibrate(self) -> bool:
        logger.info("LightSensor: no hardware calibration needed")
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._running = self._initialized = False

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(sensor_id=self.SENSOR_ID, sensor_type=self.SENSOR_TYPE,
                          model=self.MODEL, manufacturer=self.MANUFACTURER,
                          firmware_version="1.0.0", hardware_version="rev-A",
                          capabilities={"lux_range_max": 88000, "uv_index": True,
                                        "color_temp": True, "ir": True})

    def get_status(self) -> SensorStatus:
        if not self._initialized: return SensorStatus.UNINITIALIZED
        return SensorStatus.RUNNING if self._running else SensorStatus.IDLE

    def is_healthy(self) -> bool:
        return self.get_status() in (SensorStatus.RUNNING, SensorStatus.IDLE)

    def get_health_report(self) -> Dict:
        return {"status": self.get_status().value, "read_count": self._read_count,
                "last_lux": self._last_reading.illuminance_lux if self._last_reading else None}

    def read_sync(self) -> Optional[LightReading]:
        return self.read()


def get_light_sensor() -> LightSensor:
    global _GLOBAL_LIGHT
    with _GLOBAL_LIGHT_LOCK:
        if _GLOBAL_LIGHT is None:
            _GLOBAL_LIGHT = LightSensor()
        return _GLOBAL_LIGHT


def run_light_sensor_tests() -> bool:
    ls = LightSensor()
    assert ls.initialize()
    r = ls.read()
    assert r is not None and r.illuminance_lux >= 0
    assert 0.0 <= r.uv_index <= 12.0
    ls.shutdown()
    logger.info("LightSensor tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_light_sensor_tests()
