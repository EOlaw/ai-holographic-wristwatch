"""
Skin Temperature Sensor Driver for AI Holographic Wristwatch

Wrist-worn infrared thermopile + contact thermistor for skin surface
temperature measurement. Compensates for ambient temperature to estimate
core body temperature. Used for fever detection, circadian rhythm analysis,
and sleep staging.

Hardware: MLX90632 infrared thermopile + NTC thermistor, 0.1 Hz continuous.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Deque, List, Optional

from ....core.constants import SensorConstants, HealthConstants, AlertSeverity
from ....core.exceptions import SensorReadError, SensorCalibrationError, CriticalHealthAlert
from ....core.interfaces import (
    SensorInterface, SensorReading, SensorInfo, SensorType,
    SensorReadingQuality, CalibrationResult as SensorCalibrationResult,
    SensorHealthReport
)


# ============================================================================
# Enumerations
# ============================================================================

class TemperatureZone(Enum):
    """Clinical temperature classification."""
    HYPOTHERMIA = "hypothermia"     # < 35.0°C
    LOW_NORMAL = "low_normal"       # 35.0–36.0°C
    NORMAL = "normal"               # 36.1–37.2°C
    ELEVATED = "elevated"           # 37.3–38.0°C
    FEVER = "fever"                 # 38.1–39.9°C
    HIGH_FEVER = "high_fever"       # 40.0–40.9°C
    HYPERPYREXIA = "hyperpyrexia"   # ≥ 41.0°C


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class TemperatureReading:
    """Skin and estimated core temperature with clinical zone."""
    skin_temp_c: float
    ambient_temp_c: float
    core_temp_estimate_c: float
    zone: TemperatureZone
    confidence: float           # 0.0–1.0 (affected by fit, motion)
    on_wrist: bool              # False → reading invalid (not worn)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_febrile(self) -> bool:
        return self.core_temp_estimate_c >= 38.0

    @property
    def is_emergency(self) -> bool:
        return self.core_temp_estimate_c >= 41.0 or self.core_temp_estimate_c < 35.0


# ============================================================================
# Core temperature estimation
# ============================================================================

class CoreTempEstimator:
    """
    Estimate core body temperature from wrist skin temperature + ambient.

    Uses a validated empirical offset:
        T_core ≈ T_skin + 2.5°C  (at rest, good wrist contact)
    Adjusted by ambient temperature differential and activity level.
    """
    BASE_OFFSET_C: float = 2.5
    AMBIENT_FACTOR: float = 0.15   # Correction per °C ambient deviation from 22°C

    def estimate(self, skin_c: float, ambient_c: float,
                 activity_factor: float = 0.0) -> float:
        """
        Args:
            skin_c: Measured skin temperature
            ambient_c: Ambient environment temperature
            activity_factor: 0.0 (rest) – 1.0 (intense exercise)
        """
        ambient_correction = (ambient_c - 22.0) * self.AMBIENT_FACTOR
        activity_correction = activity_factor * 0.8
        core = skin_c + self.BASE_OFFSET_C - ambient_correction + activity_correction
        return round(min(45.0, max(30.0, core)), 2)

    @staticmethod
    def classify(core_c: float) -> TemperatureZone:
        if core_c < 35.0:
            return TemperatureZone.HYPOTHERMIA
        if core_c < 36.1:
            return TemperatureZone.LOW_NORMAL
        if core_c <= 37.2:
            return TemperatureZone.NORMAL
        if core_c <= 38.0:
            return TemperatureZone.ELEVATED
        if core_c <= 39.9:
            return TemperatureZone.FEVER
        if core_c <= 40.9:
            return TemperatureZone.HIGH_FEVER
        return TemperatureZone.HYPERPYREXIA


# ============================================================================
# TemperatureSensor — SensorInterface implementation
# ============================================================================

class TemperatureSensor(SensorInterface):
    """
    Wrist skin temperature sensor with core temperature estimation.

    - 0.1 Hz continuous background sampling
    - 5-sample rolling median for noise rejection
    - Circadian rhythm tracking via 24-hour trend buffer
    - Fever alert on core temp ≥ 38.0°C
    """

    _SENSOR_ID = "biometric.skin_temperature"
    _TREND_BUFFER_HOURS = 24
    _TREND_BUFFER_SIZE = int(_TREND_BUFFER_HOURS * 3600 * SensorConstants.SKIN_TEMP_SAMPLING_RATE_HZ)

    def __init__(self) -> None:
        self._estimator = CoreTempEstimator()
        self._initialized = False
        self._calibrated = False
        self._ambient_temp_c: float = 22.0
        self._activity_factor: float = 0.0
        self._lock = threading.RLock()
        self._recent_readings: Deque[float] = deque(maxlen=5)
        self._trend_buffer: Deque[TemperatureReading] = deque(
            maxlen=min(self._TREND_BUFFER_SIZE, 8640))  # cap at 24h @ 0.1Hz
        self._last_reading: Optional[TemperatureReading] = None
        self._alert_callbacks: List[Callable[[CriticalHealthAlert], None]] = []
        self._baseline_skin_temp: Optional[float] = None

    async def initialize(self) -> bool:
        await asyncio.sleep(0.02)
        with self._lock:
            self._initialized = True
        return True

    async def read(self) -> SensorReading:
        if not self._initialized:
            raise SensorReadError("Temperature sensor not initialized",
                                  sensor_id=self._SENSOR_ID)

        skin_c = self._simulate_skin_temp()
        self._recent_readings.append(skin_c)

        # Median filter
        smoothed = statistics.median(self._recent_readings)
        core_c = self._estimator.estimate(smoothed, self._ambient_temp_c, self._activity_factor)
        zone = CoreTempEstimator.classify(core_c)

        # Confidence based on how many recent samples we have and baseline proximity
        confidence = min(1.0, len(self._recent_readings) / 5.0)
        if self._baseline_skin_temp:
            deviation = abs(smoothed - self._baseline_skin_temp)
            if deviation > 3.0:
                confidence *= 0.5  # Large deviation reduces confidence

        reading = TemperatureReading(
            skin_temp_c=round(smoothed, 2),
            ambient_temp_c=self._ambient_temp_c,
            core_temp_estimate_c=core_c,
            zone=zone,
            confidence=round(confidence, 3),
            on_wrist=True,
        )

        with self._lock:
            self._last_reading = reading
            self._trend_buffer.append(reading)

        self._check_alerts(reading)

        quality = SensorReadingQuality.GOOD if confidence > 0.8 else SensorReadingQuality.FAIR
        return SensorReading(
            sensor_id=self._SENSOR_ID,
            sensor_type=SensorType.SKIN_TEMPERATURE,
            values={
                "skin_temp_c": reading.skin_temp_c,
                "core_temp_estimate_c": reading.core_temp_estimate_c,
                "zone": reading.zone.value,
                "confidence": reading.confidence,
            },
            quality=quality,
            timestamp=reading.timestamp,
            unit="°C",
        )

    async def stream(self, interval_seconds: float = 10.0) -> AsyncIterator[SensorReading]:
        """Stream temperature readings. Default 10-second interval."""
        while self._initialized:
            yield await self.read()
            await asyncio.sleep(interval_seconds)

    async def calibrate(self) -> SensorCalibrationResult:
        if not self._initialized:
            raise SensorCalibrationError("Temp sensor not initialized",
                                          sensor_id=self._SENSOR_ID)
        # Take 10 samples to establish baseline
        samples = [self._simulate_skin_temp() for _ in range(10)]
        baseline = statistics.mean(samples)
        with self._lock:
            self._calibrated = True
            self._baseline_skin_temp = baseline
        return SensorCalibrationResult(
            sensor_id=self._SENSOR_ID, success=True, timestamp=time.time(),
            baseline_values={"skin_temp_baseline_c": baseline},
        )

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(
            sensor_id=self._SENSOR_ID, sensor_type=SensorType.SKIN_TEMPERATURE,
            model="MLX90632", sampling_rate_hz=SensorConstants.SKIN_TEMP_SAMPLING_RATE_HZ,
            resolution_bits=16, is_calibrated=self._calibrated,
        )

    def get_status(self) -> str:
        return "active" if self._initialized else "off"

    def is_healthy(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        with self._lock:
            self._initialized = False

    def get_health_report(self) -> SensorHealthReport:
        return SensorHealthReport(
            sensor_id=self._SENSOR_ID, is_operational=self._initialized,
            last_read_timestamp=self._last_reading.timestamp if self._last_reading else None,
            calibration_valid=self._calibrated, issues=[],
        )

    def read_sync(self) -> SensorReading:
        return asyncio.get_event_loop().run_until_complete(self.read())

    def get_24h_trend(self) -> List[TemperatureReading]:
        with self._lock:
            return list(self._trend_buffer)

    def set_ambient_temp(self, temp_c: float) -> None:
        with self._lock:
            self._ambient_temp_c = temp_c

    def set_activity_factor(self, factor: float) -> None:
        with self._lock:
            self._activity_factor = min(1.0, max(0.0, factor))

    def add_alert_callback(self, cb: Callable[[CriticalHealthAlert], None]) -> None:
        self._alert_callbacks.append(cb)

    def _simulate_skin_temp(self) -> float:
        """Simulate realistic wrist skin temperature with circadian variation."""
        import random
        circadian = math.sin(time.time() / 43200 * math.pi) * 0.5  # 12h cycle
        return 34.0 + circadian + random.gauss(0, 0.1)

    def _check_alerts(self, reading: TemperatureReading) -> None:
        if reading.is_emergency and reading.confidence > 0.7:
            severity = AlertSeverity.CRITICAL
            msg = (f"Hyperpyrexia: {reading.core_temp_estimate_c:.1f}°C"
                   if reading.core_temp_estimate_c >= 41.0
                   else f"Hypothermia: {reading.core_temp_estimate_c:.1f}°C")
            alert = CriticalHealthAlert(
                msg, health_metric="temperature",
                measured_value=reading.core_temp_estimate_c,
                threshold_value=41.0, severity=severity,
            )
            for cb in self._alert_callbacks:
                try:
                    cb(alert)
                except Exception:
                    pass
        elif reading.is_febrile and reading.confidence > 0.7:
            alert = CriticalHealthAlert(
                f"Fever detected: {reading.core_temp_estimate_c:.1f}°C",
                health_metric="temperature",
                measured_value=reading.core_temp_estimate_c,
                threshold_value=38.0, severity=AlertSeverity.WARNING,
            )
            for cb in self._alert_callbacks:
                try:
                    cb(alert)
                except Exception:
                    pass


# ============================================================================
# Global singleton
# ============================================================================

_temp_sensor: Optional[TemperatureSensor] = None
_temp_lock = threading.Lock()


def get_temperature_sensor() -> TemperatureSensor:
    global _temp_sensor
    with _temp_lock:
        if _temp_sensor is None:
            _temp_sensor = TemperatureSensor()
    return _temp_sensor


# ============================================================================
# Tests
# ============================================================================

def run_temperature_sensor_tests() -> None:
    print("Testing temperature sensor...")

    async def _run():
        sensor = TemperatureSensor()
        assert await sensor.initialize()
        cal = await sensor.calibrate()
        assert cal.success
        reading = await sensor.read()
        assert reading.sensor_type == SensorType.SKIN_TEMPERATURE
        assert "skin_temp_c" in reading.values
        assert "core_temp_estimate_c" in reading.values
        await sensor.shutdown()

    asyncio.get_event_loop().run_until_complete(_run())
    print("  Temperature sensor tests passed.")


__version__ = "1.0.0"
__all__ = [
    "TemperatureZone", "TemperatureReading",
    "CoreTempEstimator", "TemperatureSensor",
    "get_temperature_sensor",
]

if __name__ == "__main__":
    print("AI Holographic Wristwatch — Temperature Sensor")
    print("=" * 55)
    run_temperature_sensor_tests()
