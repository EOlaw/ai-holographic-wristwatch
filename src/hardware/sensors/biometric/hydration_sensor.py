"""
Hydration Sensor Driver for AI Holographic Wristwatch

Estimates skin hydration status using bioelectrical impedance analysis (BIA)
via wrist electrodes and optical skin assessment from the PPG sensor.
Provides hydration reminders and trend tracking. Feature-flagged as beta
(30% rollout) pending clinical validation.

Hardware: BIA electrodes + optical reflectance at 0.017 Hz (every minute).
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
from typing import AsyncIterator, Deque, List, Optional

from ....core.constants import AlertSeverity
from ....core.exceptions import SensorReadError, CriticalHealthAlert
from ....core.interfaces import (
    SensorInterface, SensorReading, SensorInfo, SensorType,
    SensorReadingQuality, CalibrationResult as SensorCalibrationResult,
    SensorHealthReport
)


# ============================================================================
# Enumerations
# ============================================================================

class HydrationStatus(Enum):
    """Clinical hydration classification."""
    OVERHYDRATED = "overhydrated"       # Rare but possible
    WELL_HYDRATED = "well_hydrated"     # 55–65% body water (typical adult)
    MILDLY_DEHYDRATED = "mildly_dehydrated"   # 1–2% body weight loss
    DEHYDRATED = "dehydrated"           # 2–5% body weight loss
    SEVERELY_DEHYDRATED = "severely_dehydrated"  # > 5% body weight loss


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class BIAMeasurement:
    """Raw bioelectrical impedance measurement."""
    resistance_ohm: float       # Real component of impedance
    reactance_ohm: float        # Imaginary component
    phase_angle_deg: float      # arctan(Xc/R) — correlates with cell health
    frequency_hz: float         # Measurement frequency


@dataclass
class HydrationReading:
    """Estimated hydration status."""
    hydration_pct: float            # Estimated % body water (50–70% normal)
    status: HydrationStatus
    deficit_ml: float               # Estimated fluid deficit in mL
    recommendation_ml: float        # Recommended fluid intake for next hour
    skin_capacitance: float         # Optical hydration proxy
    bia: Optional[BIAMeasurement]
    confidence: float               # 0.0–1.0 (beta feature, lower confidence)
    timestamp: float = field(default_factory=time.time)

    @property
    def needs_hydration_alert(self) -> bool:
        return self.status in (HydrationStatus.DEHYDRATED,
                               HydrationStatus.SEVERELY_DEHYDRATED)


# ============================================================================
# BIA Algorithm
# ============================================================================

class BIAHydrationEstimator:
    """
    Estimates total body water (TBW) from bioelectrical impedance.

    Uses the Kyle (2004) population equation:
        TBW = 0.396 * (height_cm² / R) + 0.143 * weight_kg + 6.56 (female)
        TBW = 0.396 * (height_cm² / R) + 0.143 * weight_kg + 8.40 (male)
    Body water percentage = TBW / weight_kg * 100
    """

    MALE_CONSTANT = 8.40
    FEMALE_CONSTANT = 6.56
    TBW_COEFF_H2_R = 0.396
    TBW_COEFF_WEIGHT = 0.143

    def estimate(
        self,
        resistance_ohm: float,
        height_cm: float,
        weight_kg: float,
        is_male: bool = True,
    ) -> float:
        """Returns estimated body water percentage."""
        if resistance_ohm <= 0:
            return 55.0
        constant = self.MALE_CONSTANT if is_male else self.FEMALE_CONSTANT
        tbw = (self.TBW_COEFF_H2_R * (height_cm ** 2 / resistance_ohm)
               + self.TBW_COEFF_WEIGHT * weight_kg
               + constant)
        tbw_pct = (tbw / weight_kg) * 100
        return round(min(75.0, max(40.0, tbw_pct)), 1)

    @staticmethod
    def classify(hydration_pct: float, baseline_pct: float) -> HydrationStatus:
        delta = hydration_pct - baseline_pct
        if hydration_pct > 65.0:
            return HydrationStatus.OVERHYDRATED
        if delta > -1.0:
            return HydrationStatus.WELL_HYDRATED
        if delta > -2.0:
            return HydrationStatus.MILDLY_DEHYDRATED
        if delta > -5.0:
            return HydrationStatus.DEHYDRATED
        return HydrationStatus.SEVERELY_DEHYDRATED

    @staticmethod
    def estimate_deficit_ml(hydration_pct: float, weight_kg: float,
                            baseline_pct: float = 60.0) -> float:
        """Estimate fluid deficit in mL."""
        current_tbw_l = hydration_pct / 100.0 * weight_kg
        target_tbw_l = baseline_pct / 100.0 * weight_kg
        deficit_l = max(0.0, target_tbw_l - current_tbw_l)
        return round(deficit_l * 1000, 0)


# ============================================================================
# HydrationSensor — SensorInterface implementation
# ============================================================================

class HydrationSensor(SensorInterface):
    """
    Beta-flagged wrist hydration sensor combining BIA and optical reflectance.
    Provides per-minute hydration estimates and hourly intake recommendations.
    """

    _SENSOR_ID = "biometric.hydration"

    def __init__(self, weight_kg: float = 70.0, height_cm: float = 175.0,
                 is_male: bool = True) -> None:
        self._estimator = BIAHydrationEstimator()
        self._weight_kg = weight_kg
        self._height_cm = height_cm
        self._is_male = is_male
        self._initialized = False
        self._calibrated = False
        self._baseline_pct: float = 60.0
        self._lock = threading.RLock()
        self._history: Deque[HydrationReading] = deque(maxlen=1440)  # 24h @ 1/min
        self._last_reading: Optional[HydrationReading] = None
        self._total_intake_ml: float = 0.0

    async def initialize(self) -> bool:
        await asyncio.sleep(0.02)
        with self._lock:
            self._initialized = True
        return True

    async def read(self) -> SensorReading:
        if not self._initialized:
            raise SensorReadError("Hydration sensor not initialized", sensor_id=self._SENSOR_ID)

        resistance, reactance = self._simulate_bia()
        phase_angle = math.degrees(math.atan2(reactance, resistance))
        bia = BIAMeasurement(
            resistance_ohm=resistance,
            reactance_ohm=reactance,
            phase_angle_deg=round(phase_angle, 2),
            frequency_hz=50_000.0,
        )

        hydration_pct = self._estimator.estimate(resistance, self._height_cm,
                                                  self._weight_kg, self._is_male)
        status = BIAHydrationEstimator.classify(hydration_pct, self._baseline_pct)
        deficit_ml = BIAHydrationEstimator.estimate_deficit_ml(
            hydration_pct, self._weight_kg, self._baseline_pct)
        recommendation_ml = min(500.0, deficit_ml / 4.0 + 150.0)  # sip every 15 min
        skin_cap = self._simulate_skin_capacitance()

        reading = HydrationReading(
            hydration_pct=hydration_pct,
            status=status,
            deficit_ml=deficit_ml,
            recommendation_ml=round(recommendation_ml, 0),
            skin_capacitance=round(skin_cap, 4),
            bia=bia,
            confidence=0.65,  # Beta feature
        )

        with self._lock:
            self._last_reading = reading
            self._history.append(reading)

        return SensorReading(
            sensor_id=self._SENSOR_ID, sensor_type=SensorType.HEART_RATE,
            values={
                "hydration_pct": hydration_pct,
                "status": status.value,
                "deficit_ml": deficit_ml,
                "recommendation_ml": recommendation_ml,
            },
            quality=SensorReadingQuality.FAIR,  # Beta
            timestamp=reading.timestamp, unit="%",
        )

    async def stream(self, interval_seconds: float = 60.0) -> AsyncIterator[SensorReading]:
        while self._initialized:
            yield await self.read()
            await asyncio.sleep(interval_seconds)

    async def calibrate(self) -> SensorCalibrationResult:
        samples = []
        for _ in range(5):
            r, _ = self._simulate_bia()
            pct = self._estimator.estimate(r, self._height_cm, self._weight_kg, self._is_male)
            samples.append(pct)
        baseline = statistics.mean(samples)
        with self._lock:
            self._calibrated = True
            self._baseline_pct = baseline
        return SensorCalibrationResult(
            sensor_id=self._SENSOR_ID, success=True, timestamp=time.time(),
            baseline_values={"baseline_hydration_pct": baseline},
        )

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(
            sensor_id=self._SENSOR_ID, sensor_type=SensorType.HEART_RATE,
            model="BIA-WRIST-01", sampling_rate_hz=1.0 / 60.0,
            resolution_bits=12, is_calibrated=self._calibrated,
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

    def log_fluid_intake(self, ml: float) -> None:
        """Record manual fluid intake entry."""
        with self._lock:
            self._total_intake_ml += ml

    def get_daily_intake_ml(self) -> float:
        with self._lock:
            return self._total_intake_ml

    def _simulate_bia(self):
        import random
        r = max(10.0, 500 + math.sin(time.time() / 3600) * 50 + random.gauss(0, 10))
        xc = max(1.0, 60 + random.gauss(0, 3))
        return r, xc

    def _simulate_skin_capacitance(self) -> float:
        import random
        return max(0.1, 0.8 + math.sin(time.time() / 1800) * 0.2 + random.gauss(0, 0.01))


# ============================================================================
# Global singleton
# ============================================================================

_hydration_sensor: Optional[HydrationSensor] = None
_hydration_lock = threading.Lock()


def get_hydration_sensor() -> HydrationSensor:
    global _hydration_sensor
    with _hydration_lock:
        if _hydration_sensor is None:
            _hydration_sensor = HydrationSensor()
    return _hydration_sensor


# ============================================================================
# Tests
# ============================================================================

def run_hydration_sensor_tests() -> None:
    print("Testing hydration sensor...")

    async def _run():
        sensor = HydrationSensor()
        assert await sensor.initialize()
        cal = await sensor.calibrate()
        assert cal.success
        reading = await sensor.read()
        assert "hydration_pct" in reading.values
        assert 40 <= reading.values["hydration_pct"] <= 75
        await sensor.shutdown()

    asyncio.get_event_loop().run_until_complete(_run())
    print("  Hydration sensor tests passed.")


__version__ = "1.0.0"
__all__ = [
    "HydrationStatus", "BIAMeasurement", "HydrationReading",
    "BIAHydrationEstimator", "HydrationSensor",
    "get_hydration_sensor",
]

if __name__ == "__main__":
    print("AI Holographic Wristwatch — Hydration Sensor")
    print("=" * 55)
    run_hydration_sensor_tests()
