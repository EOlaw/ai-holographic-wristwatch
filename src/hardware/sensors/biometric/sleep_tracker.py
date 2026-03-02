"""
Sleep Tracker Driver for AI Holographic Wristwatch

Automatic sleep stage classification (Wake / REM / Light / Deep) using
multi-modal sensor fusion: accelerometer, PPG/HRV, skin temperature
circadian patterns, and SpO2. Detects sleep onset/offset automatically
via wrist movement and HR patterns.

Algorithm: Rule-based heuristic classifier with confidence scoring.
For production: replace with on-device TFLite sleep staging model.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, List, Optional

from ....core.constants import SensorConstants, AlertSeverity
from ....core.exceptions import SensorReadError, CriticalHealthAlert
from ....core.interfaces import (
    SensorInterface, SensorReading, SensorInfo, SensorType,
    SensorReadingQuality, CalibrationResult as SensorCalibrationResult,
    SensorHealthReport
)


# ============================================================================
# Enumerations
# ============================================================================

class SleepStage(Enum):
    """Polysomnography-aligned sleep stages."""
    AWAKE = "awake"
    REM = "rem"             # Rapid Eye Movement — dreaming
    LIGHT = "light"         # N1 + N2
    DEEP = "deep"           # N3 — slow-wave sleep, restorative


class SleepQuality(Enum):
    """Overall sleep quality assessment."""
    EXCELLENT = "excellent"     # ≥ 85 score
    GOOD = "good"               # 70–84
    FAIR = "fair"               # 50–69
    POOR = "poor"               # < 50


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class SleepEpoch:
    """A 30-second classification epoch (PSG standard)."""
    stage: SleepStage
    confidence: float           # 0.0–1.0
    heart_rate_bpm: float
    movement_count: int         # Accelerometer events in epoch
    hrv_rmssd_ms: Optional[float]
    spo2_pct: Optional[float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SleepSession:
    """A complete sleep session summary."""
    session_id: str
    start_time: float
    end_time: Optional[float]
    total_minutes: float
    rem_minutes: float
    light_minutes: float
    deep_minutes: float
    awake_minutes: float
    sleep_efficiency_pct: float      # time_asleep / time_in_bed * 100
    sleep_latency_minutes: float     # Time from bed to first sleep
    rem_latency_minutes: float       # Time from sleep to first REM
    num_awakenings: int
    quality: SleepQuality
    quality_score: float             # 0–100
    epochs: List[SleepEpoch] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.end_time is not None


# ============================================================================
# Sleep stage classifier
# ============================================================================

class SleepStageClassifier:
    """
    Rule-based heuristic sleep stage classifier.

    Inputs per 30-second epoch:
        - Average heart rate
        - HRV RMSSD
        - Movement count (accel events)
        - SpO2
        - Time of night (circadian position)

    Stage assignment logic (simplified from AASM guidelines):
        AWAKE: high movement OR HR above 85% of resting max
        DEEP: low HR + low HRV + very low movement + early night
        REM: low movement + elevated HR variability + later night
        LIGHT: default (does not fit other criteria)
    """

    def classify(
        self,
        hr_bpm: float,
        hrv_rmssd_ms: Optional[float],
        movement_count: int,
        spo2_pct: Optional[float],
        minutes_since_sleep_onset: float,
        resting_hr: float = 60.0,
    ) -> tuple[SleepStage, float]:
        """Returns (stage, confidence)."""

        # ── Awake detection ───────────────────────────────────────────────
        if movement_count > 10 or hr_bpm > resting_hr * 1.3:
            return SleepStage.AWAKE, 0.85

        # ── Deep sleep (N3) ────────────────────────────────────────────────
        is_early_night = minutes_since_sleep_onset < 180  # First 3 hours
        low_hr = hr_bpm < resting_hr * 0.85
        low_motion = movement_count <= 1
        low_hrv = hrv_rmssd_ms is not None and hrv_rmssd_ms < 25

        if low_hr and low_motion and low_hrv and is_early_night:
            return SleepStage.DEEP, 0.78

        # ── REM ────────────────────────────────────────────────────────────
        is_later_night = minutes_since_sleep_onset > 90
        rem_hr = resting_hr * 0.9 < hr_bpm < resting_hr * 1.15
        high_hrv = hrv_rmssd_ms is not None and hrv_rmssd_ms > 45
        very_low_motion = movement_count == 0

        if rem_hr and high_hrv and very_low_motion and is_later_night:
            return SleepStage.REM, 0.72

        # ── Light (default) ────────────────────────────────────────────────
        return SleepStage.LIGHT, 0.60


# ============================================================================
# SleepTracker — SensorInterface implementation
# ============================================================================

class SleepTracker(SensorInterface):
    """
    Automatic sleep tracking with stage classification and quality scoring.
    Monitors passively during wear; auto-detects sleep onset and offset.
    """

    _SENSOR_ID = "biometric.sleep"
    EPOCH_SECONDS = 30

    def __init__(self) -> None:
        self._classifier = SleepStageClassifier()
        self._initialized = False
        self._calibrated = False
        self._lock = threading.RLock()
        self._current_session: Optional[SleepSession] = None
        self._last_session: Optional[SleepSession] = None
        self._resting_hr: float = 62.0
        self._sleep_onset_time: Optional[float] = None
        self._alert_callbacks: List[Callable[[CriticalHealthAlert], None]] = []
        self._epoch_counter: int = 0

    async def initialize(self) -> bool:
        await asyncio.sleep(0.02)
        with self._lock:
            self._initialized = True
        return True

    async def read(self) -> SensorReading:
        """Read one 30-second sleep epoch."""
        if not self._initialized:
            raise SensorReadError("Sleep tracker not initialized", sensor_id=self._SENSOR_ID)

        # Gather simulated physiological data for the epoch
        hr = self._simulate_sleep_hr()
        hrv = self._simulate_sleep_hrv()
        movement = self._simulate_movement()
        spo2 = 96.5 + math.sin(time.time() / 300) * 1.0

        minutes_elapsed = 0.0
        if self._sleep_onset_time:
            minutes_elapsed = (time.time() - self._sleep_onset_time) / 60.0

        stage, confidence = self._classifier.classify(
            hr_bpm=hr,
            hrv_rmssd_ms=hrv,
            movement_count=movement,
            spo2_pct=spo2,
            minutes_since_sleep_onset=minutes_elapsed,
            resting_hr=self._resting_hr,
        )

        epoch = SleepEpoch(
            stage=stage, confidence=confidence,
            heart_rate_bpm=hr, movement_count=movement,
            hrv_rmssd_ms=hrv, spo2_pct=spo2,
        )

        with self._lock:
            self._epoch_counter += 1
            if stage != SleepStage.AWAKE and self._sleep_onset_time is None:
                self._sleep_onset_time = time.time()
                self._start_session()
            if self._current_session:
                self._current_session.epochs.append(epoch)

        quality = SensorReadingQuality.GOOD if confidence > 0.7 else SensorReadingQuality.FAIR
        return SensorReading(
            sensor_id=self._SENSOR_ID, sensor_type=SensorType.HEART_RATE,
            values={
                "stage": stage.value,
                "confidence": confidence,
                "heart_rate_bpm": hr,
                "movement_count": movement,
            },
            quality=quality, timestamp=epoch.timestamp, unit="stage",
        )

    async def stream(self, interval_seconds: float = 30.0) -> AsyncIterator[SensorReading]:
        while self._initialized:
            yield await self.read()
            await asyncio.sleep(interval_seconds)

    async def calibrate(self) -> SensorCalibrationResult:
        with self._lock:
            self._calibrated = True
        return SensorCalibrationResult(
            sensor_id=self._SENSOR_ID, success=True, timestamp=time.time(),
            baseline_values={"resting_hr_bpm": self._resting_hr},
        )

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(
            sensor_id=self._SENSOR_ID, sensor_type=SensorType.HEART_RATE,
            model="SleepStage-v1", sampling_rate_hz=1.0 / self.EPOCH_SECONDS,
            resolution_bits=0, is_calibrated=self._calibrated,
        )

    def get_status(self) -> str:
        if not self._initialized:
            return "off"
        return "sleeping" if self._sleep_onset_time else "awake"

    def is_healthy(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        with self._lock:
            if self._current_session:
                self._end_session()
            self._initialized = False

    def get_health_report(self) -> SensorHealthReport:
        return SensorHealthReport(
            sensor_id=self._SENSOR_ID, is_operational=self._initialized,
            last_read_timestamp=time.time() if self._initialized else None,
            calibration_valid=self._calibrated, issues=[],
        )

    def read_sync(self) -> SensorReading:
        return asyncio.get_event_loop().run_until_complete(self.read())

    def end_sleep_session(self) -> Optional[SleepSession]:
        """Call when user wakes up to finalize and return the session summary."""
        with self._lock:
            return self._end_session()

    def get_last_session(self) -> Optional[SleepSession]:
        with self._lock:
            return self._last_session

    def set_resting_hr(self, hr: float) -> None:
        with self._lock:
            self._resting_hr = max(40.0, min(100.0, hr))

    def add_alert_callback(self, cb: Callable[[CriticalHealthAlert], None]) -> None:
        self._alert_callbacks.append(cb)

    def _start_session(self) -> None:
        import uuid
        self._current_session = SleepSession(
            session_id=str(uuid.uuid4())[:8],
            start_time=time.time(),
            end_time=None,
            total_minutes=0, rem_minutes=0, light_minutes=0,
            deep_minutes=0, awake_minutes=0,
            sleep_efficiency_pct=0, sleep_latency_minutes=0,
            rem_latency_minutes=0, num_awakenings=0,
            quality=SleepQuality.FAIR, quality_score=0.0,
        )

    def _end_session(self) -> Optional[SleepSession]:
        if not self._current_session:
            return None
        session = self._current_session
        session.end_time = time.time()
        epochs = session.epochs
        epoch_min = self.EPOCH_SECONDS / 60.0

        session.rem_minutes = sum(epoch_min for e in epochs if e.stage == SleepStage.REM)
        session.light_minutes = sum(epoch_min for e in epochs if e.stage == SleepStage.LIGHT)
        session.deep_minutes = sum(epoch_min for e in epochs if e.stage == SleepStage.DEEP)
        session.awake_minutes = sum(epoch_min for e in epochs if e.stage == SleepStage.AWAKE)
        session.total_minutes = len(epochs) * epoch_min

        asleep_min = session.total_minutes - session.awake_minutes
        session.sleep_efficiency_pct = (
            asleep_min / max(session.total_minutes, 1) * 100)

        score = (
            min(100.0, session.sleep_efficiency_pct) * 0.3
            + min(100.0, session.deep_minutes / 90 * 100) * 0.3
            + min(100.0, session.rem_minutes / 120 * 100) * 0.2
            + min(100.0, asleep_min / 420 * 100) * 0.2
        )
        session.quality_score = round(score, 1)
        if score >= 85:
            session.quality = SleepQuality.EXCELLENT
        elif score >= 70:
            session.quality = SleepQuality.GOOD
        elif score >= 50:
            session.quality = SleepQuality.FAIR
        else:
            session.quality = SleepQuality.POOR

        self._last_session = session
        self._current_session = None
        self._sleep_onset_time = None
        return session

    def _simulate_sleep_hr(self) -> float:
        import random
        return max(40.0, self._resting_hr * 0.88 + math.sin(time.time() / 90) * 3 + random.gauss(0, 1))

    def _simulate_sleep_hrv(self) -> float:
        import random
        return max(10.0, 55.0 + math.sin(time.time() / 180) * 20 + random.gauss(0, 2))

    def _simulate_movement(self) -> int:
        import random
        return max(0, int(random.gauss(0.5, 1)))


# ============================================================================
# Global singleton
# ============================================================================

_sleep_tracker: Optional[SleepTracker] = None
_sleep_lock = threading.Lock()


def get_sleep_tracker() -> SleepTracker:
    global _sleep_tracker
    with _sleep_lock:
        if _sleep_tracker is None:
            _sleep_tracker = SleepTracker()
    return _sleep_tracker


# ============================================================================
# Tests
# ============================================================================

def run_sleep_tracker_tests() -> None:
    print("Testing sleep tracker...")

    async def _run():
        tracker = SleepTracker()
        assert await tracker.initialize()
        cal = await tracker.calibrate()
        assert cal.success
        reading = await tracker.read()
        assert "stage" in reading.values
        assert reading.values["stage"] in [s.value for s in SleepStage]
        await tracker.shutdown()

    asyncio.get_event_loop().run_until_complete(_run())
    print("  Sleep tracker tests passed.")


__version__ = "1.0.0"
__all__ = [
    "SleepStage", "SleepQuality",
    "SleepEpoch", "SleepSession",
    "SleepStageClassifier", "SleepTracker",
    "get_sleep_tracker",
]

if __name__ == "__main__":
    print("AI Holographic Wristwatch — Sleep Tracker")
    print("=" * 55)
    run_sleep_tracker_tests()
