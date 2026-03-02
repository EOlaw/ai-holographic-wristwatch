"""
Stress Detector Driver for AI Holographic Wristwatch

Multi-modal physiological stress detection combining:
  - HRV (RMSSD, LF/HF ratio) from PPG
  - Galvanic Skin Response (GSR/EDA) from wrist electrodes
  - Skin temperature variance
  - Motion patterns (stillness vs fidgeting)

Outputs a normalized 0–100 stress score and autonomic state classification.
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

class StressLevel(Enum):
    """Qualitative stress classification."""
    RELAXED = "relaxed"         # 0–20
    CALM = "calm"               # 21–40
    MODERATE = "moderate"       # 41–60
    HIGH = "high"               # 61–80
    ACUTE = "acute"             # 81–100


class AutonomicState(Enum):
    """Autonomic nervous system dominance."""
    PARASYMPATHETIC = "parasympathetic"   # Rest and digest
    BALANCED = "balanced"
    SYMPATHETIC = "sympathetic"           # Fight or flight


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class GSRReading:
    """Galvanic Skin Response measurement."""
    conductance_us: float       # Skin conductance in microsiemens
    tonic_level: float          # Slow-changing baseline
    phasic_component: float     # Fast event-related changes (SCR)
    num_scr_peaks: int          # Skin Conductance Responses in window
    timestamp: float = field(default_factory=time.time)


@dataclass
class StressReading:
    """Aggregated stress measurement from multiple physiological signals."""
    stress_score: float         # 0–100 normalized stress index
    stress_level: StressLevel
    autonomic_state: AutonomicState
    hrv_contribution: float     # 0–1 weight from HRV channel
    gsr_contribution: float     # 0–1 weight from GSR channel
    temp_contribution: float    # 0–1 weight from temperature channel
    motion_contribution: float  # 0–1 weight from motion channel
    confidence: float
    gsr: Optional[GSRReading]
    recovery_time_minutes: Optional[float]   # Estimated time to return to baseline
    timestamp: float = field(default_factory=time.time)

    @property
    def is_acute(self) -> bool:
        return self.stress_level == StressLevel.ACUTE


# ============================================================================
# GSR Signal Processor
# ============================================================================

class GSRProcessor:
    """
    Processes raw GSR/EDA signal into tonic and phasic components.

    Tonic level (SCL): slow-moving skin conductance level, reflects
        chronic arousal state.
    Phasic component (SCR): fast event-locked responses, reflects
        acute stressors.
    """
    TONIC_ALPHA: float = 0.98       # IIR coefficient for slow component
    SCR_PEAK_THRESHOLD: float = 0.1 # Minimum amplitude for SCR peak (μS)

    def __init__(self) -> None:
        self._buffer: Deque[float] = deque(maxlen=200)
        self._tonic: float = 2.0

    def add_sample(self, conductance_us: float) -> None:
        self._tonic = self.TONIC_ALPHA * self._tonic + (1 - self.TONIC_ALPHA) * conductance_us
        self._buffer.append(conductance_us)

    def compute(self) -> Optional[GSRReading]:
        if len(self._buffer) < 10:
            return None
        data = list(self._buffer)
        phasic = [v - self._tonic for v in data]
        scr_peaks = sum(
            1 for i in range(1, len(phasic) - 1)
            if phasic[i] > phasic[i - 1] and phasic[i] > phasic[i + 1]
            and phasic[i] > self.SCR_PEAK_THRESHOLD
        )
        return GSRReading(
            conductance_us=round(statistics.mean(data[-10:]), 4),
            tonic_level=round(self._tonic, 4),
            phasic_component=round(max(phasic[-10:], default=0.0), 4),
            num_scr_peaks=scr_peaks,
        )


# ============================================================================
# Stress Fusion Engine
# ============================================================================

class StressFusionEngine:
    """
    Combines HRV, GSR, temperature, and motion signals into a single
    stress score using weighted evidence fusion.
    """

    WEIGHTS = {
        "hrv": 0.40,
        "gsr": 0.35,
        "temp": 0.15,
        "motion": 0.10,
    }

    def compute_score(
        self,
        hrv_rmssd_ms: Optional[float],
        gsr: Optional[GSRReading],
        temp_variance: float,
        motion_energy: float,
    ) -> float:
        """
        Returns normalized stress score 0–100.
        Higher score = more stress.
        """
        scores = {}
        confidences = {}

        # HRV component: low RMSSD = high stress
        if hrv_rmssd_ms is not None and hrv_rmssd_ms > 0:
            # RMSSD > 80 ms → very relaxed (0), < 10 ms → acute stress (100)
            hrv_score = max(0.0, min(100.0, 100.0 * (1.0 - hrv_rmssd_ms / 80.0)))
            scores["hrv"] = hrv_score
            confidences["hrv"] = 1.0

        # GSR component: more SCRs + higher conductance = more stress
        if gsr is not None:
            gsr_normalized = min(1.0, gsr.conductance_us / 20.0)
            scr_factor = min(1.0, gsr.num_scr_peaks / 10.0)
            gsr_score = (gsr_normalized * 0.5 + scr_factor * 0.5) * 100
            scores["gsr"] = gsr_score
            confidences["gsr"] = 1.0

        # Temperature variance component
        temp_score = min(100.0, temp_variance * 200.0)
        scores["temp"] = temp_score
        confidences["temp"] = 0.6

        # Motion component: fidgeting is a stress indicator
        motion_score = min(100.0, motion_energy * 50.0)
        scores["motion"] = motion_score
        confidences["motion"] = 0.5

        if not scores:
            return 50.0

        total_weight = 0.0
        weighted_sum = 0.0
        for channel, score in scores.items():
            w = self.WEIGHTS.get(channel, 0.1) * confidences.get(channel, 0.5)
            weighted_sum += score * w
            total_weight += w

        return round(weighted_sum / max(total_weight, 0.001), 1)

    @staticmethod
    def classify(score: float) -> StressLevel:
        if score <= 20:
            return StressLevel.RELAXED
        if score <= 40:
            return StressLevel.CALM
        if score <= 60:
            return StressLevel.MODERATE
        if score <= 80:
            return StressLevel.HIGH
        return StressLevel.ACUTE

    @staticmethod
    def autonomic_state(lf_hf_ratio: Optional[float]) -> AutonomicState:
        if lf_hf_ratio is None:
            return AutonomicState.BALANCED
        if lf_hf_ratio < 1.5:
            return AutonomicState.PARASYMPATHETIC
        if lf_hf_ratio > 3.0:
            return AutonomicState.SYMPATHETIC
        return AutonomicState.BALANCED


# ============================================================================
# StressDetector — SensorInterface implementation
# ============================================================================

class StressDetector(SensorInterface):
    """Multi-modal physiological stress detector."""

    _SENSOR_ID = "biometric.stress"

    def __init__(self) -> None:
        self._gsr_processor = GSRProcessor()
        self._fusion = StressFusionEngine()
        self._initialized = False
        self._calibrated = False
        self._lock = threading.RLock()
        self._last_reading: Optional[StressReading] = None
        self._baseline_gsr: float = 2.0
        self._alert_callbacks: List[Callable[[CriticalHealthAlert], None]] = []
        self._history: Deque[StressReading] = deque(maxlen=360)  # 1 hour at 10s

    async def initialize(self) -> bool:
        await asyncio.sleep(0.02)
        with self._lock:
            self._initialized = True
        return True

    async def read(self) -> SensorReading:
        if not self._initialized:
            raise SensorReadError("Stress sensor not initialized", sensor_id=self._SENSOR_ID)

        # Gather GSR samples
        for _ in range(20):
            gsr_raw = self._simulate_gsr()
            self._gsr_processor.add_sample(gsr_raw)

        gsr = self._gsr_processor.compute()

        # Simulated inputs from other sensors (would be real sensor reads in production)
        hrv_rmssd = self._simulate_hrv_rmssd()
        temp_variance = abs(self._simulate_temp_noise())
        motion_energy = self._simulate_motion_energy()
        lf_hf_ratio = 2.0 + math.sin(time.time() / 60) * 0.5

        score = self._fusion.compute_score(hrv_rmssd, gsr, temp_variance, motion_energy)
        level = StressFusionEngine.classify(score)
        autonomic = StressFusionEngine.autonomic_state(lf_hf_ratio)

        # Estimate recovery time
        recovery_min = max(0.0, (score - 20) * 0.5) if score > 20 else None

        reading = StressReading(
            stress_score=score,
            stress_level=level,
            autonomic_state=autonomic,
            hrv_contribution=0.40,
            gsr_contribution=0.35 if gsr else 0.0,
            temp_contribution=0.15,
            motion_contribution=0.10,
            confidence=0.85,
            gsr=gsr,
            recovery_time_minutes=recovery_min,
        )

        with self._lock:
            self._last_reading = reading
            self._history.append(reading)

        self._check_alerts(reading)

        quality = SensorReadingQuality.GOOD if reading.confidence > 0.7 else SensorReadingQuality.FAIR
        return SensorReading(
            sensor_id=self._SENSOR_ID, sensor_type=SensorType.GSR,
            values={
                "stress_score": reading.stress_score,
                "stress_level": reading.stress_level.value,
                "autonomic_state": reading.autonomic_state.value,
                "gsr_conductance_us": gsr.conductance_us if gsr else None,
            },
            quality=quality, timestamp=reading.timestamp, unit="score",
        )

    async def stream(self, interval_seconds: float = 10.0) -> AsyncIterator[SensorReading]:
        while self._initialized:
            yield await self.read()
            await asyncio.sleep(interval_seconds)

    async def calibrate(self) -> SensorCalibrationResult:
        for _ in range(50):
            self._gsr_processor.add_sample(self._simulate_gsr())
        gsr = self._gsr_processor.compute()
        baseline = gsr.conductance_us if gsr else 2.0
        with self._lock:
            self._calibrated = True
            self._baseline_gsr = baseline
        return SensorCalibrationResult(
            sensor_id=self._SENSOR_ID, success=True, timestamp=time.time(),
            baseline_values={"baseline_gsr_us": baseline},
        )

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(
            sensor_id=self._SENSOR_ID, sensor_type=SensorType.GSR,
            model="GSR-MULTI-01", sampling_rate_hz=0.1,
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

    def get_stress_trend(self, minutes: int = 60) -> List[StressReading]:
        cutoff = time.time() - minutes * 60
        with self._lock:
            return [r for r in self._history if r.timestamp >= cutoff]

    def add_alert_callback(self, cb: Callable[[CriticalHealthAlert], None]) -> None:
        self._alert_callbacks.append(cb)

    def _simulate_gsr(self) -> float:
        import random
        return max(0.1, 2.0 + math.sin(time.time() / 30) * 1.0 + random.gauss(0, 0.1))

    def _simulate_hrv_rmssd(self) -> float:
        import random
        return max(5.0, 40.0 + math.sin(time.time() / 120) * 15 + random.gauss(0, 3))

    def _simulate_temp_noise(self) -> float:
        import random
        return random.gauss(0, 0.05)

    def _simulate_motion_energy(self) -> float:
        import random
        return max(0.0, 0.1 + random.gauss(0, 0.05))

    def _check_alerts(self, reading: StressReading) -> None:
        if reading.is_acute and reading.confidence > 0.7:
            alert = CriticalHealthAlert(
                f"Acute stress detected: score {reading.stress_score:.0f}/100",
                health_metric="stress",
                measured_value=reading.stress_score,
                threshold_value=80.0,
                severity=AlertSeverity.WARNING,
            )
            for cb in self._alert_callbacks:
                try:
                    cb(alert)
                except Exception:
                    pass


# ============================================================================
# Global singleton
# ============================================================================

_stress_detector: Optional[StressDetector] = None
_stress_lock = threading.Lock()


def get_stress_detector() -> StressDetector:
    global _stress_detector
    with _stress_lock:
        if _stress_detector is None:
            _stress_detector = StressDetector()
    return _stress_detector


# ============================================================================
# Tests
# ============================================================================

def run_stress_detector_tests() -> None:
    print("Testing stress detector...")

    async def _run():
        det = StressDetector()
        assert await det.initialize()
        cal = await det.calibrate()
        assert cal.success
        reading = await det.read()
        assert reading.sensor_type == SensorType.GSR
        assert "stress_score" in reading.values
        assert 0 <= reading.values["stress_score"] <= 100
        await det.shutdown()

    asyncio.get_event_loop().run_until_complete(_run())
    print("  Stress detector tests passed.")


__version__ = "1.0.0"
__all__ = [
    "StressLevel", "AutonomicState",
    "GSRReading", "StressReading",
    "GSRProcessor", "StressFusionEngine",
    "StressDetector", "get_stress_detector",
]

if __name__ == "__main__":
    print("AI Holographic Wristwatch — Stress Detector")
    print("=" * 55)
    run_stress_detector_tests()
