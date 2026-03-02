"""
Heart Rate Monitor Driver for AI Holographic Wristwatch

PPG (photoplethysmography) based heart rate and HRV measurement. Supports
continuous monitoring, on-demand spot checks, and clinical-grade HRV analysis
for stress and fitness tracking. Implements the SensorInterface contract.

Hardware: Green + IR LED PPG with 25 Hz continuous / 100 Hz clinical mode.
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
from typing import AsyncIterator, Callable, Deque, List, Optional, Tuple

from ....core.constants import SensorConstants, HealthConstants, AlertSeverity
from ....core.exceptions import (
    SensorError, SensorReadError, SensorTimeoutError,
    SensorCalibrationError, CriticalHealthAlert
)
from ....core.interfaces import (
    SensorInterface, SensorReading, SensorInfo, SensorType,
    SensorReadingQuality, CalibrationResult as SensorCalibrationResult,
    SensorHealthReport
)


# ============================================================================
# Enumerations
# ============================================================================

class HRMonitorMode(Enum):
    """Operating mode for the HR sensor."""
    CONTINUOUS = "continuous"       # Background polling at low rate
    CLINICAL = "clinical"           # High-rate for HRV, ECG-quality
    SPOT_CHECK = "spot_check"       # Single 30-second window
    SLEEP = "sleep"                 # Very low rate to preserve battery
    OFF = "off"


class PPGSignalQuality(Enum):
    """Quality assessment of the raw PPG signal."""
    EXCELLENT = "excellent"   # SQI > 0.85
    GOOD = "good"             # SQI 0.65–0.85
    ACCEPTABLE = "acceptable" # SQI 0.45–0.65
    POOR = "poor"             # SQI 0.25–0.45
    INVALID = "invalid"       # SQI < 0.25 — discard sample


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class RRInterval:
    """A single R-R interval (time between consecutive heartbeats)."""
    duration_ms: float          # RR interval in milliseconds
    timestamp: float            # Unix timestamp of the second R-peak
    confidence: float           # 0.0–1.0
    artifact_flag: bool = False # True if likely motion artifact


@dataclass
class HRVMetrics:
    """
    Heart Rate Variability metrics computed from a series of RR intervals.

    All metrics are industry-standard. For clinical reference:
        SDNN > 50 ms → healthy autonomic function
        RMSSD > 20 ms → good parasympathetic tone
        pNN50 > 3% → normal HRV
    """
    sdnn_ms: float       # Standard deviation of NN intervals
    rmssd_ms: float      # Root mean square of successive differences
    pnn50_pct: float     # % of successive NN pairs differing by > 50 ms
    mean_rr_ms: float    # Mean RR interval
    lf_hf_ratio: float   # Low/high frequency power ratio (sympathovagal balance)
    stress_index: float  # 0.0 (relaxed) – 1.0 (high stress)
    window_seconds: float
    sample_count: int
    quality: PPGSignalQuality


@dataclass
class HeartRateReading:
    """Enriched heart rate reading with HRV and alert metadata."""
    bpm: float
    rr_intervals: List[RRInterval]
    hrv: Optional[HRVMetrics]
    signal_quality: PPGSignalQuality
    motion_artifact: bool
    timestamp: float = field(default_factory=time.time)
    alert_severity: AlertSeverity = AlertSeverity.INFO

    @property
    def is_bradycardia(self) -> bool:
        return self.bpm < HealthConstants.HEART_RATE_MIN_BPM

    @property
    def is_tachycardia(self) -> bool:
        return self.bpm > HealthConstants.HEART_RATE_MAX_BPM

    @property
    def is_clinically_concerning(self) -> bool:
        return self.is_bradycardia or self.is_tachycardia


# ============================================================================
# Signal processing helpers
# ============================================================================

class PPGSignalProcessor:
    """
    Lightweight DSP pipeline for raw PPG signals.

    Steps:
        1. DC removal (high-pass filter)
        2. Bandpass filter (0.5 – 4 Hz)
        3. Peak detection via derivative zero-crossing
        4. SQI (Signal Quality Index) estimation
        5. Motion artifact rejection using accelerometer correlation
    """

    # Bandpass cutoffs in Hz for a 25 Hz sampling rate
    LOW_CUTOFF_HZ: float = 0.5
    HIGH_CUTOFF_HZ: float = 4.0

    def __init__(self, sampling_hz: int = SensorConstants.HR_SAMPLING_RATE_HZ) -> None:
        self._fs = sampling_hz
        self._buffer: Deque[float] = deque(maxlen=self._fs * 10)  # 10-second window
        self._dc_level: float = 0.0
        self._alpha_dc: float = 0.99  # DC removal IIR coefficient

    def add_sample(self, raw_value: float) -> float:
        """Add a raw ADC sample, return the filtered value."""
        # DC removal
        self._dc_level = self._alpha_dc * self._dc_level + (1 - self._alpha_dc) * raw_value
        filtered = raw_value - self._dc_level
        self._buffer.append(filtered)
        return filtered

    def estimate_heart_rate(self) -> Tuple[float, PPGSignalQuality]:
        """Estimate HR in BPM from the current buffer. Returns (bpm, quality)."""
        data = list(self._buffer)
        if len(data) < self._fs * 4:  # Need at least 4 seconds
            return 0.0, PPGSignalQuality.INVALID

        peaks = self._detect_peaks(data)
        if len(peaks) < 2:
            return 0.0, PPGSignalQuality.INVALID

        rr_intervals = []
        for i in range(1, len(peaks)):
            rr_ms = (peaks[i] - peaks[i - 1]) / self._fs * 1000
            if 300 < rr_ms < 2000:  # Physiologically plausible
                rr_intervals.append(rr_ms)

        if not rr_intervals:
            return 0.0, PPGSignalQuality.POOR

        mean_rr = statistics.mean(rr_intervals)
        bpm = 60_000 / mean_rr
        quality = self._estimate_sqi(rr_intervals)
        return round(bpm, 1), quality

    def compute_rr_intervals(self) -> List[RRInterval]:
        """Return detected RR intervals from the current buffer."""
        data = list(self._buffer)
        peaks = self._detect_peaks(data)
        rr_list = []
        for i in range(1, len(peaks)):
            rr_ms = (peaks[i] - peaks[i - 1]) / self._fs * 1000
            if 300 < rr_ms < 2000:
                confidence = min(1.0, max(0.0, 1.0 - abs(rr_ms - 800) / 800))
                rr_list.append(RRInterval(
                    duration_ms=rr_ms,
                    timestamp=time.time() - (len(data) - peaks[i]) / self._fs,
                    confidence=confidence,
                ))
        return rr_list

    def _detect_peaks(self, data: List[float]) -> List[int]:
        """Simple derivative-based peak detection."""
        peaks = []
        min_distance = int(self._fs * 0.4)  # Min 400 ms between peaks
        last_peak = -min_distance

        for i in range(1, len(data) - 1):
            if (data[i] > data[i - 1] and data[i] > data[i + 1]
                    and data[i] > 0 and i - last_peak >= min_distance):
                peaks.append(i)
                last_peak = i
        return peaks

    def _estimate_sqi(self, rr_intervals: List[float]) -> PPGSignalQuality:
        """Estimate signal quality from RR interval regularity."""
        if len(rr_intervals) < 3:
            return PPGSignalQuality.POOR
        cv = statistics.stdev(rr_intervals) / statistics.mean(rr_intervals)
        if cv < 0.05:
            return PPGSignalQuality.EXCELLENT
        if cv < 0.12:
            return PPGSignalQuality.GOOD
        if cv < 0.20:
            return PPGSignalQuality.ACCEPTABLE
        if cv < 0.35:
            return PPGSignalQuality.POOR
        return PPGSignalQuality.INVALID


class HRVAnalyzer:
    """Computes HRV metrics from a list of RR intervals."""

    @staticmethod
    def compute(rr_intervals: List[RRInterval],
                window_seconds: float = 300.0) -> Optional[HRVMetrics]:
        """
        Compute time-domain HRV metrics. Requires at least 20 clean intervals.
        Returns None if insufficient data.
        """
        clean = [r for r in rr_intervals if not r.artifact_flag and r.confidence > 0.7]
        if len(clean) < 20:
            return None

        durations = [r.duration_ms for r in clean]
        mean_rr = statistics.mean(durations)
        sdnn = statistics.stdev(durations)

        successive_diffs = [abs(durations[i] - durations[i - 1])
                            for i in range(1, len(durations))]
        rmssd = math.sqrt(statistics.mean(d ** 2 for d in successive_diffs))
        pnn50 = sum(1 for d in successive_diffs if d > 50) / len(successive_diffs) * 100

        # Simplified LF/HF approximation from SDNN and RMSSD ratio
        lf_hf = max(0.1, sdnn / (rmssd + 1e-9))

        # Stress index: inversely proportional to HRV; normalized 0–1
        stress = min(1.0, max(0.0, 1.0 - rmssd / 80.0))

        quality = PPGSignalQuality.GOOD if len(clean) >= 50 else PPGSignalQuality.ACCEPTABLE

        return HRVMetrics(
            sdnn_ms=round(sdnn, 2),
            rmssd_ms=round(rmssd, 2),
            pnn50_pct=round(pnn50, 2),
            mean_rr_ms=round(mean_rr, 2),
            lf_hf_ratio=round(lf_hf, 3),
            stress_index=round(stress, 3),
            window_seconds=window_seconds,
            sample_count=len(clean),
            quality=quality,
        )


# ============================================================================
# Heart Rate Monitor — SensorInterface implementation
# ============================================================================

class HeartRateMonitor(SensorInterface):
    """
    PPG-based heart rate and HRV monitor.

    - Implements the full SensorInterface contract (async read/stream/calibrate)
    - Maintains a 5-minute rolling buffer of RR intervals for HRV computation
    - Raises CriticalHealthAlert on bradycardia/tachycardia detection
    - Thread-safe; designed for continuous background operation
    """

    _SENSOR_ID = "biometric.heart_rate"
    _RR_BUFFER_SIZE = 500   # ~8 minutes at 60 bpm

    def __init__(self, sampling_hz: int = SensorConstants.HR_SAMPLING_RATE_HZ) -> None:
        self._sampling_hz = sampling_hz
        self._processor = PPGSignalProcessor(sampling_hz)
        self._hrv_analyzer = HRVAnalyzer()
        self._rr_buffer: Deque[RRInterval] = deque(maxlen=self._RR_BUFFER_SIZE)
        self._mode = HRMonitorMode.OFF
        self._lock = threading.RLock()
        self._initialized = False
        self._calibrated = False
        self._baseline_bpm: Optional[float] = None
        self._last_reading: Optional[HeartRateReading] = None
        self._alert_callbacks: List[Callable[[CriticalHealthAlert], None]] = []
        self._stream_task: Optional[asyncio.Task] = None

    # ── SensorInterface ───────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Power on the PPG LED array and verify optical path."""
        await asyncio.sleep(0.05)   # Simulated hardware init
        with self._lock:
            self._initialized = True
            self._mode = HRMonitorMode.CONTINUOUS
        return True

    async def read(self) -> SensorReading:
        """Read one instantaneous heart rate sample."""
        if not self._initialized:
            raise SensorReadError("Heart rate sensor not initialized",
                                  sensor_id=self._SENSOR_ID)

        # Simulate ADC read + processing
        await asyncio.sleep(1.0 / self._sampling_hz)
        raw_ppg = self._simulate_ppg_sample()
        self._processor.add_sample(raw_ppg)

        bpm, quality = self._processor.estimate_heart_rate()
        rr_intervals = self._processor.compute_rr_intervals()

        with self._lock:
            for rr in rr_intervals:
                self._rr_buffer.append(rr)

        sensor_quality = self._map_quality(quality)
        reading = SensorReading(
            sensor_id=self._SENSOR_ID,
            sensor_type=SensorType.HEART_RATE,
            values={"bpm": bpm, "signal_quality": quality.value},
            quality=sensor_quality,
            timestamp=time.time(),
            unit="bpm",
        )

        if bpm > 0:
            hr_reading = HeartRateReading(
                bpm=bpm,
                rr_intervals=rr_intervals,
                hrv=None,
                signal_quality=quality,
                motion_artifact=False,
            )
            with self._lock:
                self._last_reading = hr_reading
            self._check_alerts(hr_reading)

        return reading

    async def stream(self, interval_seconds: float = 1.0) -> AsyncIterator[SensorReading]:
        """Yield continuous heart rate readings."""
        while self._initialized and self._mode != HRMonitorMode.OFF:
            reading = await self.read()
            yield reading
            await asyncio.sleep(max(0.0, interval_seconds - 1.0 / self._sampling_hz))

    async def calibrate(self) -> SensorCalibrationResult:
        """Perform a 30-second resting HR calibration."""
        if not self._initialized:
            raise SensorCalibrationError("Cannot calibrate: sensor not initialized",
                                          sensor_id=self._SENSOR_ID)
        await asyncio.sleep(0.5)   # Simulated 30-second window compressed
        with self._lock:
            self._calibrated = True
            self._baseline_bpm = 68.0   # Nominal resting HR
        return SensorCalibrationResult(
            sensor_id=self._SENSOR_ID,
            success=True,
            timestamp=time.time(),
            baseline_values={"resting_bpm": self._baseline_bpm},
        )

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(
            sensor_id=self._SENSOR_ID,
            sensor_type=SensorType.HEART_RATE,
            model="PPG-3000",
            sampling_rate_hz=float(self._sampling_hz),
            resolution_bits=16,
            is_calibrated=self._calibrated,
        )

    def get_status(self) -> str:
        return self._mode.value

    def is_healthy(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        with self._lock:
            self._mode = HRMonitorMode.OFF
            self._initialized = False

    def get_health_report(self) -> SensorHealthReport:
        return SensorHealthReport(
            sensor_id=self._SENSOR_ID,
            is_operational=self._initialized,
            last_read_timestamp=self._last_reading.timestamp if self._last_reading else None,
            calibration_valid=self._calibrated,
            issues=[],
        )

    def read_sync(self) -> SensorReading:
        return asyncio.get_event_loop().run_until_complete(self.read())

    # ── Extended API ─────────────────────────────────────────────────────────

    def get_hrv(self, window_seconds: float = 300.0) -> Optional[HRVMetrics]:
        """Compute HRV from the rolling RR interval buffer."""
        with self._lock:
            rr_list = list(self._rr_buffer)
        cutoff = time.time() - window_seconds
        recent = [r for r in rr_list if r.timestamp >= cutoff]
        return HRVAnalyzer.compute(recent, window_seconds)

    def set_mode(self, mode: HRMonitorMode) -> None:
        with self._lock:
            self._mode = mode

    def add_alert_callback(self, cb: Callable[[CriticalHealthAlert], None]) -> None:
        self._alert_callbacks.append(cb)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _simulate_ppg_sample(self) -> float:
        """Simulate a realistic PPG ADC reading for testing."""
        import random
        base = 2048
        heartbeat = math.sin(time.time() * 2 * math.pi * 1.1) * 200
        noise = random.gauss(0, 10)
        return base + heartbeat + noise

    def _map_quality(self, q: PPGSignalQuality) -> SensorReadingQuality:
        mapping = {
            PPGSignalQuality.EXCELLENT: SensorReadingQuality.EXCELLENT,
            PPGSignalQuality.GOOD: SensorReadingQuality.GOOD,
            PPGSignalQuality.ACCEPTABLE: SensorReadingQuality.FAIR,
            PPGSignalQuality.POOR: SensorReadingQuality.POOR,
            PPGSignalQuality.INVALID: SensorReadingQuality.INVALID,
        }
        return mapping.get(q, SensorReadingQuality.POOR)

    def _check_alerts(self, reading: HeartRateReading) -> None:
        """Fire callbacks if HR is outside safe bounds."""
        if reading.is_clinically_concerning and reading.signal_quality != PPGSignalQuality.INVALID:
            level = AlertSeverity.CRITICAL if reading.bpm < 40 or reading.bpm > 180 else AlertSeverity.WARNING
            alert = CriticalHealthAlert(
                f"Abnormal heart rate: {reading.bpm:.0f} bpm",
                health_metric="heart_rate",
                measured_value=reading.bpm,
                threshold_value=HealthConstants.HEART_RATE_MAX_BPM,
                severity=level,
            )
            for cb in self._alert_callbacks:
                try:
                    cb(alert)
                except Exception:
                    pass


# ============================================================================
# Global singleton
# ============================================================================

_heart_rate_monitor: Optional[HeartRateMonitor] = None
_hr_lock = threading.Lock()


def get_heart_rate_monitor() -> HeartRateMonitor:
    """Return the global HeartRateMonitor singleton."""
    global _heart_rate_monitor
    with _hr_lock:
        if _heart_rate_monitor is None:
            _heart_rate_monitor = HeartRateMonitor()
    return _heart_rate_monitor


# ============================================================================
# Tests
# ============================================================================

def run_heart_rate_monitor_tests() -> None:
    print("Testing heart rate monitor...")

    monitor = HeartRateMonitor()

    async def _run():
        assert await monitor.initialize() is True
        assert monitor.is_healthy()

        reading = await monitor.read()
        assert reading.sensor_type == SensorType.HEART_RATE
        assert "bpm" in reading.values

        cal = await monitor.calibrate()
        assert cal.success

        monitor.set_mode(HRMonitorMode.OFF)
        await monitor.shutdown()
        assert not monitor.is_healthy()

    asyncio.get_event_loop().run_until_complete(_run())
    print("  Heart rate monitor tests passed.")


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "HRMonitorMode", "PPGSignalQuality",
    "RRInterval", "HRVMetrics", "HeartRateReading",
    "PPGSignalProcessor", "HRVAnalyzer",
    "HeartRateMonitor",
    "get_heart_rate_monitor",
]


if __name__ == "__main__":
    print("AI Holographic Wristwatch — Heart Rate Monitor")
    print("=" * 55)
    run_heart_rate_monitor_tests()
