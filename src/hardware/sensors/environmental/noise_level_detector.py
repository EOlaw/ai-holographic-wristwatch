"""
Noise Level Detector — AI Holographic Wristwatch

Measures ambient sound pressure level for:
- Hearing health protection (WHO noise exposure limits)
- Environment classification (quiet/moderate/loud)
- Smart notification volume adjustment
- Meeting/focus mode detection
- Noise dose accumulation (OSHA 8-hour TWA)
- SensorInterface compliance
"""

from __future__ import annotations

import math
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


class NoiseEnvironment(Enum):
    SILENT         = "silent"          # < 30 dB
    QUIET          = "quiet"           # 30–45 dB (library)
    MODERATE       = "moderate"        # 45–60 dB (office)
    CONVERSATIONAL = "conversational"  # 60–70 dB (restaurant)
    LOUD           = "loud"            # 70–80 dB (busy street)
    VERY_LOUD      = "very_loud"       # 80–90 dB (traffic)
    DANGEROUS      = "dangerous"       # 90–110 dB (concert)
    HAZARDOUS      = "hazardous"       # > 110 dB (power tools)


class HearingRiskLevel(Enum):
    NONE     = "none"
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass
class NoiseLevelReading(SensorReading):
    spl_db: float = 0.0                    # instantaneous SPL dBSPL
    spl_db_a: float = 0.0                  # A-weighted (hearing-relevant)
    leq_1min_db: float = 0.0               # equivalent level 1-minute
    peak_db: float = 0.0                   # peak in last minute
    noise_dose_pct: float = 0.0            # OSHA 8-hr TWA dose %
    environment: NoiseEnvironment = NoiseEnvironment.QUIET
    hearing_risk: HearingRiskLevel = HearingRiskLevel.NONE
    recommended_volume_pct: float = 50.0   # suggested headphone/speaker volume
    speech_intelligibility: float = 1.0   # 0–1, how well speech can be heard


def _db_to_environment(db: float) -> NoiseEnvironment:
    if db < 30:  return NoiseEnvironment.SILENT
    if db < 45:  return NoiseEnvironment.QUIET
    if db < 60:  return NoiseEnvironment.MODERATE
    if db < 70:  return NoiseEnvironment.CONVERSATIONAL
    if db < 80:  return NoiseEnvironment.LOUD
    if db < 90:  return NoiseEnvironment.VERY_LOUD
    if db < 110: return NoiseEnvironment.DANGEROUS
    return NoiseEnvironment.HAZARDOUS


def _db_to_hearing_risk(db_a: float) -> HearingRiskLevel:
    if db_a < 70:  return HearingRiskLevel.NONE
    if db_a < 80:  return HearingRiskLevel.LOW
    if db_a < 90:  return HearingRiskLevel.MODERATE
    if db_a < 100: return HearingRiskLevel.HIGH
    return HearingRiskLevel.CRITICAL


def _compute_noise_dose(history_db_a: List[float], duration_hr: float = 8.0) -> float:
    """OSHA noise dose: D = 100 * Σ(Ci/Ti), where Ti = 8h / 2^((Li-90)/5)."""
    if not history_db_a:
        return 0.0
    total_dose = 0.0
    for db_a in history_db_a:
        if db_a >= 80:  # OSHA action level
            allowed_hr = 8.0 / (2.0 ** ((db_a - 90.0) / 5.0))
            total_dose += (1.0 / 3600) / max(0.001, allowed_hr)  # 1-second samples
    return min(200.0, total_dose * 100.0)


_GLOBAL_NOISE: Optional["NoiseLevelDetector"] = None
_GLOBAL_NOISE_LOCK = threading.Lock()


class NoiseLevelDetector(SensorInterface):
    """Ambient noise level detector using microphone array SPL measurement."""

    SENSOR_ID    = "environmental.noise_level"
    SENSOR_TYPE  = "noise_level"
    MODEL        = "SPL-Monitor-v1"
    MANUFACTURER = "AI Holographic"

    def __init__(self) -> None:
        self._history_db_a: Deque[float] = deque(maxlen=3600)  # 1-hr at 1Hz
        self._leq_buffer: Deque[float] = deque(maxlen=60)       # 1-min Leq
        self._lock = threading.RLock()
        self._running = self._initialized = False
        self._error_count = self._read_count = 0
        self._last_reading: Optional[NoiseLevelReading] = None

    def initialize(self) -> bool:
        with self._lock:
            self._initialized = self._running = True
            logger.info("NoiseLevelDetector initialized")
            return True

    def read(self) -> Optional[NoiseLevelReading]:
        if not self._initialized:
            return None
        with self._lock:
            # Simulate typical office environment with variation
            base_db = random.gauss(52.0, 8.0)
            spl     = max(20.0, base_db)
            # A-weighting correction (simplified: +1 dB for typical office noise)
            spl_a   = spl + random.gauss(1.0, 0.5)
            spl_a   = max(20.0, spl_a)

            self._history_db_a.append(spl_a)
            self._leq_buffer.append(spl_a)

            # Leq (energy average of last 60 samples)
            leq = 10 * math.log10(sum(10**(v/10) for v in self._leq_buffer) / len(self._leq_buffer))
            peak = max(self._leq_buffer)
            dose = _compute_noise_dose(list(self._history_db_a))
            env  = _db_to_environment(spl)
            risk = _db_to_hearing_risk(spl_a)

            # Speech intelligibility degrades above ~70 dB noise
            intelligibility = max(0.0, 1.0 - max(0, spl - 60) / 50.0)
            # Recommended volume: adaptive to environment noise
            vol = min(100.0, max(10.0, 50.0 + (spl - 50) * 0.5))

            reading = NoiseLevelReading(
                sensor_id=self.SENSOR_ID, timestamp=time.time(),
                spl_db=spl, spl_db_a=spl_a, leq_1min_db=leq, peak_db=peak,
                noise_dose_pct=dose, environment=env, hearing_risk=risk,
                recommended_volume_pct=vol, speech_intelligibility=intelligibility,
                confidence=0.90,
            )
            self._last_reading = reading
            self._read_count += 1
            return reading

    async def stream(self) -> AsyncIterator[NoiseLevelReading]:
        import asyncio
        while self._running:
            r = self.read()
            if r: yield r
            await asyncio.sleep(1.0)

    def calibrate(self) -> bool:
        logger.info("NoiseLevelDetector: calibrate with SPL reference meter")
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._running = self._initialized = False

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(sensor_id=self.SENSOR_ID, sensor_type=self.SENSOR_TYPE,
                          model=self.MODEL, manufacturer=self.MANUFACTURER,
                          firmware_version="1.0.0", hardware_version="software",
                          capabilities={"spl_range_db": [20, 130], "a_weighting": True,
                                        "leq": True, "noise_dose": True})

    def get_status(self) -> SensorStatus:
        if not self._initialized: return SensorStatus.UNINITIALIZED
        return SensorStatus.RUNNING if self._running else SensorStatus.IDLE

    def is_healthy(self) -> bool:
        return self.get_status() in (SensorStatus.RUNNING, SensorStatus.IDLE)

    def get_health_report(self) -> Dict:
        return {"status": self.get_status().value, "read_count": self._read_count,
                "current_spl": self._last_reading.spl_db if self._last_reading else None,
                "noise_dose_pct": self._last_reading.noise_dose_pct if self._last_reading else 0.0}

    def read_sync(self) -> Optional[NoiseLevelReading]:
        return self.read()


def get_noise_level_detector() -> NoiseLevelDetector:
    global _GLOBAL_NOISE
    with _GLOBAL_NOISE_LOCK:
        if _GLOBAL_NOISE is None:
            _GLOBAL_NOISE = NoiseLevelDetector()
        return _GLOBAL_NOISE


def run_noise_level_detector_tests() -> bool:
    nd = NoiseLevelDetector()
    assert nd.initialize()
    r = nd.read()
    assert r is not None and 20 <= r.spl_db <= 130
    nd.shutdown()
    logger.info("NoiseLevelDetector tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_noise_level_detector_tests()
