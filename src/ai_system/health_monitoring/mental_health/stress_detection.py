"""Real-time physiological stress detection from biometric indicators."""
from __future__ import annotations

import threading
import time
import random
import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Deque, List, Optional
from collections import deque

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class StressLevel(Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class StressIndicators:
    """Biometric indicators used to compute stress score."""
    heart_rate: Optional[float] = None
    hrv: Optional[float] = None
    respiratory_rate: Optional[float] = None
    spo2: Optional[float] = None
    skin_conductance: Optional[float] = None
    activity_level: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class StressReading:
    """Computed stress assessment from biometric indicators."""
    level: StressLevel
    score: float  # 0-100
    indicators: StressIndicators
    contributing_factors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    recovery_suggestion: str = ""

    def to_dict(self) -> Dict:
        return {
            "level": self.level.value,
            "score": round(self.score, 2),
            "factors": self.contributing_factors,
            "suggestion": self.recovery_suggestion,
            "timestamp": self.timestamp,
        }


class StressDetector:
    """Detects physiological stress using multi-modal biometric fusion."""

    # Weights for each indicator's contribution to stress score
    WEIGHTS = {
        "heart_rate_elevation": 0.25,
        "hrv_suppression": 0.30,
        "respiratory_elevation": 0.20,
        "spo2_reduction": 0.15,
        "activity_mismatch": 0.10,
    }

    STRESS_THRESHOLDS = {
        StressLevel.SEVERE: 75.0,
        StressLevel.HIGH: 55.0,
        StressLevel.MODERATE: 35.0,
        StressLevel.LOW: 15.0,
        StressLevel.NONE: 0.0,
    }

    RECOVERY_SUGGESTIONS = {
        StressLevel.SEVERE: "Seek immediate rest. Consider deep breathing for 5 minutes.",
        StressLevel.HIGH: "Take a short break and practice box breathing.",
        StressLevel.MODERATE: "A brief walk or mindfulness exercise may help.",
        StressLevel.LOW: "Light stretching can help maintain calm.",
        StressLevel.NONE: "Your stress levels are optimal.",
    }

    def __init__(self, baseline_window: int = 100) -> None:
        self._lock = threading.RLock()
        self._baseline_hr: Deque[float] = deque(maxlen=baseline_window)
        self._baseline_hrv: Deque[float] = deque(maxlen=baseline_window)
        self._baseline_rr: Deque[float] = deque(maxlen=baseline_window)
        self._history: Deque[StressReading] = deque(maxlen=500)
        self._current_level = StressLevel.NONE
        logger.info("StressDetector initialised")

    def _mean(self, d: Deque[float]) -> float:
        return sum(d) / len(d) if d else 0.0

    def _update_baselines(self, indicators: StressIndicators) -> None:
        if indicators.heart_rate is not None:
            self._baseline_hr.append(indicators.heart_rate)
        if indicators.hrv is not None:
            self._baseline_hrv.append(indicators.hrv)
        if indicators.respiratory_rate is not None:
            self._baseline_rr.append(indicators.respiratory_rate)

    def _compute_score(self, indicators: StressIndicators) -> tuple[float, List[str]]:
        """Compute 0-100 stress score and contributing factors."""
        factors: List[str] = []
        component_scores: Dict[str, float] = {}

        baseline_hr = self._mean(self._baseline_hr) or 70.0
        baseline_hrv = self._mean(self._baseline_hrv) or 45.0
        baseline_rr = self._mean(self._baseline_rr) or 16.0

        # Heart rate elevation
        if indicators.heart_rate is not None:
            hr_ratio = (indicators.heart_rate - baseline_hr) / max(baseline_hr, 1)
            hr_score = min(100.0, max(0.0, hr_ratio * 100))
            if hr_score > 20:
                factors.append(f"Elevated HR ({indicators.heart_rate:.0f} bpm)")
            component_scores["heart_rate_elevation"] = hr_score

        # HRV suppression (lower HRV = higher stress)
        if indicators.hrv is not None:
            hrv_ratio = (baseline_hrv - indicators.hrv) / max(baseline_hrv, 1)
            hrv_score = min(100.0, max(0.0, hrv_ratio * 100))
            if hrv_score > 20:
                factors.append(f"Low HRV ({indicators.hrv:.0f} ms)")
            component_scores["hrv_suppression"] = hrv_score

        # Respiratory rate elevation
        if indicators.respiratory_rate is not None:
            rr_ratio = (indicators.respiratory_rate - baseline_rr) / max(baseline_rr, 1)
            rr_score = min(100.0, max(0.0, rr_ratio * 100))
            if rr_score > 20:
                factors.append(f"Rapid breathing ({indicators.respiratory_rate:.0f} br/min)")
            component_scores["respiratory_elevation"] = rr_score

        # SpO2 reduction
        if indicators.spo2 is not None:
            spo2_reduction = max(0.0, 98.0 - indicators.spo2) * 10
            spo2_score = min(100.0, spo2_reduction)
            if spo2_score > 20:
                factors.append(f"Low SpO2 ({indicators.spo2:.0f}%)")
            component_scores["spo2_reduction"] = spo2_score

        total = sum(
            self.WEIGHTS.get(k, 0.0) * v for k, v in component_scores.items()
        )
        weight_sum = sum(self.WEIGHTS[k] for k in component_scores if k in self.WEIGHTS)
        score = total / max(weight_sum, 1e-6) if weight_sum > 0 else 0.0
        return min(100.0, score), factors

    def assess(self, indicators: StressIndicators) -> StressReading:
        """Assess stress from a set of biometric indicators."""
        with self._lock:
            self._update_baselines(indicators)
            score, factors = self._compute_score(indicators)

            level = StressLevel.NONE
            for lvl in [StressLevel.SEVERE, StressLevel.HIGH, StressLevel.MODERATE, StressLevel.LOW]:
                if score >= self.STRESS_THRESHOLDS[lvl]:
                    level = lvl
                    break

            self._current_level = level
            reading = StressReading(
                level=level,
                score=score,
                indicators=indicators,
                contributing_factors=factors,
                recovery_suggestion=self.RECOVERY_SUGGESTIONS[level],
            )
            self._history.append(reading)
            if level in (StressLevel.HIGH, StressLevel.SEVERE):
                logger.warning("High stress detected: score=%.1f, level=%s", score, level.value)
            return reading

    def get_current_level(self) -> StressLevel:
        with self._lock:
            return self._current_level

    def get_recent_history(self, limit: int = 20) -> List[StressReading]:
        with self._lock:
            return list(self._history)[-limit:]

    def average_stress_score(self, window: int = 60) -> float:
        """Average stress score over the last N readings."""
        with self._lock:
            readings = list(self._history)[-window:]
            if not readings:
                return 0.0
            return sum(r.score for r in readings) / len(readings)


_STRESS_DETECTOR: Optional["StressDetector"] = None
_STRESS_DETECTOR_LOCK = threading.Lock()


def get_stress_detector() -> StressDetector:
    global _STRESS_DETECTOR
    with _STRESS_DETECTOR_LOCK:
        if _STRESS_DETECTOR is None:
            _STRESS_DETECTOR = StressDetector()
        return _STRESS_DETECTOR


def run_stress_detector_tests() -> bool:
    logger.info("Running StressDetector tests...")
    detector = StressDetector()

    # Seed baselines
    for _ in range(30):
        detector._baseline_hr.append(72.0 + random.uniform(-3, 3))
        detector._baseline_hrv.append(45.0 + random.uniform(-5, 5))
        detector._baseline_rr.append(16.0 + random.uniform(-1, 1))

    # Normal reading
    normal = StressIndicators(heart_rate=72.0, hrv=45.0, respiratory_rate=16.0, spo2=98.0)
    result = detector.assess(normal)
    assert result.level in (StressLevel.NONE, StressLevel.LOW)

    # High stress reading
    stressed = StressIndicators(heart_rate=120.0, hrv=15.0, respiratory_rate=24.0, spo2=94.0)
    result = detector.assess(stressed)
    assert result.level in (StressLevel.HIGH, StressLevel.SEVERE), f"Expected high stress, got {result.level}"
    assert len(result.contributing_factors) > 0

    logger.info("StressDetector tests passed")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = run_stress_detector_tests()
    print("Tests passed:", ok)
    sd = get_stress_detector()
    ind = StressIndicators(heart_rate=95.0, hrv=22.0, respiratory_rate=20.0, spo2=96.0)
    r = sd.assess(ind)
    print(r.to_dict())
