"""Biometric anomaly detection using Z-score and IQR-based statistical methods."""
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


class AnomalyType(Enum):
    CARDIAC = "cardiac"
    RESPIRATORY = "respiratory"
    METABOLIC = "metabolic"
    STRESS = "stress"
    TEMPERATURE = "temperature"
    BLOOD_OXYGEN = "blood_oxygen"


class AnomalySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """Represents a detected biometric anomaly."""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    metric_name: str
    observed_value: float
    expected_mean: float
    expected_std: float
    z_score: float
    timestamp: float = field(default_factory=time.time)
    description: str = ""

    def age_seconds(self) -> float:
        return time.time() - self.timestamp


@dataclass
class BaselineStats:
    """Rolling statistical baseline for a single metric."""
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    mean: float = 0.0
    std: float = 1.0
    q1: float = 0.0
    q3: float = 0.0
    n_samples: int = 0

    def update(self, value: float) -> None:
        self.values.append(value)
        self.n_samples += 1
        if len(self.values) >= 5:
            sorted_vals = sorted(self.values)
            n = len(sorted_vals)
            self.mean = sum(sorted_vals) / n
            variance = sum((v - self.mean) ** 2 for v in sorted_vals) / max(n - 1, 1)
            self.std = math.sqrt(variance) or 1e-6
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            self.q1 = sorted_vals[q1_idx]
            self.q3 = sorted_vals[min(q3_idx, n - 1)]

    def z_score(self, value: float) -> float:
        return (value - self.mean) / max(self.std, 1e-6)

    def iqr(self) -> float:
        return self.q3 - self.q1

    def is_iqr_outlier(self, value: float, factor: float = 1.5) -> bool:
        iqr = self.iqr()
        return value < (self.q1 - factor * iqr) or value > (self.q3 + factor * iqr)


class AnomalyDetector:
    """Detects biometric anomalies using rolling Z-score and IQR methods."""

    # Mapping from metric name to anomaly type
    METRIC_TYPE_MAP: Dict[str, AnomalyType] = {
        "heart_rate": AnomalyType.CARDIAC,
        "hrv": AnomalyType.CARDIAC,
        "respiratory_rate": AnomalyType.RESPIRATORY,
        "spo2": AnomalyType.BLOOD_OXYGEN,
        "temperature": AnomalyType.TEMPERATURE,
        "activity_level": AnomalyType.METABOLIC,
        "stress_index": AnomalyType.STRESS,
    }

    Z_THRESHOLDS = {
        AnomalySeverity.LOW: 2.0,
        AnomalySeverity.MEDIUM: 2.5,
        AnomalySeverity.HIGH: 3.0,
        AnomalySeverity.CRITICAL: 4.0,
    }

    MIN_SAMPLES_FOR_DETECTION = 20

    def __init__(self, z_threshold: float = 2.5, use_iqr: bool = True) -> None:
        self._lock = threading.RLock()
        self._baselines: Dict[str, BaselineStats] = {}
        self._z_threshold = z_threshold
        self._use_iqr = use_iqr
        self._recent_anomalies: Deque[AnomalyEvent] = deque(maxlen=100)
        logger.info("AnomalyDetector initialised (z_threshold=%.1f, iqr=%s)", z_threshold, use_iqr)

    def _get_or_create_baseline(self, metric: str) -> BaselineStats:
        if metric not in self._baselines:
            self._baselines[metric] = BaselineStats()
        return self._baselines[metric]

    def update_baseline(self, metric: str, value: float) -> None:
        """Feed a new value into the rolling baseline for a metric."""
        with self._lock:
            baseline = self._get_or_create_baseline(metric)
            baseline.update(value)

    def detect(self, metric: str, value: float) -> Optional[AnomalyEvent]:
        """Check a metric value against its baseline; return AnomalyEvent if anomalous."""
        with self._lock:
            baseline = self._get_or_create_baseline(metric)
            baseline.update(value)
            if baseline.n_samples < self.MIN_SAMPLES_FOR_DETECTION:
                return None

            z = abs(baseline.z_score(value))
            iqr_flag = self._use_iqr and baseline.is_iqr_outlier(value, factor=1.5)

            # Determine severity
            severity = None
            for sev in [AnomalySeverity.CRITICAL, AnomalySeverity.HIGH,
                        AnomalySeverity.MEDIUM, AnomalySeverity.LOW]:
                if z >= self.Z_THRESHOLDS[sev]:
                    severity = sev
                    break

            if severity is None and not iqr_flag:
                return None

            if severity is None:
                severity = AnomalySeverity.LOW

            anomaly_type = self.METRIC_TYPE_MAP.get(metric, AnomalyType.METABOLIC)
            event = AnomalyEvent(
                anomaly_type=anomaly_type,
                severity=severity,
                metric_name=metric,
                observed_value=value,
                expected_mean=baseline.mean,
                expected_std=baseline.std,
                z_score=z,
                description=(
                    f"{metric} value {value:.2f} deviates {z:.2f} std from "
                    f"mean {baseline.mean:.2f} (severity={severity.value})"
                ),
            )
            self._recent_anomalies.append(event)
            logger.warning("Anomaly detected: %s", event.description)
            return event

    def get_recent_anomalies(
        self, anomaly_type: Optional[AnomalyType] = None, limit: int = 20
    ) -> List[AnomalyEvent]:
        with self._lock:
            events = list(self._recent_anomalies)
            if anomaly_type:
                events = [e for e in events if e.anomaly_type == anomaly_type]
            return events[-limit:]

    def get_baseline_summary(self, metric: str) -> Dict:
        with self._lock:
            if metric not in self._baselines:
                return {}
            b = self._baselines[metric]
            return {
                "metric": metric,
                "mean": round(b.mean, 3),
                "std": round(b.std, 3),
                "q1": round(b.q1, 3),
                "q3": round(b.q3, 3),
                "n_samples": b.n_samples,
            }

    def reset_baseline(self, metric: str) -> None:
        with self._lock:
            if metric in self._baselines:
                del self._baselines[metric]
                logger.info("Baseline reset for metric: %s", metric)


_DETECTOR: Optional["AnomalyDetector"] = None
_DETECTOR_LOCK = threading.Lock()


def get_anomaly_detector() -> AnomalyDetector:
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            _DETECTOR = AnomalyDetector()
        return _DETECTOR


def run_anomaly_detector_tests() -> bool:
    logger.info("Running AnomalyDetector tests...")
    detector = AnomalyDetector(z_threshold=2.5, use_iqr=True)

    # Seed baseline with normal heart rate data
    for _ in range(50):
        detector.update_baseline("heart_rate", random.gauss(72, 5))

    # Normal value should not trigger
    result = detector.detect("heart_rate", 74.0)
    assert result is None, "False positive for normal HR"

    # Extreme outlier should trigger
    result = detector.detect("heart_rate", 180.0)
    assert result is not None, "Missed cardiac anomaly"
    assert result.anomaly_type == AnomalyType.CARDIAC
    assert result.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL)

    # Verify summary
    summary = detector.get_baseline_summary("heart_rate")
    assert "mean" in summary

    logger.info("AnomalyDetector tests passed")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = run_anomaly_detector_tests()
    print("Tests passed:", ok)
    det = get_anomaly_detector()
    for v in [70, 72, 68, 74, 71, 73, 69, 75, 72, 70] * 5:
        det.update_baseline("heart_rate", v + random.uniform(-2, 2))
    event = det.detect("heart_rate", 160.0)
    print("Anomaly:", event)
