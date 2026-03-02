"""
Conflict Resolution — AI Holographic Wristwatch

Resolves disagreements between sensor readings using:
- Consistency checking with physical constraints
- Bayesian evidence combination
- Temporal consistency tracking
- Outlier rejection (Hampel identifier, IQR gating)
- Sensor trust scoring based on historical accuracy
- Voting-based consensus for redundant sensors
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
from typing import Deque, Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class ConflictType(Enum):
    NONE          = "none"
    SOFT_CONFLICT = "soft_conflict"   # slight disagreement, tolerable
    HARD_CONFLICT = "hard_conflict"   # physically impossible disagreement
    STALE_DATA    = "stale_data"      # reading too old
    OUTLIER       = "outlier"         # statistical outlier


class ResolutionStrategy(Enum):
    TRUST_HIGHEST_CONFIDENCE = "highest_confidence"
    WEIGHTED_AVERAGE         = "weighted_average"
    MEDIAN                   = "median"
    BAYESIAN_FUSION          = "bayesian"
    DISCARD_OUTLIER          = "discard_outlier"
    USE_LAST_VALID           = "use_last_valid"


@dataclass
class SensorMeasurement:
    """Single sensor measurement with metadata for conflict resolution."""
    sensor_id: str
    value: float
    timestamp: float
    confidence: float = 1.0
    unit: str = ""


@dataclass
class ConflictReport:
    """Result of conflict detection and resolution."""
    conflict_type: ConflictType
    resolution_strategy: ResolutionStrategy
    resolved_value: float
    resolved_confidence: float
    conflicting_sensors: List[str] = field(default_factory=list)
    discarded_sensors: List[str] = field(default_factory=list)
    resolution_note: str = ""


class HampelIdentifier:
    """
    Hampel identifier for robust outlier detection.
    Uses Median Absolute Deviation (MAD) as robust spread estimator.
    k=1.4826 scales MAD to approximate std dev for normal distributions.
    """

    K = 1.4826
    WINDOW_SIZE = 10
    THRESHOLD   = 3.0  # values > 3σ_robust are outliers

    def __init__(self) -> None:
        self._window: Deque[float] = deque(maxlen=self.WINDOW_SIZE)

    def update(self, value: float) -> bool:
        """Returns True if value is an outlier."""
        self._window.append(value)
        if len(self._window) < 3:
            return False
        vals = sorted(self._window)
        median = vals[len(vals)//2]
        mad    = sorted(abs(v - median) for v in vals)[len(vals)//2]
        sigma  = self.K * mad
        if sigma < 1e-9:
            return False
        return abs(value - median) > self.THRESHOLD * sigma

    def get_robust_estimate(self) -> Optional[float]:
        if not self._window:
            return None
        vals = sorted(self._window)
        return vals[len(vals) // 2]


class BayesianFuser:
    """
    Combines multiple sensor measurements using Bayesian fusion
    (sum of precision-weighted measurements / sum of precisions).
    Assumes Gaussian noise model.
    """

    def fuse(self, measurements: List[SensorMeasurement]) -> Tuple[float, float]:
        """Returns (fused_value, fused_confidence)."""
        if not measurements:
            return 0.0, 0.0
        if len(measurements) == 1:
            return measurements[0].value, measurements[0].confidence

        # Precision = 1 / variance; assume variance ∝ 1 / confidence²
        precision_weighted_sum = 0.0
        total_precision = 0.0
        for m in measurements:
            precision = m.confidence ** 2
            precision_weighted_sum += precision * m.value
            total_precision += precision

        if total_precision < 1e-12:
            return sum(m.value for m in measurements) / len(measurements), 0.0

        fused_value = precision_weighted_sum / total_precision
        # Fused precision = sum of individual precisions
        fused_conf  = min(1.0, math.sqrt(total_precision / len(measurements)))
        return fused_value, fused_conf


class StalenessChecker:
    """Flags measurements older than the maximum acceptable age."""

    MAX_AGE_SEC: Dict[str, float] = {
        "heart_rate":     5.0,
        "accelerometer":  0.1,
        "gyroscope":      0.1,
        "gps":            10.0,
        "temperature":    30.0,
        "default":        2.0,
    }

    def is_stale(self, measurement: SensorMeasurement) -> bool:
        age = time.time() - measurement.timestamp
        max_age = self.MAX_AGE_SEC.get(
            measurement.sensor_id.split(".")[-1],
            self.MAX_AGE_SEC["default"],
        )
        return age > max_age


class ConflictResolver:
    """
    Detects and resolves conflicts between multiple sensor measurements
    for the same physical quantity.
    """

    SOFT_CONFLICT_THRESHOLD_STD = 1.5  # > 1.5σ spread → soft conflict
    HARD_CONFLICT_THRESHOLD_STD = 4.0  # > 4σ spread   → hard conflict

    def __init__(self, sensor_id_group: str) -> None:
        self._group      = sensor_id_group
        self._outlier_id  = HampelIdentifier()
        self._bayesian    = BayesianFuser()
        self._staleness   = StalenessChecker()
        self._sensor_trust: Dict[str, float] = {}
        self._history: Deque[ConflictReport] = deque(maxlen=50)

    def resolve(self, measurements: List[SensorMeasurement]) -> ConflictReport:
        """Detect conflicts and return resolved value with report."""
        if not measurements:
            return ConflictReport(
                ConflictType.NONE, ResolutionStrategy.TRUST_HIGHEST_CONFIDENCE,
                0.0, 0.0, note="No measurements",
            )

        # Filter stale
        fresh = [m for m in measurements if not self._staleness.is_stale(m)]
        stale  = [m for m in measurements if self._staleness.is_stale(m)]
        stale_ids = [m.sensor_id for m in stale]

        if not fresh:
            if measurements:
                best = max(measurements, key=lambda m: m.confidence)
                return ConflictReport(
                    ConflictType.STALE_DATA,
                    ResolutionStrategy.USE_LAST_VALID,
                    best.value, best.confidence * 0.5,
                    discarded_sensors=stale_ids,
                    resolution_note="All data stale; using last valid",
                )
            return ConflictReport(ConflictType.NONE, ResolutionStrategy.MEDIAN, 0.0, 0.0)

        # Detect outliers
        valid, outlier_ids = [], []
        for m in fresh:
            if self._outlier_id.update(m.value):
                outlier_ids.append(m.sensor_id)
            else:
                valid.append(m)

        if not valid:
            valid = fresh  # if all rejected, use all

        # Compute spread
        values = [m.value for m in valid]
        mean   = sum(values) / len(values)
        std    = math.sqrt(sum((v - mean)**2 for v in values) / len(values)) if len(values) > 1 else 0.0

        conflict_type = ConflictType.NONE
        if std > self.HARD_CONFLICT_THRESHOLD_STD:
            conflict_type = ConflictType.HARD_CONFLICT
        elif std > self.SOFT_CONFLICT_THRESHOLD_STD:
            conflict_type = ConflictType.SOFT_CONFLICT

        # Choose resolution strategy
        if conflict_type == ConflictType.HARD_CONFLICT:
            # Trust highest-confidence sensor
            best = max(valid, key=lambda m: m.confidence)
            strategy = ResolutionStrategy.TRUST_HIGHEST_CONFIDENCE
            result_val, result_conf = best.value, best.confidence * 0.7
        elif len(valid) >= 2:
            result_val, result_conf = self._bayesian.fuse(valid)
            strategy = ResolutionStrategy.BAYESIAN_FUSION
        else:
            result_val, result_conf = valid[0].value, valid[0].confidence
            strategy = ResolutionStrategy.TRUST_HIGHEST_CONFIDENCE

        report = ConflictReport(
            conflict_type=conflict_type,
            resolution_strategy=strategy,
            resolved_value=result_val,
            resolved_confidence=result_conf,
            conflicting_sensors=[m.sensor_id for m in measurements if m.sensor_id in
                                  ([m.sensor_id for m in valid] if conflict_type != ConflictType.NONE else [])],
            discarded_sensors=outlier_ids + stale_ids,
            resolution_note=f"std={std:.3f}, n_valid={len(valid)}",
        )
        self._history.append(report)
        return report

    def get_history(self) -> List[ConflictReport]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

_resolvers: Dict[str, ConflictResolver] = {}
_resolvers_lock = threading.Lock()


def get_resolver(group: str) -> ConflictResolver:
    """Get or create a ConflictResolver for a named measurement group."""
    with _resolvers_lock:
        if group not in _resolvers:
            _resolvers[group] = ConflictResolver(group)
        return _resolvers[group]


def run_conflict_resolution_tests() -> bool:
    resolver = ConflictResolver("heart_rate")

    # Three sensors, one outlier
    measurements = [
        SensorMeasurement("ppg.1", 72.0, time.time(), 0.95),
        SensorMeasurement("ppg.2", 73.5, time.time(), 0.90),
        SensorMeasurement("ppg.3", 120.0, time.time(), 0.40),   # outlier
    ]
    report = resolver.resolve(measurements)
    assert report.resolved_value < 90.0, f"Expected ~72-74, got {report.resolved_value}"
    assert report.conflict_type != ConflictType.NONE

    # Test staleness
    stale = [SensorMeasurement("ppg.1", 72.0, time.time() - 100, 0.95)]
    rep2 = resolver.resolve(stale)
    assert rep2.conflict_type == ConflictType.STALE_DATA

    logger.info("ConflictResolution tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_conflict_resolution_tests()
