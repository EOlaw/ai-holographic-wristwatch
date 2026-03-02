"""Multi-day health trend analysis using linear regression on biometric time series."""
from __future__ import annotations

import threading
import time
import random
import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Deque, List, Optional, Tuple
from collections import deque

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TrendPoint:
    """Single data point for trend analysis."""
    timestamp: float
    value: float
    metric: str


@dataclass
class TrendResult:
    """Result of trend analysis for a single metric."""
    metric: str
    direction: TrendDirection
    slope: float
    r_squared: float
    mean_value: float
    std_value: float
    data_points: int
    period_hours: float
    description: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "slope": round(self.slope, 6),
            "r_squared": round(self.r_squared, 4),
            "mean_value": round(self.mean_value, 3),
            "std_value": round(self.std_value, 3),
            "data_points": self.data_points,
            "period_hours": round(self.period_hours, 2),
            "description": self.description,
        }


class TrendAnalyzer:
    """Performs linear regression-based trend analysis on health metrics over time."""

    IMPROVEMENT_DIRECTION: Dict[str, str] = {
        "heart_rate": "lower",
        "spo2": "higher",
        "hrv": "higher",
        "respiratory_rate": "lower",
        "temperature": "stable",
        "stress_index": "lower",
        "activity_level": "higher",
        "sleep_quality": "higher",
    }

    SLOPE_THRESHOLD = 0.001  # per second, below which = stable
    VOLATILITY_CV_THRESHOLD = 0.25  # coefficient of variation

    def __init__(self, window_hours: float = 24.0, max_points: int = 2000) -> None:
        self._lock = threading.RLock()
        self._data: Dict[str, Deque[TrendPoint]] = {}
        self._window_seconds = window_hours * 3600
        self._max_points = max_points
        self._cached_results: Dict[str, TrendResult] = {}
        logger.info("TrendAnalyzer initialised (window=%.1fh)", window_hours)

    def add_point(self, metric: str, value: float, timestamp: Optional[float] = None) -> None:
        """Add a data point for a metric."""
        with self._lock:
            if metric not in self._data:
                self._data[metric] = deque(maxlen=self._max_points)
            ts = timestamp or time.time()
            self._data[metric].append(TrendPoint(ts, value, metric))
            # Invalidate cache
            self._cached_results.pop(metric, None)

    def _linear_regression(self, xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
        """Return (slope, intercept, r_squared) for the given x/y data."""
        n = len(xs)
        if n < 2:
            return 0.0, 0.0, 0.0
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xx = sum(x * x for x in xs)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sum_xx - sum_x ** 2
        if abs(denom) < 1e-12:
            return 0.0, sum_y / n, 0.0
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        if ss_tot < 1e-12:
            return slope, intercept, 1.0
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r_squared = max(0.0, 1.0 - ss_res / ss_tot)
        return slope, intercept, r_squared

    def analyze(self, metric: str, force: bool = False) -> Optional[TrendResult]:
        """Compute trend for a metric; uses cache unless force=True."""
        with self._lock:
            if not force and metric in self._cached_results:
                return self._cached_results[metric]
            if metric not in self._data:
                return None
            now = time.time()
            cutoff = now - self._window_seconds
            points = [p for p in self._data[metric] if p.timestamp >= cutoff]
            if len(points) < 5:
                return TrendResult(
                    metric=metric,
                    direction=TrendDirection.INSUFFICIENT_DATA,
                    slope=0.0, r_squared=0.0, mean_value=0.0, std_value=0.0,
                    data_points=len(points), period_hours=0.0,
                )
            xs = [p.timestamp - points[0].timestamp for p in points]
            ys = [p.value for p in points]
            slope, _, r_sq = self._linear_regression(xs, ys)
            mean_v = sum(ys) / len(ys)
            variance = sum((v - mean_v) ** 2 for v in ys) / max(len(ys) - 1, 1)
            std_v = math.sqrt(variance)
            cv = std_v / max(abs(mean_v), 1e-6)
            period_hours = (points[-1].timestamp - points[0].timestamp) / 3600

            # Determine direction
            if cv > self.VOLATILITY_CV_THRESHOLD:
                direction = TrendDirection.VOLATILE
            elif abs(slope) < self.SLOPE_THRESHOLD:
                direction = TrendDirection.STABLE
            else:
                improve_dir = self.IMPROVEMENT_DIRECTION.get(metric, "stable")
                if improve_dir == "lower":
                    direction = TrendDirection.IMPROVING if slope < 0 else TrendDirection.DECLINING
                elif improve_dir == "higher":
                    direction = TrendDirection.IMPROVING if slope > 0 else TrendDirection.DECLINING
                else:
                    direction = TrendDirection.STABLE

            desc = (
                f"{metric}: {direction.value} over {period_hours:.1f}h "
                f"(slope={slope:.4f}/s, R²={r_sq:.3f})"
            )
            result = TrendResult(
                metric=metric, direction=direction, slope=slope,
                r_squared=r_sq, mean_value=mean_v, std_value=std_v,
                data_points=len(points), period_hours=period_hours, description=desc,
            )
            self._cached_results[metric] = result
            return result

    def analyze_all(self) -> Dict[str, TrendResult]:
        with self._lock:
            metrics = list(self._data.keys())
        return {m: self.analyze(m, force=True) for m in metrics if self.analyze(m) is not None}

    def get_declining_metrics(self) -> List[str]:
        results = self.analyze_all()
        return [m for m, r in results.items() if r.direction == TrendDirection.DECLINING]


_ANALYZER: Optional["TrendAnalyzer"] = None
_ANALYZER_LOCK = threading.Lock()


def get_trend_analyzer() -> TrendAnalyzer:
    global _ANALYZER
    with _ANALYZER_LOCK:
        if _ANALYZER is None:
            _ANALYZER = TrendAnalyzer()
        return _ANALYZER


def run_trend_analyzer_tests() -> bool:
    logger.info("Running TrendAnalyzer tests...")
    analyzer = TrendAnalyzer(window_hours=1.0)
    now = time.time()
    # Flat HR
    for i in range(30):
        analyzer.add_point("heart_rate", 72 + random.uniform(-1, 1), now - (30 - i) * 60)
    result = analyzer.analyze("heart_rate")
    assert result is not None
    assert result.direction in (TrendDirection.STABLE, TrendDirection.VOLATILE, TrendDirection.IMPROVING, TrendDirection.DECLINING)

    # Improving HRV (increasing trend)
    for i in range(30):
        analyzer.add_point("hrv", 30 + i * 0.5, now - (30 - i) * 60)
    result = analyzer.analyze("hrv")
    assert result is not None
    assert result.direction == TrendDirection.IMPROVING, f"Expected IMPROVING, got {result.direction}"

    logger.info("TrendAnalyzer tests passed")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = run_trend_analyzer_tests()
    print("Tests passed:", ok)
    ta = get_trend_analyzer()
    now = time.time()
    for i in range(50):
        ta.add_point("heart_rate", 70 + i * 0.1 + random.uniform(-1, 1), now - (50 - i) * 120)
    r = ta.analyze("heart_rate")
    print(r.to_dict() if r else "No result")
