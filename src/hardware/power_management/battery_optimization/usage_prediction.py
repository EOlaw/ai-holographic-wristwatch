"""Usage pattern prediction for proactive battery management.

Uses a sliding-window exponential moving average to predict future power
consumption and estimates time-to-empty / time-to-full under current load.
"""
from __future__ import annotations

import threading
import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class UsageContext(Enum):
    IDLE = auto()
    LIGHT_USE = auto()    # Glances, notifications
    MODERATE_USE = auto() # Navigation, fitness tracking
    HEAVY_USE = auto()    # Hologram, video, calls
    CHARGING = auto()


@dataclass
class UsagePrediction:
    context: UsageContext = UsageContext.IDLE
    current_power_mw: float = 0.0
    avg_power_1min_mw: float = 0.0
    avg_power_5min_mw: float = 0.0
    predicted_power_mw: float = 0.0
    time_to_empty_min: float = 0.0
    time_to_full_min: float = 0.0
    soc_pct: float = 80.0
    battery_capacity_mah: float = 300.0
    confidence_pct: float = 75.0
    timestamp: float = field(default_factory=time.time)


class UsagePredictor:
    """Sliding-window power consumption predictor with EMA smoothing."""

    BATTERY_CAPACITY_MAH = 300.0
    BATTERY_VOLTAGE_V = 3.7
    WINDOW_SHORT = 60      # 1-minute window (samples at 1 Hz)
    WINDOW_LONG = 300      # 5-minute window

    # Typical power by context (mW)
    CONTEXT_POWER_MW: dict[UsageContext, float] = {
        UsageContext.IDLE:         8.0,
        UsageContext.LIGHT_USE:   45.0,
        UsageContext.MODERATE_USE: 130.0,
        UsageContext.HEAVY_USE:   380.0,
        UsageContext.CHARGING:   -800.0,  # Negative = net energy gain
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._history: deque[float] = deque(maxlen=self.WINDOW_LONG)
        self._soc_pct = 80.0
        self._context = UsageContext.IDLE
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._prediction = UsagePrediction()
        logger.info("UsagePredictor initialised")

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._predict_loop, daemon=True, name="usage-predict")
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def update_context(self, context: UsageContext) -> None:
        with self._lock:
            self._context = context

    def update_soc(self, soc_pct: float) -> None:
        with self._lock:
            self._soc_pct = max(0.0, min(100.0, soc_pct))

    def _simulate_power_reading(self) -> float:
        base = self.CONTEXT_POWER_MW.get(self._context, 50.0)
        return max(0.0, base + random.gauss(0, base * 0.1))

    def _ema(self, values: list[float], alpha: float = 0.2) -> float:
        if not values:
            return 0.0
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    def _time_to_empty(self, avg_power_mw: float) -> float:
        if avg_power_mw <= 0:
            return float("inf")
        capacity_mwh = self._soc_pct / 100.0 * self.BATTERY_CAPACITY_MAH * self.BATTERY_VOLTAGE_V
        return (capacity_mwh / avg_power_mw) * 60.0  # minutes

    def _time_to_full(self, avg_power_mw: float) -> float:
        """Estimate time to full during charging (avg_power_mw is negative)."""
        if avg_power_mw >= 0:
            return float("inf")
        remaining_mwh = (1.0 - self._soc_pct / 100.0) * self.BATTERY_CAPACITY_MAH * self.BATTERY_VOLTAGE_V
        return (remaining_mwh / abs(avg_power_mw)) * 60.0

    def _predict_loop(self) -> None:
        while self._running:
            with self._lock:
                sample = self._simulate_power_reading()
                self._history.append(sample)
                hist = list(self._history)
                avg_1m = self._ema(hist[-self.WINDOW_SHORT:] if len(hist) >= self.WINDOW_SHORT else hist)
                avg_5m = self._ema(hist)
                predicted = avg_1m * 1.05 + random.gauss(0, 5)
                tte = self._time_to_empty(avg_1m)
                ttf = self._time_to_full(avg_1m)
                conf = min(100.0, 50.0 + len(hist) / self.WINDOW_LONG * 50.0)
                self._prediction = UsagePrediction(
                    context=self._context,
                    current_power_mw=round(sample, 1),
                    avg_power_1min_mw=round(avg_1m, 1),
                    avg_power_5min_mw=round(avg_5m, 1),
                    predicted_power_mw=round(max(0, predicted), 1),
                    time_to_empty_min=round(min(tte, 9999), 1),
                    time_to_full_min=round(min(ttf, 9999), 1),
                    soc_pct=self._soc_pct,
                    battery_capacity_mah=self.BATTERY_CAPACITY_MAH,
                    confidence_pct=round(conf, 1),
                    timestamp=time.time(),
                )
            time.sleep(1.0)

    def get_prediction(self) -> UsagePrediction:
        with self._lock:
            return UsagePrediction(**vars(self._prediction))


_GLOBAL_USAGE_PREDICTOR: Optional[UsagePredictor] = None
_GLOBAL_USAGE_PREDICTOR_LOCK = threading.Lock()


def get_usage_predictor() -> UsagePredictor:
    global _GLOBAL_USAGE_PREDICTOR
    with _GLOBAL_USAGE_PREDICTOR_LOCK:
        if _GLOBAL_USAGE_PREDICTOR is None:
            _GLOBAL_USAGE_PREDICTOR = UsagePredictor()
    return _GLOBAL_USAGE_PREDICTOR


def run_usage_prediction_tests() -> bool:
    logger.info("=== UsagePrediction tests ===")
    pred = UsagePredictor()
    pred.update_context(UsageContext.MODERATE_USE)
    pred.update_soc(60.0)
    pred.start()
    time.sleep(0.15)
    p = pred.get_prediction()
    assert p.current_power_mw >= 0
    assert p.soc_pct == 60.0
    assert p.time_to_empty_min > 0
    pred.stop()
    logger.info("UsagePrediction tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_usage_prediction_tests()
