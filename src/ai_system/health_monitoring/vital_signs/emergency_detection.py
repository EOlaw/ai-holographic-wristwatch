"""Emergency health event detection for life-threatening biometric states."""
from __future__ import annotations

import threading
import time
import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Deque, List, Optional, Callable
from collections import deque

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class EmergencyLevel(Enum):
    NONE = "none"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"
    LIFE_THREATENING = "life_threatening"


class EmergencyType(Enum):
    CARDIAC_ARREST = "cardiac_arrest"
    ATRIAL_FIBRILLATION = "atrial_fibrillation"
    HYPOXIA = "hypoxia"
    HYPERTHERMIA = "hyperthermia"
    HYPOTHERMIA = "hypothermia"
    HYPERTENSION_CRISIS = "hypertension_crisis"
    RESPIRATORY_DISTRESS = "respiratory_distress"
    FALL_DETECTED = "fall_detected"


@dataclass
class EmergencyEvent:
    """A detected emergency health event."""
    event_type: EmergencyType
    level: EmergencyLevel
    description: str
    metric_values: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    alert_sent: bool = False

    def resolve(self) -> None:
        self.resolved = True
        self.resolved_at = time.time()

    def age_seconds(self) -> float:
        return time.time() - self.timestamp


@dataclass
class EmergencyThreshold:
    metric: str
    low_critical: Optional[float]
    low_warning: Optional[float]
    high_warning: Optional[float]
    high_critical: Optional[float]


class EmergencyDetector:
    """Detects life-threatening biometric conditions and escalates appropriately."""

    THRESHOLDS: List[EmergencyThreshold] = [
        EmergencyThreshold("heart_rate", 30.0, 45.0, 130.0, 180.0),
        EmergencyThreshold("spo2", 85.0, 90.0, None, None),
        EmergencyThreshold("temperature", 34.0, 35.5, 38.5, 40.0),
        EmergencyThreshold("bp_systolic", 70.0, 90.0, 160.0, 180.0),
        EmergencyThreshold("respiratory_rate", 6.0, 10.0, 25.0, 30.0),
    ]

    EMERGENCY_MAP: Dict[str, Dict[str, EmergencyType]] = {
        "heart_rate": {"high": EmergencyType.CARDIAC_ARREST, "low": EmergencyType.CARDIAC_ARREST},
        "spo2": {"low": EmergencyType.HYPOXIA},
        "temperature": {"high": EmergencyType.HYPERTHERMIA, "low": EmergencyType.HYPOTHERMIA},
        "bp_systolic": {"high": EmergencyType.HYPERTENSION_CRISIS},
        "respiratory_rate": {"high": EmergencyType.RESPIRATORY_DISTRESS, "low": EmergencyType.RESPIRATORY_DISTRESS},
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_events: Dict[str, EmergencyEvent] = {}
        self._event_history: Deque[EmergencyEvent] = deque(maxlen=200)
        self._callbacks: List[Callable[[EmergencyEvent], None]] = []
        self._consecutive_violations: Dict[str, int] = {}
        self.REQUIRED_CONSECUTIVE = 3
        logger.info("EmergencyDetector initialised")

    def register_callback(self, cb: Callable[[EmergencyEvent], None]) -> None:
        with self._lock:
            self._callbacks.append(cb)

    def _fire_callbacks(self, event: EmergencyEvent) -> None:
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:  # noqa: BLE001
                logger.error("Emergency callback error: %s", exc)

    def evaluate(self, metric: str, value: float) -> Optional[EmergencyEvent]:
        """Evaluate a metric value; return EmergencyEvent if threshold breached."""
        with self._lock:
            threshold = next((t for t in self.THRESHOLDS if t.metric == metric), None)
            if threshold is None:
                return None

            level = EmergencyLevel.NONE
            direction = None

            if threshold.low_critical is not None and value <= threshold.low_critical:
                level = EmergencyLevel.CRITICAL
                direction = "low"
            elif threshold.high_critical is not None and value >= threshold.high_critical:
                level = EmergencyLevel.CRITICAL
                direction = "high"
            elif threshold.low_warning is not None and value <= threshold.low_warning:
                level = EmergencyLevel.WARNING
                direction = "low"
            elif threshold.high_warning is not None and value >= threshold.high_warning:
                level = EmergencyLevel.WARNING
                direction = "high"

            if level == EmergencyLevel.NONE:
                self._consecutive_violations[metric] = 0
                # Resolve active event if exists
                if metric in self._active_events:
                    self._active_events[metric].resolve()
                    del self._active_events[metric]
                return None

            # Require consecutive readings
            self._consecutive_violations[metric] = self._consecutive_violations.get(metric, 0) + 1
            if self._consecutive_violations[metric] < self.REQUIRED_CONSECUTIVE:
                return None

            # Determine emergency type
            etype_map = self.EMERGENCY_MAP.get(metric, {})
            etype = etype_map.get(direction, EmergencyType.CARDIAC_ARREST)

            # Elevate to LIFE_THREATENING for critical
            if level == EmergencyLevel.CRITICAL:
                level = EmergencyLevel.LIFE_THREATENING

            event = EmergencyEvent(
                event_type=etype,
                level=level,
                description=f"EMERGENCY: {metric}={value:.1f} ({direction} threshold breached)",
                metric_values={metric: value},
            )
            self._active_events[metric] = event
            self._event_history.append(event)
            logger.critical("EMERGENCY DETECTED: %s", event.description)
            self._fire_callbacks(event)
            return event

    def get_active_emergencies(self) -> List[EmergencyEvent]:
        with self._lock:
            return list(self._active_events.values())

    def get_event_history(self, limit: int = 50) -> List[EmergencyEvent]:
        with self._lock:
            return list(self._event_history)[-limit:]

    def is_emergency_active(self) -> bool:
        with self._lock:
            return bool(self._active_events)

    def highest_level(self) -> EmergencyLevel:
        with self._lock:
            if not self._active_events:
                return EmergencyLevel.NONE
            levels = [e.level for e in self._active_events.values()]
            priority = [
                EmergencyLevel.LIFE_THREATENING, EmergencyLevel.CRITICAL,
                EmergencyLevel.WARNING, EmergencyLevel.WATCH, EmergencyLevel.NONE,
            ]
            for lvl in priority:
                if lvl in levels:
                    return lvl
            return EmergencyLevel.NONE


_EMERGENCY_DETECTOR: Optional["EmergencyDetector"] = None
_EMERGENCY_DETECTOR_LOCK = threading.Lock()


def get_emergency_detector() -> EmergencyDetector:
    global _EMERGENCY_DETECTOR
    with _EMERGENCY_DETECTOR_LOCK:
        if _EMERGENCY_DETECTOR is None:
            _EMERGENCY_DETECTOR = EmergencyDetector()
        return _EMERGENCY_DETECTOR


def run_emergency_detector_tests() -> bool:
    logger.info("Running EmergencyDetector tests...")
    detector = EmergencyDetector()
    detector.REQUIRED_CONSECUTIVE = 1

    events: List[EmergencyEvent] = []
    detector.register_callback(lambda e: events.append(e))

    # Normal values
    result = detector.evaluate("heart_rate", 72.0)
    assert result is None, "False positive for normal HR"

    # Life-threatening low SpO2
    result = detector.evaluate("spo2", 82.0)
    assert result is not None, "Missed SpO2 emergency"
    assert result.event_type == EmergencyType.HYPOXIA
    assert result.level == EmergencyLevel.LIFE_THREATENING

    # Confirm callback fired
    assert len(events) == 1

    logger.info("EmergencyDetector tests passed")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = run_emergency_detector_tests()
    print("Tests passed:", ok)
    det = get_emergency_detector()
    det.REQUIRED_CONSECUTIVE = 1
    e = det.evaluate("heart_rate", 200.0)
    print("Emergency event:", e)
    print("Active emergencies:", det.get_active_emergencies())
