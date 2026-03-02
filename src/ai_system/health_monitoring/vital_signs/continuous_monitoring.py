"""Continuous vital signs monitoring module for real-time biometric data aggregation."""
from __future__ import annotations

import threading
import time
import random
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Deque, List, Optional, Callable
from collections import deque

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class VitalSignType(Enum):
    HEART_RATE = "heart_rate"
    SPO2 = "spo2"
    BLOOD_PRESSURE_SYSTOLIC = "bp_systolic"
    BLOOD_PRESSURE_DIASTOLIC = "bp_diastolic"
    TEMPERATURE = "temperature"
    RESPIRATORY_RATE = "respiratory_rate"
    HRV = "hrv"
    ACTIVITY_LEVEL = "activity_level"


class MonitoringState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    ALERT = "alert"
    EMERGENCY = "emergency"


@dataclass
class VitalReading:
    """Single timestamped vital sign measurement."""
    sign_type: VitalSignType
    value: float
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    unit: str = ""

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def is_fresh(self, max_age: float = 30.0) -> bool:
        return self.age_seconds() < max_age


@dataclass
class VitalSnapshot:
    """Complete snapshot of all vital signs at a point in time."""
    timestamp: float = field(default_factory=time.time)
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    temperature: Optional[float] = None
    respiratory_rate: Optional[float] = None
    hrv: Optional[float] = None
    activity_level: Optional[float] = None
    state: MonitoringState = MonitoringState.IDLE

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "heart_rate": self.heart_rate,
            "spo2": self.spo2,
            "bp_systolic": self.bp_systolic,
            "bp_diastolic": self.bp_diastolic,
            "temperature": self.temperature,
            "respiratory_rate": self.respiratory_rate,
            "hrv": self.hrv,
            "activity_level": self.activity_level,
            "state": self.state.value,
        }


class VitalSignsMonitor:
    """Real-time vital signs aggregator with alert generation and rolling history."""

    NORMAL_RANGES: Dict[VitalSignType, tuple] = {
        VitalSignType.HEART_RATE: (50.0, 100.0),
        VitalSignType.SPO2: (95.0, 100.0),
        VitalSignType.BLOOD_PRESSURE_SYSTOLIC: (90.0, 120.0),
        VitalSignType.BLOOD_PRESSURE_DIASTOLIC: (60.0, 80.0),
        VitalSignType.TEMPERATURE: (36.1, 37.2),
        VitalSignType.RESPIRATORY_RATE: (12.0, 20.0),
        VitalSignType.HRV: (20.0, 70.0),
        VitalSignType.ACTIVITY_LEVEL: (0.0, 10.0),
    }

    def __init__(self, history_size: int = 500, poll_interval: float = 1.0) -> None:
        self._lock = threading.RLock()
        self._history: Dict[VitalSignType, Deque[VitalReading]] = {
            vt: deque(maxlen=history_size) for vt in VitalSignType
        }
        self._latest: Dict[VitalSignType, Optional[VitalReading]] = {
            vt: None for vt in VitalSignType
        }
        self._state = MonitoringState.IDLE
        self._poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._alert_callbacks: List[Callable[[VitalReading, str], None]] = []
        self._snapshot_history: Deque[VitalSnapshot] = deque(maxlen=1440)  # 24h at 1/min
        logger.info("VitalSignsMonitor initialised (history=%d)", history_size)

    def register_alert_callback(self, cb: Callable[[VitalReading, str], None]) -> None:
        with self._lock:
            self._alert_callbacks.append(cb)

    def ingest(self, reading: VitalReading) -> None:
        """Accept a new biometric reading and check bounds."""
        with self._lock:
            self._history[reading.sign_type].append(reading)
            self._latest[reading.sign_type] = reading
            self._check_alert(reading)

    def _check_alert(self, reading: VitalReading) -> None:
        low, high = self.NORMAL_RANGES.get(reading.sign_type, (float("-inf"), float("inf")))
        if not (low <= reading.value <= high):
            msg = (
                f"{reading.sign_type.value} out of range: "
                f"{reading.value:.1f} (normal {low}-{high})"
            )
            logger.warning("ALERT: %s", msg)
            self._state = MonitoringState.ALERT
            for cb in self._alert_callbacks:
                try:
                    cb(reading, msg)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Alert callback error: %s", exc)

    def get_snapshot(self) -> VitalSnapshot:
        with self._lock:
            snap = VitalSnapshot(state=self._state)
            lat = self._latest
            snap.heart_rate = lat[VitalSignType.HEART_RATE].value if lat[VitalSignType.HEART_RATE] else None
            snap.spo2 = lat[VitalSignType.SPO2].value if lat[VitalSignType.SPO2] else None
            snap.bp_systolic = lat[VitalSignType.BLOOD_PRESSURE_SYSTOLIC].value if lat[VitalSignType.BLOOD_PRESSURE_SYSTOLIC] else None
            snap.bp_diastolic = lat[VitalSignType.BLOOD_PRESSURE_DIASTOLIC].value if lat[VitalSignType.BLOOD_PRESSURE_DIASTOLIC] else None
            snap.temperature = lat[VitalSignType.TEMPERATURE].value if lat[VitalSignType.TEMPERATURE] else None
            snap.respiratory_rate = lat[VitalSignType.RESPIRATORY_RATE].value if lat[VitalSignType.RESPIRATORY_RATE] else None
            snap.hrv = lat[VitalSignType.HRV].value if lat[VitalSignType.HRV] else None
            snap.activity_level = lat[VitalSignType.ACTIVITY_LEVEL].value if lat[VitalSignType.ACTIVITY_LEVEL] else None
            return snap

    def get_history(self, sign_type: VitalSignType, limit: int = 60) -> List[VitalReading]:
        with self._lock:
            hist = list(self._history[sign_type])
            return hist[-limit:]

    def _simulate_readings(self) -> None:
        """Produce synthetic vital sign readings for testing."""
        t = time.time()
        simulated = [
            VitalReading(VitalSignType.HEART_RATE, 70 + 5 * math.sin(t / 60), t),
            VitalReading(VitalSignType.SPO2, 98.0 + random.uniform(-0.5, 0.5), t),
            VitalReading(VitalSignType.BLOOD_PRESSURE_SYSTOLIC, 115 + random.uniform(-5, 5), t),
            VitalReading(VitalSignType.BLOOD_PRESSURE_DIASTOLIC, 75 + random.uniform(-3, 3), t),
            VitalReading(VitalSignType.TEMPERATURE, 36.6 + random.uniform(-0.2, 0.2), t),
            VitalReading(VitalSignType.RESPIRATORY_RATE, 16 + random.uniform(-1, 1), t),
            VitalReading(VitalSignType.HRV, 45 + random.uniform(-10, 10), t),
            VitalReading(VitalSignType.ACTIVITY_LEVEL, max(0, random.gauss(2, 1)), t),
        ]
        for r in simulated:
            self.ingest(r)

    def _run_loop(self) -> None:
        while self._running:
            self._simulate_readings()
            snap = self.get_snapshot()
            with self._lock:
                self._snapshot_history.append(snap)
            time.sleep(self._poll_interval)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._state = MonitoringState.ACTIVE
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="VitalSignsMonitor")
        self._thread.start()
        logger.info("VitalSignsMonitor started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._state = MonitoringState.IDLE
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("VitalSignsMonitor stopped")

    @property
    def state(self) -> MonitoringState:
        with self._lock:
            return self._state


_MONITOR: Optional["VitalSignsMonitor"] = None
_MONITOR_LOCK = threading.Lock()


def get_vital_signs_monitor() -> VitalSignsMonitor:
    global _MONITOR
    with _MONITOR_LOCK:
        if _MONITOR is None:
            _MONITOR = VitalSignsMonitor()
        return _MONITOR


def run_vital_signs_monitor_tests() -> bool:
    logger.info("Running VitalSignsMonitor tests...")
    monitor = VitalSignsMonitor(history_size=50, poll_interval=0.05)
    alerts: List[str] = []
    monitor.register_alert_callback(lambda r, msg: alerts.append(msg))
    monitor.start()
    time.sleep(0.3)
    monitor.stop()
    snap = monitor.get_snapshot()
    assert snap.heart_rate is not None, "No heart rate recorded"
    hist = monitor.get_history(VitalSignType.HEART_RATE, limit=5)
    assert len(hist) > 0, "No history"
    # inject out-of-range reading
    monitor.ingest(VitalReading(VitalSignType.HEART_RATE, 200.0))
    assert len(alerts) > 0, "No alert fired for out-of-range HR"
    logger.info("VitalSignsMonitor tests passed (%d alert(s))", len(alerts))
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = run_vital_signs_monitor_tests()
    print("Tests passed:", ok)
    mon = get_vital_signs_monitor()
    mon.start()
    time.sleep(2)
    print(mon.get_snapshot().to_dict())
    mon.stop()
