"""
Calibration Manager — AI Holographic Wristwatch

Manages system-wide sensor calibration lifecycle:
- Orchestrates calibration sequences for all sensors
- Stores and retrieves calibration data with versioning
- Detects calibration drift and schedules recalibration
- Temperature compensation for calibration parameters
- Factory calibration vs user calibration management
- SensorInterface compliance for health monitoring
"""

from __future__ import annotations

import json
import math
import threading
import time
import random
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from src.core.interfaces.sensor_interface import SensorInterface, SensorReading, SensorStatus, SensorInfo
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class CalibrationState(Enum):
    UNCALIBRATED  = "uncalibrated"
    IN_PROGRESS   = "in_progress"
    VALID         = "valid"
    STALE         = "stale"       # calibration too old
    DRIFTED       = "drifted"     # drift detected
    FAILED        = "failed"


class CalibrationTrigger(Enum):
    FACTORY       = "factory"
    USER_MANUAL   = "user_manual"
    AUTO_DRIFT    = "auto_drift"
    TEMPERATURE   = "temperature"
    SCHEDULED     = "scheduled"


@dataclass
class SensorCalibrationRecord:
    """Calibration record for a single sensor."""
    sensor_id: str
    state: CalibrationState = CalibrationState.UNCALIBRATED
    trigger: CalibrationTrigger = CalibrationTrigger.FACTORY
    calibrated_at: float = 0.0
    expires_at: float = 0.0
    temperature_at_calibration_c: float = 25.0
    accuracy_estimate: float = 1.0    # 0=worst, 1=best
    bias_x: float = 0.0
    bias_y: float = 0.0
    bias_z: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    drift_rate: float = 0.0           # estimated drift per hour
    version: int = 1
    metadata: Dict = field(default_factory=dict)

    def is_valid(self) -> bool:
        return (self.state == CalibrationState.VALID
                and time.time() < self.expires_at)

    def age_hours(self) -> float:
        return (time.time() - self.calibrated_at) / 3600.0


@dataclass
class CalibrationManagerReading(SensorReading):
    sensor_calibration_states: Dict[str, str] = field(default_factory=dict)
    sensors_needing_calibration: List[str] = field(default_factory=list)
    last_calibration_sensor: str = ""
    last_calibration_timestamp: float = 0.0
    overall_system_accuracy: float = 1.0
    temperature_compensation_active: bool = False
    current_temperature_c: float = 25.0


class DriftDetector:
    """
    Detects calibration drift by monitoring rolling statistics of sensor readings.
    Alerts when mean bias exceeds a threshold relative to calibration baseline.
    """

    DRIFT_THRESHOLD  = 0.05   # 5% of full scale
    WINDOW_SIZE      = 100

    def __init__(self, baseline_bias: float = 0.0) -> None:
        self._baseline = baseline_bias
        self._buffer: List[float] = []

    def update(self, value: float) -> Tuple[bool, float]:
        """Returns (drift_detected, drift_amount)."""
        self._buffer.append(value)
        if len(self._buffer) > self.WINDOW_SIZE:
            self._buffer.pop(0)
        if len(self._buffer) < 10:
            return False, 0.0
        current_mean = sum(self._buffer) / len(self._buffer)
        drift = abs(current_mean - self._baseline)
        return drift > self.DRIFT_THRESHOLD, drift


class TemperatureCompensator:
    """
    Applies temperature correction to sensor readings.
    Uses linear TC model: corrected = raw + TC * (T - T_cal)
    """

    def __init__(self, tc_coefficient: float = 0.001) -> None:
        self._tc = tc_coefficient   # units per °C

    def compensate(self, raw_value: float, current_temp: float,
                   cal_temp: float) -> float:
        delta_t = current_temp - cal_temp
        correction = self._tc * delta_t
        return raw_value - correction


_GLOBAL_CAL_MGR: Optional["CalibrationManager"] = None
_GLOBAL_CAL_MGR_LOCK = threading.Lock()


class CalibrationManager(SensorInterface):
    """
    System-wide calibration lifecycle manager.
    Tracks calibration state for all sensors, detects drift,
    and orchestrates recalibration procedures.
    """

    SENSOR_ID    = "fusion.calibration_manager"
    SENSOR_TYPE  = "calibration_manager"
    MODEL        = "CalMgr-v1"
    MANUFACTURER = "AI Holographic"

    # Default calibration validity periods (hours)
    VALIDITY_HOURS: Dict[str, float] = {
        "accelerometer": 720.0,   # 30 days
        "gyroscope":     168.0,   # 7 days
        "magnetometer":  24.0,    # 1 day (environment-dependent)
        "heart_rate":    168.0,
        "temperature":   720.0,
        "pressure":      720.0,
    }

    def __init__(self) -> None:
        self._records: Dict[str, SensorCalibrationRecord] = {}
        self._drift_detectors: Dict[str, DriftDetector] = {}
        self._tc = TemperatureCompensator()
        self._current_temp = 25.0
        self._calibration_callbacks: List[Callable[[str, CalibrationState], None]] = []

        self._lock = threading.RLock()
        self._running = self._initialized = False
        self._read_count = 0
        self._last_reading: Optional[CalibrationManagerReading] = None

    def initialize(self) -> bool:
        with self._lock:
            self._initialize_factory_calibrations()
            self._initialized = self._running = True
            logger.info("CalibrationManager initialized")
            return True

    def read(self) -> Optional[CalibrationManagerReading]:
        if not self._initialized:
            return None
        with self._lock:
            self._current_temp = random.gauss(25.0, 2.0)
            needs_cal = [sid for sid, rec in self._records.items() if not rec.is_valid()]

            accuracy_vals = [r.accuracy_estimate for r in self._records.values()]
            overall_accuracy = sum(accuracy_vals) / max(1, len(accuracy_vals))

            last_cal_sensor = ""
            last_cal_ts = 0.0
            for sid, rec in self._records.items():
                if rec.calibrated_at > last_cal_ts:
                    last_cal_ts = rec.calibrated_at
                    last_cal_sensor = sid

            reading = CalibrationManagerReading(
                sensor_id=self.SENSOR_ID,
                timestamp=time.time(),
                sensor_calibration_states={sid: r.state.value for sid, r in self._records.items()},
                sensors_needing_calibration=needs_cal,
                last_calibration_sensor=last_cal_sensor,
                last_calibration_timestamp=last_cal_ts,
                overall_system_accuracy=overall_accuracy,
                temperature_compensation_active=True,
                current_temperature_c=self._current_temp,
                confidence=overall_accuracy,
            )
            self._last_reading = reading
            self._read_count += 1
            return reading

    async def stream(self):
        import asyncio
        while self._running:
            r = self.read()
            if r: yield r
            await asyncio.sleep(60.0)

    def calibrate(self) -> bool:
        with self._lock:
            for sensor_id in list(self._records.keys()):
                self._do_calibrate(sensor_id)
            logger.info("All sensors recalibrated")
            return True

    def shutdown(self) -> None:
        with self._lock:
            self._running = self._initialized = False

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(
            sensor_id=self.SENSOR_ID, sensor_type=self.SENSOR_TYPE,
            model=self.MODEL, manufacturer=self.MANUFACTURER,
            firmware_version="1.0.0", hardware_version="software",
            capabilities={"drift_detection": True, "temperature_compensation": True,
                          "managed_sensors": len(self.VALIDITY_HOURS)},
        )

    def get_status(self) -> SensorStatus:
        if not self._initialized: return SensorStatus.UNINITIALIZED
        return SensorStatus.RUNNING if self._running else SensorStatus.IDLE

    def is_healthy(self) -> bool:
        return self.get_status() in (SensorStatus.RUNNING, SensorStatus.IDLE)

    def get_health_report(self) -> Dict:
        with self._lock:
            return {
                "status": self.get_status().value,
                "tracked_sensors": len(self._records),
                "valid_calibrations": sum(1 for r in self._records.values() if r.is_valid()),
                "temperature_c": self._current_temp,
            }

    def read_sync(self):
        return self.read()

    # ------------------------------------------------------------------
    # Calibration management API
    # ------------------------------------------------------------------

    def register_sensor(self, sensor_id: str) -> None:
        with self._lock:
            if sensor_id not in self._records:
                self._records[sensor_id] = SensorCalibrationRecord(sensor_id=sensor_id)
                self._drift_detectors[sensor_id] = DriftDetector()
                logger.info(f"Sensor '{sensor_id}' registered for calibration management")

    def calibrate_sensor(self, sensor_id: str,
                         trigger: CalibrationTrigger = CalibrationTrigger.USER_MANUAL) -> bool:
        with self._lock:
            return self._do_calibrate(sensor_id, trigger)

    def get_calibration_record(self, sensor_id: str) -> Optional[SensorCalibrationRecord]:
        with self._lock:
            return self._records.get(sensor_id)

    def update_drift(self, sensor_id: str, current_value: float) -> bool:
        """Feed latest reading to drift detector. Returns True if drift detected."""
        with self._lock:
            detector = self._drift_detectors.get(sensor_id)
            if not detector:
                return False
            drifted, amount = detector.update(current_value)
            if drifted:
                rec = self._records.get(sensor_id)
                if rec and rec.state == CalibrationState.VALID:
                    rec.state = CalibrationState.DRIFTED
                    rec.drift_rate = amount
                    logger.warning(f"Drift detected for '{sensor_id}': {amount:.4f}")
                    self._notify_callbacks(sensor_id, CalibrationState.DRIFTED)
            return drifted

    def register_callback(self, cb: Callable[[str, CalibrationState], None]) -> None:
        with self._lock:
            self._calibration_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_factory_calibrations(self) -> None:
        """Seed factory calibration records for known sensors."""
        for sensor_id in self.VALIDITY_HOURS:
            validity = self.VALIDITY_HOURS[sensor_id]
            rec = SensorCalibrationRecord(
                sensor_id=sensor_id,
                state=CalibrationState.VALID,
                trigger=CalibrationTrigger.FACTORY,
                calibrated_at=time.time(),
                expires_at=time.time() + validity * 3600,
                temperature_at_calibration_c=25.0,
                accuracy_estimate=0.95,
            )
            self._records[sensor_id] = rec
            self._drift_detectors[sensor_id] = DriftDetector()

    def _do_calibrate(
        self, sensor_id: str,
        trigger: CalibrationTrigger = CalibrationTrigger.USER_MANUAL
    ) -> bool:
        rec = self._records.get(sensor_id)
        if not rec:
            return False
        validity_h = self.VALIDITY_HOURS.get(sensor_id, 168.0)
        rec.state = CalibrationState.VALID
        rec.trigger = trigger
        rec.calibrated_at = time.time()
        rec.expires_at = time.time() + validity_h * 3600
        rec.temperature_at_calibration_c = self._current_temp
        rec.accuracy_estimate = random.gauss(0.97, 0.01)
        rec.version += 1
        self._notify_callbacks(sensor_id, CalibrationState.VALID)
        logger.info(f"Sensor '{sensor_id}' calibrated (trigger={trigger.value})")
        return True

    def _notify_callbacks(self, sensor_id: str, state: CalibrationState) -> None:
        for cb in self._calibration_callbacks:
            try:
                cb(sensor_id, state)
            except Exception as exc:
                logger.warning(f"Calibration callback error: {exc}")


def get_calibration_manager() -> CalibrationManager:
    global _GLOBAL_CAL_MGR
    with _GLOBAL_CAL_MGR_LOCK:
        if _GLOBAL_CAL_MGR is None:
            _GLOBAL_CAL_MGR = CalibrationManager()
        return _GLOBAL_CAL_MGR


def run_calibration_manager_tests() -> bool:
    cm = CalibrationManager()
    assert cm.initialize()
    r = cm.read()
    assert r is not None
    assert r.overall_system_accuracy > 0.0
    cm.register_sensor("test_sensor")
    assert cm.calibrate_sensor("test_sensor")
    rec = cm.get_calibration_record("test_sensor")
    assert rec and rec.is_valid()
    cm.shutdown()
    logger.info("CalibrationManager tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_calibration_manager_tests()
