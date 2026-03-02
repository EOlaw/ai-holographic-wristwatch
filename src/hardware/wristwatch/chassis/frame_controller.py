"""Titanium chassis flex and vibration control for the AI Holographic Wristwatch."""
from __future__ import annotations
import threading
import time
import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from src.core.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FrameState(Enum):
    """Operational states of the titanium chassis frame."""
    NOMINAL = "nominal"
    FLEX_DETECTED = "flex_detected"
    VIBRATION_ACTIVE = "vibration_active"
    DAMPENING = "dampening"
    OVERHEAT = "overheat"
    IMPACT_LOCKED = "impact_locked"
    CALIBRATING = "calibrating"


@dataclass
class FrameReading:
    """Snapshot of chassis frame sensor data."""
    timestamp: float = field(default_factory=time.time)
    flex_magnitude_um: float = 0.0          # micrometers of flex
    vibration_hz: float = 0.0              # dominant vibration frequency
    vibration_amplitude_g: float = 0.0    # acceleration in g
    case_temperature_c: float = 25.0
    dampening_active: bool = False
    state: FrameState = FrameState.NOMINAL


class FrameController:
    """Controls the titanium chassis frame: vibration dampening, flex detection, and thermal."""

    FLEX_ALERT_UM = 80.0        # micrometers — alert threshold
    FLEX_CRITICAL_UM = 150.0    # micrometers — shutdown threshold
    VIBRATION_ALERT_HZ = 200.0  # Hz — alert
    TEMP_MAX_C = 55.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = FrameState.NOMINAL
        self._history: List[FrameReading] = []
        self._dampening_on = False
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        logger.info("FrameController initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin continuous chassis monitoring in background thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="FrameMonitor", daemon=True
        )
        self._monitor_thread.start()
        logger.info("FrameController monitor started")

    def stop(self) -> None:
        """Stop the monitor thread gracefully."""
        with self._lock:
            self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=3.0)
        logger.info("FrameController monitor stopped")

    def get_reading(self) -> FrameReading:
        """Return the most recent chassis reading."""
        with self._lock:
            if self._history:
                return self._history[-1]
        return self._sample_sensors()

    def get_state(self) -> FrameState:
        """Return current frame state."""
        with self._lock:
            return self._state

    def enable_dampening(self) -> None:
        """Activate active vibration-dampening piezo elements."""
        with self._lock:
            self._dampening_on = True
        logger.info("Vibration dampening ENABLED")

    def disable_dampening(self) -> None:
        """Deactivate dampening to conserve power."""
        with self._lock:
            self._dampening_on = False
        logger.info("Vibration dampening DISABLED")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_sensors(self) -> FrameReading:
        """Simulate reading IMU + flex + thermal sensors."""
        flex = random.gauss(10.0, 5.0)
        vib_hz = random.uniform(0.0, 50.0)
        vib_amp = random.uniform(0.0, 0.3)
        temp = random.gauss(32.0, 3.0)
        return FrameReading(
            flex_magnitude_um=max(0.0, flex),
            vibration_hz=vib_hz,
            vibration_amplitude_g=vib_amp,
            case_temperature_c=temp,
            dampening_active=self._dampening_on,
            state=self._state,
        )

    def _evaluate_state(self, reading: FrameReading) -> FrameState:
        """Determine next state from sensor data."""
        if reading.case_temperature_c >= self.TEMP_MAX_C:
            return FrameState.OVERHEAT
        if reading.flex_magnitude_um >= self.FLEX_CRITICAL_UM:
            return FrameState.IMPACT_LOCKED
        if reading.flex_magnitude_um >= self.FLEX_ALERT_UM:
            return FrameState.FLEX_DETECTED
        if reading.vibration_hz >= self.VIBRATION_ALERT_HZ:
            return FrameState.VIBRATION_ACTIVE
        if self._dampening_on:
            return FrameState.DAMPENING
        return FrameState.NOMINAL

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            with self._lock:
                if not self._running:
                    break
            reading = self._sample_sensors()
            new_state = self._evaluate_state(reading)
            reading.state = new_state
            with self._lock:
                self._state = new_state
                self._history.append(reading)
                if len(self._history) > 500:
                    self._history = self._history[-500:]
            if new_state in (FrameState.OVERHEAT, FrameState.IMPACT_LOCKED):
                logger.warning("FrameController: critical state %s", new_state.value)
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_GLOBAL_FRAME_CONTROLLER: Optional[FrameController] = None
_GLOBAL_FRAME_CONTROLLER_LOCK = threading.Lock()


def get_frame_controller() -> FrameController:
    """Return the process-wide FrameController singleton."""
    global _GLOBAL_FRAME_CONTROLLER
    with _GLOBAL_FRAME_CONTROLLER_LOCK:
        if _GLOBAL_FRAME_CONTROLLER is None:
            _GLOBAL_FRAME_CONTROLLER = FrameController()
    return _GLOBAL_FRAME_CONTROLLER


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_frame_controller_tests() -> bool:
    """Smoke-test the FrameController."""
    try:
        ctrl = FrameController()
        ctrl.start()
        time.sleep(0.3)
        reading = ctrl.get_reading()
        assert reading.flex_magnitude_um >= 0.0, "flex must be non-negative"
        ctrl.enable_dampening()
        assert ctrl.get_state() is not None
        ctrl.stop()
        logger.info("FrameController tests PASSED")
        return True
    except Exception as exc:
        logger.error("FrameController tests FAILED: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ok = run_frame_controller_tests()
    print("frame_controller tests:", "PASS" if ok else "FAIL")
    ctrl = get_frame_controller()
    ctrl.start()
    for _ in range(3):
        print(ctrl.get_reading())
        time.sleep(0.2)
    ctrl.stop()
