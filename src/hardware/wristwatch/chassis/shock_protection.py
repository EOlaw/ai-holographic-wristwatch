"""Drop and impact protection system for the AI Holographic Wristwatch."""
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


class ImpactSeverity(Enum):
    """Severity classification for detected impacts."""
    NONE = "none"
    MINOR = "minor"           # < 5 g
    MODERATE = "moderate"     # 5–20 g
    SEVERE = "severe"         # 20–60 g
    CRITICAL = "critical"     # > 60 g — emergency data save triggered


@dataclass
class ImpactEvent:
    """Record of a single impact or drop event."""
    timestamp: float = field(default_factory=time.time)
    peak_acceleration_g: float = 0.0
    severity: ImpactSeverity = ImpactSeverity.NONE
    duration_ms: float = 0.0
    pre_protection_activated: bool = False
    data_save_triggered: bool = False
    axis: str = "z"                   # dominant axis: x/y/z


@dataclass
class AccelerometerReading:
    """Raw 3-axis accelerometer sample."""
    timestamp: float = field(default_factory=time.time)
    x_g: float = 0.0
    y_g: float = 0.0
    z_g: float = 0.0

    @property
    def magnitude_g(self) -> float:
        return (self.x_g**2 + self.y_g**2 + self.z_g**2) ** 0.5


class ShockProtectionSystem:
    """Accelerometer-triggered pre-protection and emergency data save on drop detection."""

    MINOR_THRESHOLD_G = 5.0
    MODERATE_THRESHOLD_G = 20.0
    SEVERE_THRESHOLD_G = 60.0
    FREEFALL_THRESHOLD_G = 0.3      # total accel below this = freefall
    FREEFALL_CONFIRM_MS = 50.0      # min freefall duration to trigger pre-protection
    SAMPLE_RATE_HZ = 400

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._impact_history: List[ImpactEvent] = []
        self._pre_protection_active = False
        self._freefall_start: Optional[float] = None
        self._callbacks: List = []
        logger.info("ShockProtectionSystem initialised (sample rate %d Hz)", self.SAMPLE_RATE_HZ)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background shock monitoring."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="ShockMonitor", daemon=True
        )
        self._monitor_thread.start()
        logger.info("ShockProtectionSystem monitor started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("ShockProtectionSystem monitor stopped")

    def register_impact_callback(self, callback) -> None:
        """Register a callable invoked on every ImpactEvent."""
        with self._lock:
            self._callbacks.append(callback)

    def get_impact_history(self) -> List[ImpactEvent]:
        """Return a copy of recent impact events."""
        with self._lock:
            return list(self._impact_history)

    def is_pre_protection_active(self) -> bool:
        with self._lock:
            return self._pre_protection_active

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def classify_impact(peak_g: float) -> ImpactSeverity:
        if peak_g < ShockProtectionSystem.MINOR_THRESHOLD_G:
            return ImpactSeverity.NONE
        elif peak_g < ShockProtectionSystem.MODERATE_THRESHOLD_G:
            return ImpactSeverity.MINOR
        elif peak_g < ShockProtectionSystem.SEVERE_THRESHOLD_G:
            return ImpactSeverity.MODERATE
        elif peak_g < 100.0:
            return ImpactSeverity.SEVERE
        else:
            return ImpactSeverity.CRITICAL

    def _dominant_axis(self, r: AccelerometerReading) -> str:
        components = {"x": abs(r.x_g), "y": abs(r.y_g), "z": abs(r.z_g)}
        return max(components, key=components.__getitem__)

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _sample_accelerometer(self) -> AccelerometerReading:
        """Simulate IMU reading with occasional shock spikes."""
        spike = random.random() < 0.002  # 0.2 % chance of event per sample
        if spike:
            mag = random.uniform(5.0, 120.0)
        else:
            mag = random.gauss(0.0, 0.05)
        angle_xy = random.uniform(0, 6.28)
        z_frac = random.uniform(-1, 1)
        xy_frac = (1 - z_frac**2) ** 0.5 if abs(z_frac) <= 1 else 0.0
        return AccelerometerReading(
            x_g=mag * xy_frac * 0.7,
            y_g=mag * xy_frac * 0.3,
            z_g=mag * z_frac,
        )

    def _trigger_data_save(self) -> None:
        """Simulate emergency data flush (would flush flash / FRAM in real hw)."""
        logger.warning("ShockProtection: CRITICAL impact — triggering emergency data save")

    def _activate_pre_protection(self) -> None:
        with self._lock:
            self._pre_protection_active = True
        logger.info("ShockProtection: freefall detected — pre-protection ACTIVATED")

    def _deactivate_pre_protection(self) -> None:
        with self._lock:
            self._pre_protection_active = False
        logger.debug("ShockProtection: pre-protection deactivated")

    def _monitor_loop(self) -> None:
        interval = 1.0 / self.SAMPLE_RATE_HZ
        while True:
            with self._lock:
                if not self._running:
                    break

            reading = self._sample_accelerometer()
            mag = reading.magnitude_g
            now = time.time()

            # Freefall detection
            if mag < self.FREEFALL_THRESHOLD_G:
                if self._freefall_start is None:
                    self._freefall_start = now
                elif (now - self._freefall_start) * 1000 >= self.FREEFALL_CONFIRM_MS:
                    if not self._pre_protection_active:
                        self._activate_pre_protection()
            else:
                if self._freefall_start is not None:
                    self._freefall_start = None
                if self._pre_protection_active:
                    # Impact has occurred — record event
                    severity = self.classify_impact(mag)
                    if severity != ImpactSeverity.NONE:
                        event = ImpactEvent(
                            peak_acceleration_g=mag,
                            severity=severity,
                            duration_ms=random.uniform(2.0, 8.0),
                            pre_protection_activated=True,
                            data_save_triggered=(severity == ImpactSeverity.CRITICAL),
                            axis=self._dominant_axis(reading),
                        )
                        if event.data_save_triggered:
                            self._trigger_data_save()
                        with self._lock:
                            self._impact_history.append(event)
                            if len(self._impact_history) > 200:
                                self._impact_history = self._impact_history[-200:]
                            cbs = list(self._callbacks)
                        for cb in cbs:
                            try:
                                cb(event)
                            except Exception:
                                pass
                    self._deactivate_pre_protection()

            time.sleep(interval)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_GLOBAL_SHOCK: Optional[ShockProtectionSystem] = None
_GLOBAL_SHOCK_LOCK = threading.Lock()


def get_shock_protection_system() -> ShockProtectionSystem:
    global _GLOBAL_SHOCK
    with _GLOBAL_SHOCK_LOCK:
        if _GLOBAL_SHOCK is None:
            _GLOBAL_SHOCK = ShockProtectionSystem()
    return _GLOBAL_SHOCK


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_shock_protection_tests() -> bool:
    """Smoke-test the ShockProtectionSystem."""
    try:
        sys = ShockProtectionSystem()
        events_received: List[ImpactEvent] = []
        sys.register_impact_callback(events_received.append)
        sys.start()
        time.sleep(0.5)
        sys.stop()

        assert sys.classify_impact(3.0) == ImpactSeverity.NONE
        assert sys.classify_impact(10.0) == ImpactSeverity.MINOR
        assert sys.classify_impact(40.0) == ImpactSeverity.MODERATE
        assert sys.classify_impact(80.0) == ImpactSeverity.SEVERE
        assert sys.classify_impact(150.0) == ImpactSeverity.CRITICAL

        logger.info("ShockProtectionSystem tests PASSED")
        return True
    except Exception as exc:
        logger.error("ShockProtectionSystem tests FAILED: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ok = run_shock_protection_tests()
    print("shock_protection tests:", "PASS" if ok else "FAIL")

    sps = get_shock_protection_system()
    sps.start()
    time.sleep(1.0)
    sps.stop()
    print("Impact history:", sps.get_impact_history())
