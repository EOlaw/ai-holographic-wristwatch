"""IP68/5ATM water resistance monitoring for the AI Holographic Wristwatch."""
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


class WaterResistanceLevel(Enum):
    """Water resistance rating classifications."""
    NOT_RATED = "not_rated"
    IPX4 = "ipx4"          # splash resistant
    IP67 = "ip67"          # 1 m / 30 min
    IP68 = "ip68"          # 2 m+ / defined by manufacturer
    ATM5 = "5atm"          # 50 m static water pressure


class SealStatus(Enum):
    """Current integrity state of the gasket system."""
    INTACT = "intact"
    DEGRADED = "degraded"          # partial loss — warn user
    BREACHED = "breached"          # ingress detected — critical
    UNKNOWN = "unknown"


@dataclass
class SealIntegrityReading:
    """Snapshot from the pressure / capacitive seal sensors."""
    timestamp: float = field(default_factory=time.time)
    internal_pressure_hpa: float = 1013.25    # nominal sea-level
    external_pressure_hpa: float = 1013.25
    pressure_delta_hpa: float = 0.0
    capacitive_moisture_pf: float = 0.0       # pF above baseline = moisture
    seal_status: SealStatus = SealStatus.INTACT
    depth_estimate_m: float = 0.0
    ingress_detected: bool = False
    rating: WaterResistanceLevel = WaterResistanceLevel.IP68


class WaterproofingMonitor:
    """Monitors seals via pressure sensors and capacitive ingress detection."""

    PRESSURE_DELTA_WARN_HPA = 10.0       # hPa differential = suspect ingress
    PRESSURE_DELTA_CRITICAL_HPA = 30.0   # definite ingress
    MOISTURE_WARN_PF = 2.0               # pF above baseline
    MOISTURE_CRITICAL_PF = 5.0
    WATER_DENSITY = 1025.0               # kg/m³ (seawater)
    G = 9.81                             # m/s²
    RATED_DEPTH_M = 50.0                 # 5 ATM
    RATED_LEVEL = WaterResistanceLevel.ATM5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._history: List[SealIntegrityReading] = []
        self._alert_callbacks: List = []
        self._simulated_depth_m: float = 0.0
        logger.info(
            "WaterproofingMonitor initialised — rated %s / %.0f m",
            self.RATED_LEVEL.value, self.RATED_DEPTH_M,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin continuous seal monitoring."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="WaterproofMonitor", daemon=True
        )
        self._monitor_thread.start()
        logger.info("WaterproofingMonitor started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=3.0)

    def get_latest_reading(self) -> SealIntegrityReading:
        with self._lock:
            if self._history:
                return self._history[-1]
        return self._sample_sensors()

    def simulate_submersion(self, depth_m: float) -> None:
        """For testing: simulate the watch being at a given water depth."""
        with self._lock:
            self._simulated_depth_m = max(0.0, depth_m)
        logger.info("Simulating submersion at %.1f m", depth_m)

    def register_alert_callback(self, callback) -> None:
        with self._lock:
            self._alert_callbacks.append(callback)

    def get_seal_history(self) -> List[SealIntegrityReading]:
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _depth_to_pressure_hpa(self, depth_m: float) -> float:
        """Convert water depth to absolute pressure in hPa."""
        return 1013.25 + (self.WATER_DENSITY * self.G * depth_m) / 100.0

    def _sample_sensors(self) -> SealIntegrityReading:
        """Simulate pressure and capacitive moisture sensor readings."""
        with self._lock:
            depth = self._simulated_depth_m
        internal = 1013.25 + random.gauss(0.0, 0.5)  # slight noise
        external = self._depth_to_pressure_hpa(depth) + random.gauss(0.0, 1.0)
        delta = max(0.0, external - internal)
        moisture = max(0.0, random.gauss(0.0, 0.3) + (0.5 if delta > 5 else 0.0))
        depth_est = (external - 1013.25) * 100.0 / (self.WATER_DENSITY * self.G)
        depth_est = max(0.0, depth_est)
        ingress = (delta >= self.PRESSURE_DELTA_CRITICAL_HPA
                   or moisture >= self.MOISTURE_CRITICAL_PF)
        if ingress:
            status = SealStatus.BREACHED
        elif delta >= self.PRESSURE_DELTA_WARN_HPA or moisture >= self.MOISTURE_WARN_PF:
            status = SealStatus.DEGRADED
        else:
            status = SealStatus.INTACT
        return SealIntegrityReading(
            internal_pressure_hpa=internal,
            external_pressure_hpa=external,
            pressure_delta_hpa=delta,
            capacitive_moisture_pf=moisture,
            seal_status=status,
            depth_estimate_m=depth_est,
            ingress_detected=ingress,
            rating=self.RATED_LEVEL,
        )

    def _fire_alerts(self, reading: SealIntegrityReading) -> None:
        if reading.seal_status in (SealStatus.DEGRADED, SealStatus.BREACHED):
            level = "CRITICAL" if reading.ingress_detected else "WARN"
            logger.warning(
                "Seal %s: delta=%.1f hPa moisture=%.2f pF",
                level, reading.pressure_delta_hpa, reading.capacitive_moisture_pf,
            )
            with self._lock:
                cbs = list(self._alert_callbacks)
            for cb in cbs:
                try:
                    cb(reading)
                except Exception:
                    pass

    def _monitor_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break
            reading = self._sample_sensors()
            self._fire_alerts(reading)
            with self._lock:
                self._history.append(reading)
                if len(self._history) > 600:
                    self._history = self._history[-600:]
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_GLOBAL_WP: Optional[WaterproofingMonitor] = None
_GLOBAL_WP_LOCK = threading.Lock()


def get_waterproofing_monitor() -> WaterproofingMonitor:
    global _GLOBAL_WP
    with _GLOBAL_WP_LOCK:
        if _GLOBAL_WP is None:
            _GLOBAL_WP = WaterproofingMonitor()
    return _GLOBAL_WP


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_waterproofing_tests() -> bool:
    try:
        monitor = WaterproofingMonitor()
        monitor.start()
        time.sleep(0.5)

        reading = monitor.get_latest_reading()
        assert reading.seal_status == SealStatus.INTACT, "should be intact at surface"

        monitor.simulate_submersion(3.0)
        time.sleep(1.0)
        reading = monitor.get_latest_reading()
        assert reading.depth_estimate_m >= 0.0

        # Verify depth conversion
        p = monitor._depth_to_pressure_hpa(10.0)
        assert p > 1013.25, "deeper should be higher pressure"

        monitor.stop()
        logger.info("WaterproofingMonitor tests PASSED")
        return True
    except Exception as exc:
        logger.error("WaterproofingMonitor tests FAILED: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ok = run_waterproofing_tests()
    print("waterproofing tests:", "PASS" if ok else "FAIL")

    mon = get_waterproofing_monitor()
    mon.start()
    for depth in (0.0, 5.0, 20.0, 60.0):
        mon.simulate_submersion(depth)
        time.sleep(0.7)
        r = mon.get_latest_reading()
        print(f"  Depth {depth} m → status={r.seal_status.value} Δp={r.pressure_delta_hpa:.1f} hPa")
    mon.stop()
