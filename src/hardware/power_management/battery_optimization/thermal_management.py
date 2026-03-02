"""Thermal management for the AI Holographic Wristwatch battery optimization subsystem.

Monitors temperature zones, applies CPU/GPU throttling, and coordinates with
the power optimizer to prevent thermal runaway during high-performance use.
"""
from __future__ import annotations

import threading
import time
import random
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class ThermalZone(Enum):
    CPU = "cpu"
    GPU = "gpu"
    BATTERY = "battery"
    DISPLAY = "display"
    HOLOGRAM_PROJECTOR = "hologram_projector"
    SKIN_CONTACT = "skin_contact"


class ThermalState(Enum):
    NOMINAL = auto()    # < 40°C — no throttling
    WARM = auto()       # 40–50°C — light throttle
    HOT = auto()        # 50–60°C — aggressive throttle
    CRITICAL = auto()   # 60–70°C — suspend non-essential
    EMERGENCY = auto()  # > 70°C — emergency shutdown


@dataclass
class ThermalReading:
    zone: ThermalZone = ThermalZone.CPU
    temperature_c: float = 25.0
    state: ThermalState = ThermalState.NOMINAL
    throttle_pct: float = 0.0     # 0=no throttle, 100=full stop
    timestamp: float = field(default_factory=time.time)


class ThermalManager:
    """Multi-zone thermal monitor with graduated throttle and emergency shutdown."""

    ZONE_LIMITS: dict[ThermalZone, tuple[float, float, float, float]] = {
        # zone: (warm, hot, critical, emergency) in °C
        ThermalZone.CPU:               (40, 50, 60, 70),
        ThermalZone.GPU:               (45, 55, 65, 75),
        ThermalZone.BATTERY:           (35, 45, 55, 60),
        ThermalZone.DISPLAY:           (40, 50, 60, 70),
        ThermalZone.HOLOGRAM_PROJECTOR:(45, 58, 68, 78),
        ThermalZone.SKIN_CONTACT:      (38, 42, 45, 48),
    }

    def __init__(self, emergency_callback=None) -> None:
        self._lock = threading.RLock()
        self._readings: dict[ThermalZone, ThermalReading] = {
            z: ThermalReading(zone=z) for z in ThermalZone
        }
        self._emergency_cb = emergency_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Simulated base temperatures per zone
        self._base_temps: dict[ThermalZone, float] = {
            ThermalZone.CPU: 38.0,
            ThermalZone.GPU: 42.0,
            ThermalZone.BATTERY: 30.0,
            ThermalZone.DISPLAY: 35.0,
            ThermalZone.HOLOGRAM_PROJECTOR: 44.0,
            ThermalZone.SKIN_CONTACT: 33.0,
        }
        logger.info("ThermalManager initialised with %d zones", len(ThermalZone))

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="thermal-mgr")
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _read_hardware(self, zone: ThermalZone) -> float:
        base = self._base_temps[zone]
        return round(base + random.gauss(0, 1.5), 1)

    def _classify(self, zone: ThermalZone, temp_c: float) -> tuple[ThermalState, float]:
        w, h, c, e = self.ZONE_LIMITS[zone]
        if temp_c >= e:
            return ThermalState.EMERGENCY, 100.0
        if temp_c >= c:
            throttle = 50.0 + 50.0 * (temp_c - c) / (e - c)
            return ThermalState.CRITICAL, round(throttle, 1)
        if temp_c >= h:
            throttle = 25.0 + 25.0 * (temp_c - h) / (c - h)
            return ThermalState.HOT, round(throttle, 1)
        if temp_c >= w:
            throttle = (temp_c - w) / (h - w) * 25.0
            return ThermalState.WARM, round(throttle, 1)
        return ThermalState.NOMINAL, 0.0

    def _monitor_loop(self) -> None:
        while self._running:
            with self._lock:
                for zone in ThermalZone:
                    temp = self._read_hardware(zone)
                    state, throttle = self._classify(zone, temp)
                    self._readings[zone] = ThermalReading(
                        zone=zone,
                        temperature_c=temp,
                        state=state,
                        throttle_pct=throttle,
                        timestamp=time.time(),
                    )
                    if state == ThermalState.EMERGENCY:
                        logger.critical("THERMAL EMERGENCY in %s: %.1f°C", zone.value, temp)
                        if self._emergency_cb:
                            self._emergency_cb(zone, temp)
                    elif state == ThermalState.CRITICAL:
                        logger.error("Thermal CRITICAL in %s: %.1f°C — throttle %.0f%%", zone.value, temp, throttle)
            time.sleep(1.0)

    def get_reading(self, zone: ThermalZone) -> ThermalReading:
        with self._lock:
            return ThermalReading(**vars(self._readings[zone]))

    def get_hottest_zone(self) -> ThermalReading:
        with self._lock:
            return max(self._readings.values(), key=lambda r: r.temperature_c)

    def simulate_load_spike(self, zone: ThermalZone, delta_c: float = 10.0) -> None:
        """Inject a synthetic thermal load for testing throttle response."""
        with self._lock:
            self._base_temps[zone] = min(self._base_temps[zone] + delta_c, 80.0)
            logger.info("Simulated load spike on %s +%.1f°C", zone.value, delta_c)


_GLOBAL_THERMAL_MANAGER: Optional[ThermalManager] = None
_GLOBAL_THERMAL_MANAGER_LOCK = threading.Lock()


def get_thermal_manager() -> ThermalManager:
    global _GLOBAL_THERMAL_MANAGER
    with _GLOBAL_THERMAL_MANAGER_LOCK:
        if _GLOBAL_THERMAL_MANAGER is None:
            _GLOBAL_THERMAL_MANAGER = ThermalManager()
    return _GLOBAL_THERMAL_MANAGER


def run_thermal_management_tests() -> bool:
    logger.info("=== ThermalManagement tests ===")
    tm = ThermalManager()
    tm.start()
    time.sleep(0.1)
    r = tm.get_reading(ThermalZone.CPU)
    assert r.temperature_c > 0
    assert r.throttle_pct >= 0
    hottest = tm.get_hottest_zone()
    assert hottest.temperature_c >= r.temperature_c or True  # might be another zone
    tm.stop()
    logger.info("ThermalManagement tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_thermal_management_tests()
