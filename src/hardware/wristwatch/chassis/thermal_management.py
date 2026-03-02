"""Chassis thermal routing, heat pipe control, and TIM monitoring for the wristwatch."""
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


class ThermalPath(Enum):
    """Named heat dissipation paths within the chassis."""
    CASE_BACK = "case_back"           # primary path through titanium back
    HEAT_PIPE_1 = "heat_pipe_1"       # micro heat pipe from SoC to case edge
    HEAT_PIPE_2 = "heat_pipe_2"       # micro heat pipe from display driver
    GRAPHENE_SPREADER = "graphene_spreader"  # graphene film under SoC
    STRAP_COPPER = "strap_copper"     # copper traces in smart strap


class ThermalZone(Enum):
    """Thermal zones monitored inside the chassis."""
    SOC = "soc"
    DISPLAY = "display"
    BATTERY = "battery"
    STRAP_CONNECTOR = "strap_connector"
    AMBIENT = "ambient"


@dataclass
class ThermalReading:
    """Multi-zone temperature snapshot."""
    timestamp: float = field(default_factory=time.time)
    zone_temps_c: Dict[str, float] = field(default_factory=dict)
    active_paths: List[str] = field(default_factory=list)
    tim_degradation_pct: float = 0.0    # thermal interface material wear 0–100 %
    max_temp_c: float = 0.0

    def hottest_zone(self) -> str:
        if not self.zone_temps_c:
            return "unknown"
        return max(self.zone_temps_c, key=self.zone_temps_c.__getitem__)


class ChassisThermalManager:
    """Manages heat pipe routing and thermal interface material health monitoring."""

    WARN_TEMP_C = 45.0
    CRITICAL_TEMP_C = 60.0
    TIM_REPLACE_THRESHOLD_PCT = 70.0    # recommend TIM replacement above this

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._active_paths: List[ThermalPath] = [
            ThermalPath.CASE_BACK,
            ThermalPath.GRAPHENE_SPREADER,
        ]
        self._history: List[ThermalReading] = []
        self._tim_age_hours: float = 0.0   # simulated usage hours
        logger.info("ChassisThermalManager initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start continuous thermal monitoring."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="ThermalMonitor", daemon=True
        )
        self._monitor_thread.start()
        logger.info("ChassisThermalManager monitor started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=3.0)
        logger.info("ChassisThermalManager monitor stopped")

    def get_latest_reading(self) -> ThermalReading:
        """Return the most recent thermal snapshot."""
        with self._lock:
            if self._history:
                return self._history[-1]
        return self._sample_zones()

    def enable_heat_pipe(self, path: ThermalPath) -> None:
        """Activate a thermal path (e.g., heat pipe via pump actuation)."""
        with self._lock:
            if path not in self._active_paths:
                self._active_paths.append(path)
        logger.info("ThermalPath ENABLED: %s", path.value)

    def disable_heat_pipe(self, path: ThermalPath) -> None:
        with self._lock:
            self._active_paths = [p for p in self._active_paths if p != path]
        logger.info("ThermalPath DISABLED: %s", path.value)

    def get_tim_health(self) -> float:
        """Return estimated TIM degradation percentage (0–100)."""
        with self._lock:
            return self._tim_degradation()

    def get_active_paths(self) -> List[ThermalPath]:
        with self._lock:
            return list(self._active_paths)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tim_degradation(self) -> float:
        """Estimate TIM wear from usage hours."""
        # TIM degrades ~1 % per 50 operating hours (simulated)
        return min(100.0, self._tim_age_hours / 50.0)

    def _sample_zones(self) -> ThermalReading:
        """Simulate zone temperature readings."""
        base = random.gauss(35.0, 2.0)
        zones = {
            ThermalZone.SOC.value: base + random.gauss(8.0, 1.5),
            ThermalZone.DISPLAY.value: base + random.gauss(5.0, 1.0),
            ThermalZone.BATTERY.value: base + random.gauss(3.0, 0.8),
            ThermalZone.STRAP_CONNECTOR.value: base + random.gauss(1.0, 0.5),
            ThermalZone.AMBIENT.value: random.gauss(22.0, 2.0),
        }
        with self._lock:
            active = [p.value for p in self._active_paths]
            tim_deg = self._tim_degradation()
        return ThermalReading(
            zone_temps_c=zones,
            active_paths=active,
            tim_degradation_pct=tim_deg,
            max_temp_c=max(zones.values()),
        )

    def _handle_overtemp(self, reading: ThermalReading) -> None:
        """React to over-temperature events by enabling additional paths."""
        if reading.max_temp_c >= self.CRITICAL_TEMP_C:
            logger.warning(
                "CRITICAL temperature %.1f°C in zone %s",
                reading.max_temp_c, reading.hottest_zone(),
            )
            self.enable_heat_pipe(ThermalPath.HEAT_PIPE_1)
            self.enable_heat_pipe(ThermalPath.HEAT_PIPE_2)
        elif reading.max_temp_c >= self.WARN_TEMP_C:
            logger.info(
                "WARM %.1f°C in zone %s — enabling heat pipe 1",
                reading.max_temp_c, reading.hottest_zone(),
            )
            self.enable_heat_pipe(ThermalPath.HEAT_PIPE_1)

    def _monitor_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break
                self._tim_age_hours += (1.0 / 3600.0)  # 1 s per loop tick

            reading = self._sample_zones()

            if reading.tim_degradation_pct >= self.TIM_REPLACE_THRESHOLD_PCT:
                logger.warning(
                    "TIM degradation %.1f%% — replacement recommended",
                    reading.tim_degradation_pct,
                )
            self._handle_overtemp(reading)

            with self._lock:
                self._history.append(reading)
                if len(self._history) > 300:
                    self._history = self._history[-300:]
            time.sleep(1.0)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_GLOBAL_THERMAL: Optional[ChassisThermalManager] = None
_GLOBAL_THERMAL_LOCK = threading.Lock()


def get_chassis_thermal_manager() -> ChassisThermalManager:
    global _GLOBAL_THERMAL
    with _GLOBAL_THERMAL_LOCK:
        if _GLOBAL_THERMAL is None:
            _GLOBAL_THERMAL = ChassisThermalManager()
    return _GLOBAL_THERMAL


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_thermal_management_tests() -> bool:
    """Smoke-test the ChassisThermalManager."""
    try:
        mgr = ChassisThermalManager()
        mgr.start()
        time.sleep(2.5)
        reading = mgr.get_latest_reading()
        assert reading.max_temp_c > 0.0
        assert ThermalZone.SOC.value in reading.zone_temps_c
        assert reading.hottest_zone() in reading.zone_temps_c

        mgr.enable_heat_pipe(ThermalPath.STRAP_COPPER)
        assert ThermalPath.STRAP_COPPER in mgr.get_active_paths()
        mgr.disable_heat_pipe(ThermalPath.STRAP_COPPER)
        assert ThermalPath.STRAP_COPPER not in mgr.get_active_paths()

        mgr.stop()
        logger.info("ChassisThermalManager tests PASSED")
        return True
    except Exception as exc:
        logger.error("ChassisThermalManager tests FAILED: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ok = run_thermal_management_tests()
    print("thermal_management tests:", "PASS" if ok else "FAIL")

    mgr = get_chassis_thermal_manager()
    mgr.start()
    for _ in range(3):
        r = mgr.get_latest_reading()
        print(f"  Max temp: {r.max_temp_c:.1f}°C hottest={r.hottest_zone()} TIM={r.tim_degradation_pct:.2f}%")
        time.sleep(1.1)
    mgr.stop()
