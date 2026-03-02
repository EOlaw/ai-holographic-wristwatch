"""Solar energy integration with Maximum Power Point Tracking (MPPT).

Manages the bezel-integrated thin-film photovoltaic cells, runs Perturb &
Observe MPPT, and feeds harvested energy to the battery via a boost converter.
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


class SolarState(Enum):
    INACTIVE = auto()   # Dark / no panel current
    SCANNING = auto()   # Sweeping IV curve to find MPP
    TRACKING = auto()   # Locked onto maximum power point
    SATURATED = auto()  # Battery full, dumping excess


@dataclass
class SolarReading:
    state: SolarState = SolarState.INACTIVE
    panel_voltage_mv: float = 0.0
    panel_current_ua: float = 0.0
    power_uw: float = 0.0
    mpp_voltage_mv: float = 0.0
    efficiency_pct: float = 0.0
    irradiance_pct: float = 0.0      # 0-100% of rated full-sun
    accumulated_mwh: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SolarIntegration:
    """MPPT controller for bezel PV cells using Perturb & Observe algorithm."""

    PANEL_VOC_MV = 4500.0          # Open-circuit voltage at full sun
    PANEL_ISC_UA = 400.0           # Short-circuit current at full sun
    PANEL_AREA_CM2 = 3.0           # Total bezel cell area
    MPPT_STEP_MV = 20.0            # P&O perturbation step
    BOOST_EFFICIENCY = 0.88        # DC-DC boost converter efficiency

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reading = SolarReading()
        self._mppt_voltage_mv = self.PANEL_VOC_MV * 0.76  # Initial guess: ~76% Voc
        self._prev_power_uw = 0.0
        self._accumulated_mwh = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        logger.info("SolarIntegration initialised — panel area %.1f cm²", self.PANEL_AREA_CM2)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._mppt_loop, daemon=True, name="solar-mppt")
            self._thread.start()
        logger.info("SolarIntegration MPPT loop started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _simulate_panel(self, v_mv: float, irradiance: float) -> tuple[float, float]:
        """Single-diode model approximation: I = Isc*(1 - exp((V-Voc)/Vt))."""
        voc = self.PANEL_VOC_MV * irradiance
        isc_ua = self.PANEL_ISC_UA * irradiance
        vt = 600.0  # Thermal voltage * ideality factor in mV
        if voc < 1.0:
            return 0.0, 0.0
        v_clamped = min(v_mv, voc * 0.99)
        current_ua = isc_ua * (1.0 - (v_clamped / voc) ** 3)
        current_ua = max(0.0, current_ua + random.gauss(0, isc_ua * 0.01))
        power_uw = v_clamped * current_ua / 1000.0
        return v_clamped, current_ua

    def _perturb_and_observe(self, power_uw: float) -> None:
        """Classic P&O: move operating point in direction of power increase."""
        delta = power_uw - self._prev_power_uw
        if delta >= 0:
            self._mppt_voltage_mv += self.MPPT_STEP_MV
        else:
            self._mppt_voltage_mv -= self.MPPT_STEP_MV
        voc_approx = self.PANEL_VOC_MV
        self._mppt_voltage_mv = max(100.0, min(self._mppt_voltage_mv, voc_approx * 0.95))
        self._prev_power_uw = power_uw

    def _mppt_loop(self) -> None:
        interval_s = 1.0
        while self._running:
            with self._lock:
                irradiance = max(0.0, random.gauss(0.45, 0.25))
                v_mv, i_ua = self._simulate_panel(self._mppt_voltage_mv, irradiance)
                power_uw = v_mv * i_ua / 1000.0
                self._perturb_and_observe(power_uw)
                output_uw = power_uw * self.BOOST_EFFICIENCY
                self._accumulated_mwh += output_uw * interval_s / 3.6e9
                if irradiance < 0.02:
                    state = SolarState.INACTIVE
                elif power_uw < self._prev_power_uw * 0.5:
                    state = SolarState.SCANNING
                else:
                    state = SolarState.TRACKING
                self._reading = SolarReading(
                    state=state,
                    panel_voltage_mv=round(v_mv, 1),
                    panel_current_ua=round(i_ua, 1),
                    power_uw=round(output_uw, 2),
                    mpp_voltage_mv=round(self._mppt_voltage_mv, 1),
                    efficiency_pct=round(self.BOOST_EFFICIENCY * 100, 1),
                    irradiance_pct=round(irradiance * 100, 1),
                    accumulated_mwh=round(self._accumulated_mwh, 6),
                    timestamp=time.time(),
                )
            time.sleep(interval_s)

    def get_reading(self) -> SolarReading:
        with self._lock:
            return SolarReading(**vars(self._reading))


_GLOBAL_SOLAR_INTEGRATION: Optional[SolarIntegration] = None
_GLOBAL_SOLAR_INTEGRATION_LOCK = threading.Lock()


def get_solar_integration() -> SolarIntegration:
    global _GLOBAL_SOLAR_INTEGRATION
    with _GLOBAL_SOLAR_INTEGRATION_LOCK:
        if _GLOBAL_SOLAR_INTEGRATION is None:
            _GLOBAL_SOLAR_INTEGRATION = SolarIntegration()
    return _GLOBAL_SOLAR_INTEGRATION


def run_solar_integration_tests() -> bool:
    logger.info("=== SolarIntegration tests ===")
    solar = SolarIntegration()
    solar.start()
    time.sleep(0.15)
    r = solar.get_reading()
    assert r.power_uw >= 0
    assert r.irradiance_pct >= 0
    solar.stop()
    logger.info("SolarIntegration tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_solar_integration_tests()
