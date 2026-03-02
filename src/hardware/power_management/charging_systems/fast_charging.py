"""Fast charging controller for USB-C PD and proprietary quick-charge protocols.

Manages USB Power Delivery negotiation, thermal-aware rate adjustment, and
charging profile execution for the AI Holographic Wristwatch LiPo battery.
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


class ChargingMode(Enum):
    STANDARD = auto()    # 5W USB-C standard
    FAST = auto()        # 18W USB-PD fast charge
    WIRELESS = auto()    # Qi wireless
    REVERSE = auto()     # Reverse wireless charging


class ChargingPhase(Enum):
    PRE_CHARGE = auto()   # Low-voltage recovery (<3.0V)
    CC = auto()           # Constant current bulk charge
    CV = auto()           # Constant voltage topping
    DONE = auto()         # Charge complete
    FAULT = auto()        # Fault condition


@dataclass
class ChargingStatus:
    mode: ChargingMode = ChargingMode.STANDARD
    phase: ChargingPhase = ChargingPhase.CC
    voltage_v: float = 4.0
    current_ma: float = 0.0
    power_w: float = 0.0
    target_voltage_v: float = 4.35
    temperature_c: float = 25.0
    negotiated_pd_voltage_v: float = 5.0
    negotiated_pd_current_a: float = 1.0
    efficiency_pct: float = 92.0
    time_to_full_min: float = 60.0
    is_active: bool = False


class ChargingController:
    """USB-C PD fast charging controller with thermal-aware rate adjustment."""

    THERMAL_DERATE_THRESHOLD_C = 40.0
    THERMAL_CUTOFF_C = 50.0
    CC_CURRENT_MA = 800.0
    CV_VOLTAGE_V = 4.35
    PRE_CHARGE_CURRENT_MA = 100.0
    TERMINATION_CURRENT_MA = 40.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._status = ChargingStatus()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        logger.info("ChargingController initialised")

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._charge_loop, daemon=True, name="fast-charge")
            self._thread.start()
        logger.info("ChargingController started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("ChargingController stopped")

    def negotiate_pd_profile(self, max_voltage_v: float = 9.0, max_current_a: float = 2.0) -> bool:
        """Simulate USB-PD contract negotiation with the charger."""
        with self._lock:
            # Prefer 9V/2A (18W) if available, fall back to 5V
            if max_voltage_v >= 9.0 and max_current_a >= 2.0:
                self._status.negotiated_pd_voltage_v = 9.0
                self._status.negotiated_pd_current_a = 2.0
                self._status.mode = ChargingMode.FAST
                logger.info("PD negotiated: 9V/2A (18W)")
            else:
                self._status.negotiated_pd_voltage_v = 5.0
                self._status.negotiated_pd_current_a = 1.0
                self._status.mode = ChargingMode.STANDARD
                logger.info("PD negotiated: 5V/1A (5W)")
            self._status.is_active = True
        return True

    def _read_hardware(self) -> tuple[float, float, float]:
        """Simulate reading voltage, current, and temperature from hardware."""
        v = self._status.voltage_v + random.gauss(0, 0.005)
        i = max(0.0, self._status.current_ma + random.gauss(0, 5))
        t = self._status.temperature_c + random.gauss(0, 0.3)
        return v, i, t

    def _thermal_derate_current(self, temp_c: float, base_current_ma: float) -> float:
        """Reduce charge current linearly above thermal derate threshold."""
        if temp_c >= self.THERMAL_CUTOFF_C:
            logger.warning("Thermal cutoff reached — stopping charge")
            return 0.0
        if temp_c > self.THERMAL_DERATE_THRESHOLD_C:
            factor = 1.0 - (temp_c - self.THERMAL_DERATE_THRESHOLD_C) / (
                self.THERMAL_CUTOFF_C - self.THERMAL_DERATE_THRESHOLD_C
            )
            return base_current_ma * max(factor, 0.1)
        return base_current_ma

    def _charge_loop(self) -> None:
        while self._running:
            with self._lock:
                v, i_meas, temp = self._read_hardware()
                target_i = self._select_phase_current(v)
                target_i = self._thermal_derate_current(temp, target_i)
                # Simulate voltage rise under constant current
                self._status.voltage_v = min(v + target_i * 0.000005, self.CV_VOLTAGE_V)
                self._status.current_ma = target_i
                self._status.power_w = round(self._status.voltage_v * target_i / 1000, 3)
                self._status.temperature_c = temp
                remaining_mah = max(0, (self.CV_VOLTAGE_V - self._status.voltage_v) * 5000)
                self._status.time_to_full_min = (remaining_mah / max(target_i, 1)) * 60 if target_i > 0 else 999
            time.sleep(1.0)

    def _select_phase_current(self, voltage_v: float) -> float:
        if voltage_v < 3.0:
            self._status.phase = ChargingPhase.PRE_CHARGE
            return self.PRE_CHARGE_CURRENT_MA
        if voltage_v < self.CV_VOLTAGE_V:
            self._status.phase = ChargingPhase.CC
            return self.CC_CURRENT_MA
        self._status.phase = ChargingPhase.CV
        return max(self.TERMINATION_CURRENT_MA, self.CC_CURRENT_MA * 0.1)

    def get_status(self) -> ChargingStatus:
        with self._lock:
            return ChargingStatus(**vars(self._status))


_GLOBAL_CHARGING_CONTROLLER: Optional[ChargingController] = None
_GLOBAL_CHARGING_CONTROLLER_LOCK = threading.Lock()


def get_charging_controller() -> ChargingController:
    global _GLOBAL_CHARGING_CONTROLLER
    with _GLOBAL_CHARGING_CONTROLLER_LOCK:
        if _GLOBAL_CHARGING_CONTROLLER is None:
            _GLOBAL_CHARGING_CONTROLLER = ChargingController()
    return _GLOBAL_CHARGING_CONTROLLER


def run_fast_charging_tests() -> bool:
    logger.info("=== FastCharging tests ===")
    ctrl = ChargingController()
    ok = ctrl.negotiate_pd_profile(max_voltage_v=9.0, max_current_a=2.0)
    assert ok, "PD negotiation failed"
    assert ctrl.get_status().mode == ChargingMode.FAST
    ctrl.start()
    time.sleep(0.1)
    s = ctrl.get_status()
    assert s.is_active
    ctrl.stop()
    logger.info("FastCharging tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_fast_charging_tests()
