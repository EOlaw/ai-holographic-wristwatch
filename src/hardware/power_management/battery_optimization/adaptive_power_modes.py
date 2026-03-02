"""Adaptive power mode controller using Dynamic Voltage and Frequency Scaling.

Selects the optimal SoC-based power mode and applies DVFS settings to CPU,
GPU, and radio subsystems to balance performance and battery life.
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


class PowerMode(Enum):
    ACTIVE = auto()       # Full performance — hologram rendering, all radios on
    BALANCED = auto()     # Moderate DVFS, selective radio duty-cycling
    LOW_POWER = auto()    # Reduced clock, BLE-only, display dimmed
    ULTRA_SAVE = auto()   # Minimal clock, no display, BLE-only heartbeat
    SLEEP = auto()        # Deep sleep — RTC + wakeup interrupt only


@dataclass
class PowerBudget:
    mode: PowerMode = PowerMode.BALANCED
    cpu_freq_mhz: float = 800.0
    cpu_voltage_mv: float = 1050.0
    gpu_freq_mhz: float = 400.0
    display_brightness_pct: float = 80.0
    radio_active: bool = True
    hologram_enabled: bool = True
    estimated_power_mw: float = 250.0
    target_runtime_h: float = 8.0


# DVFS operating points: (mode, cpu_mhz, cpu_mv, gpu_mhz, display_pct, radio, holo, est_mw, runtime_h)
_DVFS_TABLE: dict[PowerMode, PowerBudget] = {
    PowerMode.ACTIVE:     PowerBudget(PowerMode.ACTIVE,     1200, 1150, 600, 100, True,  True,  480, 4),
    PowerMode.BALANCED:   PowerBudget(PowerMode.BALANCED,    800, 1050, 400,  80, True,  True,  250, 8),
    PowerMode.LOW_POWER:  PowerBudget(PowerMode.LOW_POWER,   400,  950, 200,  40, True,  False, 120, 16),
    PowerMode.ULTRA_SAVE: PowerBudget(PowerMode.ULTRA_SAVE,  200,  850,   0,  10, False, False,  40, 48),
    PowerMode.SLEEP:      PowerBudget(PowerMode.SLEEP,        50,  800,   0,   0, False, False,   5, 384),
}


class AdaptivePowerModes:
    """Adaptive power mode manager with SoC-driven automatic transitions."""

    SOC_THRESHOLDS = {
        PowerMode.SLEEP:      5,
        PowerMode.ULTRA_SAVE: 10,
        PowerMode.LOW_POWER:  25,
        PowerMode.BALANCED:   50,
        PowerMode.ACTIVE:     100,
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current_mode = PowerMode.BALANCED
        self._budget = _DVFS_TABLE[PowerMode.BALANCED]
        self._soc_pct = 80.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        logger.info("AdaptivePowerModes initialised in BALANCED mode")

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="power-modes")
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def set_mode(self, mode: PowerMode) -> PowerBudget:
        with self._lock:
            if self._current_mode != mode:
                logger.info("Power mode: %s -> %s", self._current_mode.name, mode.name)
            self._current_mode = mode
            self._budget = _DVFS_TABLE[mode]
            return PowerBudget(**vars(self._budget))

    def update_soc(self, soc_pct: float) -> None:
        with self._lock:
            self._soc_pct = max(0.0, min(100.0, soc_pct))

    def _auto_select_mode(self, soc: float) -> PowerMode:
        if soc <= self.SOC_THRESHOLDS[PowerMode.SLEEP]:
            return PowerMode.SLEEP
        if soc <= self.SOC_THRESHOLDS[PowerMode.ULTRA_SAVE]:
            return PowerMode.ULTRA_SAVE
        if soc <= self.SOC_THRESHOLDS[PowerMode.LOW_POWER]:
            return PowerMode.LOW_POWER
        if soc <= self.SOC_THRESHOLDS[PowerMode.BALANCED]:
            return PowerMode.BALANCED
        return PowerMode.ACTIVE

    def _monitor_loop(self) -> None:
        while self._running:
            with self._lock:
                suggested = self._auto_select_mode(self._soc_pct)
                if suggested != self._current_mode:
                    self.set_mode(suggested)
            time.sleep(5.0)

    def get_budget(self) -> PowerBudget:
        with self._lock:
            return PowerBudget(**vars(self._budget))


_GLOBAL_ADAPTIVE_POWER_MODES: Optional[AdaptivePowerModes] = None
_GLOBAL_ADAPTIVE_POWER_MODES_LOCK = threading.Lock()


def get_adaptive_power_modes() -> AdaptivePowerModes:
    global _GLOBAL_ADAPTIVE_POWER_MODES
    with _GLOBAL_ADAPTIVE_POWER_MODES_LOCK:
        if _GLOBAL_ADAPTIVE_POWER_MODES is None:
            _GLOBAL_ADAPTIVE_POWER_MODES = AdaptivePowerModes()
    return _GLOBAL_ADAPTIVE_POWER_MODES


def run_adaptive_power_modes_tests() -> bool:
    logger.info("=== AdaptivePowerModes tests ===")
    apm = AdaptivePowerModes()
    budget = apm.set_mode(PowerMode.LOW_POWER)
    assert budget.mode == PowerMode.LOW_POWER
    assert budget.cpu_freq_mhz < 500
    apm.update_soc(8.0)
    apm.start()
    time.sleep(0.1)
    apm.stop()
    assert apm.get_budget().mode in (PowerMode.SLEEP, PowerMode.ULTRA_SAVE)
    logger.info("AdaptivePowerModes tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_adaptive_power_modes_tests()
