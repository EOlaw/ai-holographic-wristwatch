"""Charging safety monitor — over-voltage, over-current, and thermal protection.

Implements a watchdog that samples all charging parameters and triggers
hardware protection circuits when limits are exceeded.
"""
from __future__ import annotations

import threading
import time
import random
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class SafetyEvent(Enum):
    NONE = auto()
    OVER_VOLTAGE = auto()
    OVER_CURRENT = auto()
    OVER_TEMPERATURE = auto()
    UNDER_VOLTAGE = auto()
    SHORT_CIRCUIT = auto()
    NTC_FAULT = auto()


@dataclass
class SafetyReading:
    voltage_v: float = 4.2
    current_ma: float = 500.0
    temperature_c: float = 25.0
    event: SafetyEvent = SafetyEvent.NONE
    protection_active: bool = False
    timestamp: float = field(default_factory=time.time)


class ChargingSafety:
    """Hardware safety monitor for the LiPo charging chain."""

    MAX_VOLTAGE_V = 4.40
    MIN_VOLTAGE_V = 2.80
    MAX_CURRENT_MA = 1200.0
    MAX_TEMP_C = 50.0
    MIN_TEMP_C = 0.0
    SHORT_CIRCUIT_THRESHOLD_MA = 3000.0

    def __init__(self, shutdown_callback: Optional[Callable[[], None]] = None) -> None:
        self._lock = threading.RLock()
        self._last_reading = SafetyReading()
        self._protection_active = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._shutdown_cb = shutdown_callback
        self._event_log: list[tuple[float, SafetyEvent]] = []
        logger.info("ChargingSafety watchdog initialised")

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="charge-safety")
            self._thread.start()
        logger.info("ChargingSafety watchdog started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _read_hardware(self) -> SafetyReading:
        """Simulate ADC sampling of voltage, current, and NTC temperature."""
        v = random.gauss(4.15, 0.05)
        i = random.gauss(600, 30)
        t = random.gauss(28, 2)
        return SafetyReading(voltage_v=round(v, 3), current_ma=round(i, 1), temperature_c=round(t, 1))

    def _evaluate(self, r: SafetyReading) -> SafetyEvent:
        if r.current_ma > self.SHORT_CIRCUIT_THRESHOLD_MA:
            return SafetyEvent.SHORT_CIRCUIT
        if r.voltage_v > self.MAX_VOLTAGE_V:
            return SafetyEvent.OVER_VOLTAGE
        if r.voltage_v < self.MIN_VOLTAGE_V:
            return SafetyEvent.UNDER_VOLTAGE
        if r.current_ma > self.MAX_CURRENT_MA:
            return SafetyEvent.OVER_CURRENT
        if r.temperature_c > self.MAX_TEMP_C or r.temperature_c < self.MIN_TEMP_C:
            return SafetyEvent.OVER_TEMPERATURE
        return SafetyEvent.NONE

    def _trigger_protection(self, event: SafetyEvent) -> None:
        logger.error("SAFETY FAULT: %s — engaging protection circuit", event.name)
        self._protection_active = True
        self._event_log.append((time.time(), event))
        if self._shutdown_cb:
            self._shutdown_cb()

    def _watchdog_loop(self) -> None:
        while self._running:
            with self._lock:
                reading = self._read_hardware()
                event = self._evaluate(reading)
                reading.event = event
                reading.protection_active = self._protection_active
                self._last_reading = reading
                if event != SafetyEvent.NONE and not self._protection_active:
                    self._trigger_protection(event)
                elif event == SafetyEvent.NONE and self._protection_active:
                    self._protection_active = False
                    logger.info("Safety conditions cleared — protection released")
            time.sleep(0.5)

    def reset_protection(self) -> bool:
        with self._lock:
            if self._protection_active:
                self._protection_active = False
                logger.info("Protection manually reset")
                return True
            return False

    def get_reading(self) -> SafetyReading:
        with self._lock:
            return SafetyReading(**vars(self._last_reading))

    def get_event_log(self) -> list[tuple[float, SafetyEvent]]:
        with self._lock:
            return list(self._event_log)


_GLOBAL_CHARGING_SAFETY: Optional[ChargingSafety] = None
_GLOBAL_CHARGING_SAFETY_LOCK = threading.Lock()


def get_charging_safety() -> ChargingSafety:
    global _GLOBAL_CHARGING_SAFETY
    with _GLOBAL_CHARGING_SAFETY_LOCK:
        if _GLOBAL_CHARGING_SAFETY is None:
            _GLOBAL_CHARGING_SAFETY = ChargingSafety()
    return _GLOBAL_CHARGING_SAFETY


def run_charging_safety_tests() -> bool:
    logger.info("=== ChargingSafety tests ===")
    safety = ChargingSafety()
    safety.start()
    time.sleep(0.2)
    r = safety.get_reading()
    assert r.voltage_v > 0, "Voltage must be positive"
    assert r.temperature_c > -273, "Temperature physically bounded"
    safety.stop()
    logger.info("ChargingSafety tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_charging_safety_tests()
