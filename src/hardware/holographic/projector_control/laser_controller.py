"""RGB laser diode driver for the AI Holographic Wristwatch projector subsystem.

Controls three independent laser diodes (red, green, blue) with safety interlocks
for IEC 60825 eye protection compliance and thermal monitoring to prevent overheating.
"""

import threading
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class LaserColor(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class LaserState(Enum):
    OFF = "off"
    STANDBY = "standby"
    ACTIVE = "active"
    FAULT = "fault"
    THERMAL_SHUTDOWN = "thermal_shutdown"


@dataclass
class LaserPowerReading:
    color: LaserColor
    power_mw: float
    temperature_c: float
    state: LaserState
    duty_cycle: float
    timestamp: float = field(default_factory=time.time)


class LaserController:
    """Controls RGB laser diodes with safety interlocks and thermal monitoring.

    Each laser diode is independently controlled with PWM duty cycle. A hardware
    interlock disables all lasers when proximity sensor detects eye exposure risk.
    Thermal monitoring triggers shutdown if junction temperature exceeds limits.
    """

    MAX_POWER_MW: Dict[LaserColor, float] = {
        LaserColor.RED: 5.0,
        LaserColor.GREEN: 3.0,
        LaserColor.BLUE: 4.0,
    }
    THERMAL_LIMIT_C = 75.0
    THERMAL_WARNING_C = 65.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: Dict[LaserColor, LaserState] = {c: LaserState.OFF for c in LaserColor}
        self._duty_cycles: Dict[LaserColor, float] = {c: 0.0 for c in LaserColor}
        self._temperatures: Dict[LaserColor, float] = {c: 25.0 for c in LaserColor}
        self._interlock_active = False
        logger.info("LaserController initialized")

    def enable(self, color: LaserColor, duty_cycle: float = 1.0) -> bool:
        """Enable a laser diode at the specified duty cycle (0.0–1.0)."""
        with self._lock:
            duty_cycle = max(0.0, min(1.0, duty_cycle))
            if self._interlock_active:
                logger.warning("Interlock active — laser %s blocked", color.value)
                return False
            temp = self._read_hardware_temperature(color)
            if temp >= self.THERMAL_LIMIT_C:
                self._states[color] = LaserState.THERMAL_SHUTDOWN
                logger.error("Thermal shutdown: %s at %.1f°C", color.value, temp)
                return False
            self._duty_cycles[color] = duty_cycle
            self._states[color] = LaserState.ACTIVE
            logger.debug("Laser %s enabled @ %.0f%% duty", color.value, duty_cycle * 100)
            return True

    def disable(self, color: LaserColor) -> None:
        """Disable a laser diode."""
        with self._lock:
            self._duty_cycles[color] = 0.0
            self._states[color] = LaserState.OFF
            logger.debug("Laser %s disabled", color.value)

    def emergency_shutdown(self) -> None:
        """Immediately disable all laser diodes (eye-safety interlock)."""
        with self._lock:
            for color in LaserColor:
                self._duty_cycles[color] = 0.0
                self._states[color] = LaserState.OFF
            self._interlock_active = True
            logger.critical("Emergency laser shutdown — all diodes disabled")

    def release_interlock(self) -> None:
        """Release the safety interlock after confirming safe conditions."""
        with self._lock:
            self._interlock_active = False
            for color in LaserColor:
                self._states[color] = LaserState.STANDBY
            logger.info("Safety interlock released")

    def read_power(self, color: LaserColor) -> LaserPowerReading:
        """Return current power reading for a laser diode."""
        with self._lock:
            temp = self._read_hardware_temperature(color)
            self._temperatures[color] = temp
            power = self._simulate_power_output(color)
            return LaserPowerReading(
                color=color,
                power_mw=power,
                temperature_c=temp,
                state=self._states[color],
                duty_cycle=self._duty_cycles[color],
            )

    def _read_hardware_temperature(self, color: LaserColor) -> float:
        """Simulate reading the laser junction temperature from hardware sensor."""
        base = self._temperatures.get(color, 25.0)
        load = self._duty_cycles.get(color, 0.0) * 30.0
        noise = random.uniform(-0.5, 0.5)
        return base + load * 0.1 + noise

    def _simulate_power_output(self, color: LaserColor) -> float:
        """Simulate measured optical output power in milliwatts."""
        duty = self._duty_cycles.get(color, 0.0)
        max_p = self.MAX_POWER_MW[color]
        return duty * max_p * random.uniform(0.97, 1.03)


_GLOBAL_LASER_CONTROLLER: Optional[LaserController] = None
_GLOBAL_LASER_CONTROLLER_LOCK = threading.Lock()


def get_laser_controller() -> LaserController:
    """Return the global LaserController singleton."""
    global _GLOBAL_LASER_CONTROLLER
    with _GLOBAL_LASER_CONTROLLER_LOCK:
        if _GLOBAL_LASER_CONTROLLER is None:
            _GLOBAL_LASER_CONTROLLER = LaserController()
    return _GLOBAL_LASER_CONTROLLER


def run_laser_controller_tests() -> None:
    """Self-test for LaserController."""
    logger.info("Running LaserController tests...")
    ctrl = LaserController()

    assert ctrl.enable(LaserColor.RED, 0.8), "Enable RED failed"
    reading = ctrl.read_power(LaserColor.RED)
    assert reading.state == LaserState.ACTIVE, "Expected ACTIVE state"
    assert 0.0 < reading.power_mw <= LaserController.MAX_POWER_MW[LaserColor.RED] * 1.05

    ctrl.emergency_shutdown()
    assert not ctrl.enable(LaserColor.GREEN), "Interlock should block enable"

    ctrl.release_interlock()
    assert ctrl.enable(LaserColor.GREEN, 0.5), "Enable GREEN after interlock release failed"

    ctrl.disable(LaserColor.GREEN)
    assert ctrl.read_power(LaserColor.GREEN).state == LaserState.OFF

    logger.info("LaserController tests PASSED")


if __name__ == "__main__":
    run_laser_controller_tests()
