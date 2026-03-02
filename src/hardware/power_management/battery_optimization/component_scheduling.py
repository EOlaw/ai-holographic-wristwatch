"""Component scheduling and power gating for battery optimization.

Manages duty-cycling and power gating of individual hardware subsystems
(display, radios, sensors, holographic projector) based on usage context.
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


class ComponentID(Enum):
    DISPLAY = "display"
    HOLOGRAM = "hologram"
    BLE = "ble"
    WIFI = "wifi"
    NFC = "nfc"
    CELLULAR = "cellular"
    GPS = "gps"
    HEART_RATE = "heart_rate"
    IMU = "imu"
    MICROPHONE = "microphone"
    SPEAKER = "speaker"


class ComponentState(Enum):
    ON = auto()
    OFF = auto()
    DUTY_CYCLED = auto()   # Periodic wake/sleep
    SUSPENDED = auto()     # Retained register state, clock stopped


@dataclass
class ComponentSchedule:
    component: ComponentID
    state: ComponentState = ComponentState.ON
    duty_cycle_pct: float = 100.0   # Active percentage when duty-cycled
    wake_period_ms: int = 1000
    power_mw: float = 0.0
    last_active_ts: float = field(default_factory=time.time)


class ComponentScheduler:
    """Power-gates and duty-cycles hardware components to conserve energy."""

    # Nominal power draw per component (mW)
    COMPONENT_POWER_MW: dict[ComponentID, float] = {
        ComponentID.DISPLAY:    80.0,
        ComponentID.HOLOGRAM:  150.0,
        ComponentID.BLE:         6.0,
        ComponentID.WIFI:       80.0,
        ComponentID.NFC:        10.0,
        ComponentID.CELLULAR:  120.0,
        ComponentID.GPS:        25.0,
        ComponentID.HEART_RATE:  1.5,
        ComponentID.IMU:         0.8,
        ComponentID.MICROPHONE:  1.2,
        ComponentID.SPEAKER:    20.0,
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._schedules: dict[ComponentID, ComponentSchedule] = {
            cid: ComponentSchedule(component=cid, power_mw=self.COMPONENT_POWER_MW[cid])
            for cid in ComponentID
        }
        self._running = False
        self._thread: Optional[threading.Thread] = None
        logger.info("ComponentScheduler initialised with %d components", len(ComponentID))

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._scheduler_loop, daemon=True, name="comp-sched")
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def set_component_state(
        self,
        component: ComponentID,
        state: ComponentState,
        duty_cycle_pct: float = 100.0,
    ) -> None:
        with self._lock:
            sched = self._schedules[component]
            old = sched.state
            sched.state = state
            sched.duty_cycle_pct = duty_cycle_pct
            if state == ComponentState.OFF:
                sched.power_mw = 0.0
            elif state == ComponentState.DUTY_CYCLED:
                sched.power_mw = self.COMPONENT_POWER_MW[component] * duty_cycle_pct / 100.0
            else:
                sched.power_mw = self.COMPONENT_POWER_MW[component]
            logger.debug("Component %s: %s -> %s", component.value, old.name, state.name)

    def apply_power_profile(self, active_components: set[ComponentID]) -> None:
        """Gate off all components not in the active set."""
        with self._lock:
            for cid in ComponentID:
                if cid in active_components:
                    self.set_component_state(cid, ComponentState.ON)
                else:
                    self.set_component_state(cid, ComponentState.OFF)

    def get_total_power_mw(self) -> float:
        with self._lock:
            return sum(s.power_mw for s in self._schedules.values())

    def get_schedule(self, component: ComponentID) -> ComponentSchedule:
        with self._lock:
            s = self._schedules[component]
            return ComponentSchedule(**vars(s))

    def _scheduler_loop(self) -> None:
        while self._running:
            with self._lock:
                for cid, sched in self._schedules.items():
                    if sched.state == ComponentState.DUTY_CYCLED:
                        # Simulate periodic wakeup
                        if random.random() < sched.duty_cycle_pct / 100.0:
                            sched.last_active_ts = time.time()
            time.sleep(0.5)


_GLOBAL_COMPONENT_SCHEDULER: Optional[ComponentScheduler] = None
_GLOBAL_COMPONENT_SCHEDULER_LOCK = threading.Lock()


def get_component_scheduler() -> ComponentScheduler:
    global _GLOBAL_COMPONENT_SCHEDULER
    with _GLOBAL_COMPONENT_SCHEDULER_LOCK:
        if _GLOBAL_COMPONENT_SCHEDULER is None:
            _GLOBAL_COMPONENT_SCHEDULER = ComponentScheduler()
    return _GLOBAL_COMPONENT_SCHEDULER


def run_component_scheduling_tests() -> bool:
    logger.info("=== ComponentScheduling tests ===")
    sched = ComponentScheduler()
    sched.start()
    sched.set_component_state(ComponentID.WIFI, ComponentState.OFF)
    assert sched.get_schedule(ComponentID.WIFI).power_mw == 0.0
    sched.set_component_state(ComponentID.BLE, ComponentState.DUTY_CYCLED, duty_cycle_pct=20.0)
    ble = sched.get_schedule(ComponentID.BLE)
    assert abs(ble.power_mw - ComponentScheduler.COMPONENT_POWER_MW[ComponentID.BLE] * 0.2) < 0.01
    total = sched.get_total_power_mw()
    assert total >= 0
    sched.stop()
    logger.info("ComponentScheduling tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_component_scheduling_tests()
