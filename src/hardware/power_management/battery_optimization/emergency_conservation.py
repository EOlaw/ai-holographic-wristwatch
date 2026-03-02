"""Emergency battery conservation mode for the AI Holographic Wristwatch.

Activates when SoC drops below critical thresholds, shutting down non-essential
subsystems and preserving core timekeeping and emergency communication.
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


class ConservationLevel(Enum):
    NONE = auto()          # Normal operation
    LEVEL_1 = auto()       # SoC < 20% — reduce brightness, disable WiFi
    LEVEL_2 = auto()       # SoC < 10% — disable hologram, BLE-only
    LEVEL_3 = auto()       # SoC <  5% — disable display, RTC + BLE ping only
    EMERGENCY = auto()     # SoC <  2% — save state and prepare shutdown


@dataclass
class ConservationStatus:
    level: ConservationLevel = ConservationLevel.NONE
    soc_pct: float = 80.0
    estimated_survival_min: float = 0.0
    disabled_subsystems: list[str] = field(default_factory=list)
    active_subsystems: list[str] = field(default_factory=list)
    power_saved_mw: float = 0.0
    timestamp: float = field(default_factory=time.time)


# Subsystems to disable per level (cumulative)
_DISABLE_AT_LEVEL: dict[ConservationLevel, list[str]] = {
    ConservationLevel.LEVEL_1: ["hologram", "wifi", "gps", "speaker"],
    ConservationLevel.LEVEL_2: ["display", "cellular", "nfc", "heart_rate"],
    ConservationLevel.LEVEL_3: ["imu", "microphone"],
    ConservationLevel.EMERGENCY: ["all_except_rtc"],
}

_ALL_SUBSYSTEMS = ["hologram", "wifi", "gps", "speaker", "display", "cellular",
                   "nfc", "heart_rate", "imu", "microphone", "ble", "rtc"]


class EmergencyConservation:
    """Manages staged emergency power conservation as battery depletes."""

    SOC_THRESHOLDS = {
        ConservationLevel.EMERGENCY: 2.0,
        ConservationLevel.LEVEL_3:   5.0,
        ConservationLevel.LEVEL_2:  10.0,
        ConservationLevel.LEVEL_1:  20.0,
        ConservationLevel.NONE:    100.0,
    }
    POWER_SAVINGS_MW = {
        ConservationLevel.NONE:      0.0,
        ConservationLevel.LEVEL_1:  210.0,
        ConservationLevel.LEVEL_2:   85.0,
        ConservationLevel.LEVEL_3:   20.0,
        ConservationLevel.EMERGENCY: 10.0,
    }

    def __init__(self, shutdown_callback: Optional[Callable[[], None]] = None) -> None:
        self._lock = threading.RLock()
        self._soc_pct = 80.0
        self._level = ConservationLevel.NONE
        self._status = ConservationStatus()
        self._shutdown_cb = shutdown_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        logger.info("EmergencyConservation initialised")

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._conservation_loop, daemon=True, name="emergency-conserve"
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def update_soc(self, soc_pct: float) -> None:
        with self._lock:
            self._soc_pct = max(0.0, min(100.0, soc_pct))

    def _select_level(self, soc: float) -> ConservationLevel:
        for level in (
            ConservationLevel.EMERGENCY,
            ConservationLevel.LEVEL_3,
            ConservationLevel.LEVEL_2,
            ConservationLevel.LEVEL_1,
        ):
            if soc <= self.SOC_THRESHOLDS[level]:
                return level
        return ConservationLevel.NONE

    def _disabled_subsystems(self, level: ConservationLevel) -> list[str]:
        disabled: set[str] = set()
        for lvl in (ConservationLevel.LEVEL_1, ConservationLevel.LEVEL_2,
                    ConservationLevel.LEVEL_3, ConservationLevel.EMERGENCY):
            disabled.update(_DISABLE_AT_LEVEL[lvl])
            if lvl == level:
                break
        return sorted(disabled)

    def _conservation_loop(self) -> None:
        while self._running:
            with self._lock:
                new_level = self._select_level(self._soc_pct)
                if new_level != self._level:
                    logger.warning(
                        "Conservation level: %s -> %s (SoC=%.1f%%)",
                        self._level.name, new_level.name, self._soc_pct
                    )
                    self._level = new_level
                    if new_level == ConservationLevel.EMERGENCY and self._shutdown_cb:
                        self._shutdown_cb()

                disabled = self._disabled_subsystems(self._level)
                active = [s for s in _ALL_SUBSYSTEMS if s not in disabled]
                saved = sum(self.POWER_SAVINGS_MW[l] for l in list(ConservationLevel)
                            if list(ConservationLevel).index(l) <=
                               list(ConservationLevel).index(self._level)
                            and l != ConservationLevel.NONE)
                # Rough survival estimate: remaining capacity / residual draw
                residual_mw = max(5.0, 50.0 - saved)
                remaining_mwh = self._soc_pct / 100.0 * 300.0 * 3.7
                survival_min = (remaining_mwh / residual_mw) * 60.0
                self._status = ConservationStatus(
                    level=self._level,
                    soc_pct=self._soc_pct,
                    estimated_survival_min=round(survival_min, 1),
                    disabled_subsystems=disabled,
                    active_subsystems=active,
                    power_saved_mw=round(saved, 1),
                    timestamp=time.time(),
                )
            time.sleep(2.0)

    def get_status(self) -> ConservationStatus:
        with self._lock:
            s = self._status
            return ConservationStatus(
                level=s.level, soc_pct=s.soc_pct,
                estimated_survival_min=s.estimated_survival_min,
                disabled_subsystems=list(s.disabled_subsystems),
                active_subsystems=list(s.active_subsystems),
                power_saved_mw=s.power_saved_mw,
                timestamp=s.timestamp,
            )


_GLOBAL_EMERGENCY_CONSERVATION: Optional[EmergencyConservation] = None
_GLOBAL_EMERGENCY_CONSERVATION_LOCK = threading.Lock()


def get_emergency_conservation() -> EmergencyConservation:
    global _GLOBAL_EMERGENCY_CONSERVATION
    with _GLOBAL_EMERGENCY_CONSERVATION_LOCK:
        if _GLOBAL_EMERGENCY_CONSERVATION is None:
            _GLOBAL_EMERGENCY_CONSERVATION = EmergencyConservation()
    return _GLOBAL_EMERGENCY_CONSERVATION


def run_emergency_conservation_tests() -> bool:
    logger.info("=== EmergencyConservation tests ===")
    ec = EmergencyConservation()
    ec.update_soc(8.0)
    ec.start()
    time.sleep(0.1)
    s = ec.get_status()
    assert s.level in (ConservationLevel.LEVEL_2, ConservationLevel.LEVEL_1)
    assert s.power_saved_mw > 0
    ec.update_soc(3.0)
    time.sleep(0.1)
    ec.stop()
    logger.info("EmergencyConservation tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_emergency_conservation_tests()
