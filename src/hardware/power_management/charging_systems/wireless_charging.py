"""Qi wireless charging receiver for the AI Holographic Wristwatch.

Manages Qi coil tuning, foreign-object detection (FOD), power transfer
negotiation, and alignment feedback for wireless charging pad compatibility.
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


class WirelessChargingState(Enum):
    IDLE = auto()           # No pad detected
    PING = auto()           # Sending presence ping
    IDENTIFICATION = auto() # Exchanging ID/config packets
    NEGOTIATION = auto()    # Power class negotiation
    TRANSFER = auto()       # Active power transfer
    FOD_DETECTED = auto()   # Foreign object detected — suspended
    FAULT = auto()          # Hardware fault


@dataclass
class WirelessStatus:
    state: WirelessChargingState = WirelessChargingState.IDLE
    received_power_mw: float = 0.0
    coil_voltage_mv: float = 0.0
    coil_current_ma: float = 0.0
    frequency_khz: float = 100.0     # Qi operating frequency (100-205 kHz)
    alignment_pct: float = 0.0       # 0=no alignment, 100=perfect
    fod_score: float = 0.0           # Higher = more likely FOD present
    temperature_c: float = 25.0
    efficiency_pct: float = 0.0
    is_reverse: bool = False          # True when acting as transmitter
    timestamp: float = field(default_factory=time.time)


class WirelessChargingReceiver:
    """Qi-compliant wireless charging receiver with FOD and coil management."""

    QI_FREQ_KHZ = 100.0
    MAX_POWER_MW = 5000.0          # 5W Qi baseline; up to 15W Qi EPP
    FOD_THRESHOLD = 0.65           # FOD score above this suspends charging
    ALIGNMENT_THRESHOLD_PCT = 30.0 # Minimum alignment to start transfer
    COIL_RESISTANCE_OHM = 0.3

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._status = WirelessStatus()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        logger.info("WirelessChargingReceiver initialised")

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._receiver_loop, daemon=True, name="wireless-charge"
            )
            self._thread.start()
        logger.info("WirelessChargingReceiver started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _simulate_alignment(self) -> float:
        """Simulate coil alignment using NFC/RSSI proxy — random walk."""
        current = self._status.alignment_pct
        delta = random.gauss(0, 5)
        return max(0.0, min(100.0, current + delta))

    def _simulate_fod(self, alignment: float) -> float:
        """Foreign-object detection score: lower alignment increases FOD risk."""
        base = random.uniform(0.05, 0.15)
        if alignment < 40:
            base += random.uniform(0, 0.4)
        return round(min(base, 1.0), 3)

    def _simulate_transfer(self, alignment: float, fod_score: float) -> tuple[float, float, float]:
        """Compute received power, voltage, and efficiency given alignment and FOD."""
        if fod_score > self.FOD_THRESHOLD or alignment < self.ALIGNMENT_THRESHOLD_PCT:
            return 0.0, 0.0, 0.0
        alignment_factor = alignment / 100.0
        power_mw = self.MAX_POWER_MW * alignment_factor * random.uniform(0.85, 1.0)
        efficiency = (72.0 + 20.0 * alignment_factor) * random.uniform(0.95, 1.05)
        voltage_mv = 5000.0 * alignment_factor + random.gauss(0, 50)
        return round(power_mw, 1), round(voltage_mv, 1), round(min(efficiency, 92.0), 1)

    def _receiver_loop(self) -> None:
        while self._running:
            with self._lock:
                alignment = self._simulate_alignment()
                fod = self._simulate_fod(alignment)
                power_mw, voltage_mv, efficiency = self._simulate_transfer(alignment, fod)
                current_ma = (power_mw / voltage_mv * 1000) if voltage_mv > 0 else 0.0
                temp = self._status.temperature_c + random.gauss(0, 0.2)

                if fod > self.FOD_THRESHOLD:
                    state = WirelessChargingState.FOD_DETECTED
                    logger.warning("FOD detected (score=%.2f) — suspending transfer", fod)
                elif alignment < self.ALIGNMENT_THRESHOLD_PCT:
                    state = WirelessChargingState.PING
                elif power_mw > 0:
                    state = WirelessChargingState.TRANSFER
                else:
                    state = WirelessChargingState.IDLE

                self._status = WirelessStatus(
                    state=state,
                    received_power_mw=power_mw,
                    coil_voltage_mv=voltage_mv,
                    coil_current_ma=round(current_ma, 1),
                    frequency_khz=self.QI_FREQ_KHZ + random.gauss(0, 0.5),
                    alignment_pct=round(alignment, 1),
                    fod_score=fod,
                    temperature_c=round(temp, 1),
                    efficiency_pct=efficiency,
                    timestamp=time.time(),
                )
            time.sleep(1.0)

    def enable_reverse_charging(self, enable: bool) -> bool:
        """Switch coil to transmitter mode for reverse wireless charging."""
        with self._lock:
            self._status.is_reverse = enable
            logger.info("Reverse wireless charging: %s", "ENABLED" if enable else "DISABLED")
        return True

    def get_status(self) -> WirelessStatus:
        with self._lock:
            return WirelessStatus(**vars(self._status))


_GLOBAL_WIRELESS_CHARGING_RECEIVER: Optional[WirelessChargingReceiver] = None
_GLOBAL_WIRELESS_CHARGING_RECEIVER_LOCK = threading.Lock()


def get_wireless_charging_receiver() -> WirelessChargingReceiver:
    global _GLOBAL_WIRELESS_CHARGING_RECEIVER
    with _GLOBAL_WIRELESS_CHARGING_RECEIVER_LOCK:
        if _GLOBAL_WIRELESS_CHARGING_RECEIVER is None:
            _GLOBAL_WIRELESS_CHARGING_RECEIVER = WirelessChargingReceiver()
    return _GLOBAL_WIRELESS_CHARGING_RECEIVER


def run_wireless_charging_tests() -> bool:
    logger.info("=== WirelessCharging tests ===")
    rcvr = WirelessChargingReceiver()
    rcvr.start()
    time.sleep(0.15)
    s = rcvr.get_status()
    assert s.fod_score >= 0
    assert s.alignment_pct >= 0
    assert rcvr.enable_reverse_charging(True)
    assert rcvr.get_status().is_reverse
    rcvr.stop()
    logger.info("WirelessCharging tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_wireless_charging_tests()
