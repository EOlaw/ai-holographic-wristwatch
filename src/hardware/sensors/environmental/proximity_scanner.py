"""
Proximity Scanner — AI Holographic Wristwatch

Detects nearby people, objects, and devices for:
- Privacy-aware hologram dimming when others are nearby
- Tap-to-share with nearby devices (NFC/UWB ranging)
- Obstacle detection for hologram projection
- Personal space awareness (COVID/social distancing mode)
- Bluetooth LE device proximity scanning
- Infrared proximity sensor for object distance
- SensorInterface compliance
"""

from __future__ import annotations

import threading
import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Deque, Dict, List, Optional

from src.core.interfaces.sensor_interface import SensorInterface, SensorReading, SensorStatus, SensorInfo
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class ProximityZone(Enum):
    INTIMATE    = "intimate"    # 0–0.45m (hand/wrist range)
    PERSONAL    = "personal"    # 0.45–1.2m (comfortable interaction)
    SOCIAL      = "social"      # 1.2–3.6m (social distance)
    PUBLIC      = "public"      # > 3.6m (public space)
    NO_OBJECT   = "no_object"   # nothing detected


class NearbyDeviceType(Enum):
    PHONE       = "phone"
    WATCH       = "watch"
    HEADPHONES  = "headphones"
    LAPTOP      = "laptop"
    BEACON      = "beacon"
    UNKNOWN     = "unknown"


@dataclass
class NearbyDevice:
    device_id: str
    device_type: NearbyDeviceType
    rssi_dbm: float
    estimated_distance_m: float
    name: str = ""
    supports_uwb: bool = False
    uwb_distance_m: Optional[float] = None


@dataclass
class ProximityReading(SensorReading):
    ir_distance_cm: float = 0.0               # infrared object distance
    proximity_zone: ProximityZone = ProximityZone.NO_OBJECT
    nearby_devices: List[NearbyDevice] = field(default_factory=list)
    nearest_person_estimate_m: float = 10.0   # BLE RSSI-based estimate
    privacy_alert: bool = False               # someone within 1m
    hologram_should_dim: bool = False
    object_in_projection_path: bool = False


_GLOBAL_PROX: Optional["ProximityScanner"] = None
_GLOBAL_PROX_LOCK = threading.Lock()


class ProximityScanner(SensorInterface):
    """
    Multi-mode proximity scanner: IR sensor + BLE RSSI + UWB ranging.
    """

    SENSOR_ID    = "environmental.proximity"
    SENSOR_TYPE  = "proximity"
    MODEL        = "VCNL4040+UWB-DW3000"
    MANUFACTURER = "Vishay/Qorvo"

    PRIVACY_DISTANCE_M = 1.0   # alert if someone within 1m

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._running = self._initialized = False
        self._read_count = 0
        self._last_reading: Optional[ProximityReading] = None
        self._known_devices: List[NearbyDevice] = []

    def initialize(self) -> bool:
        with self._lock:
            self._initialized = self._running = True
            logger.info(f"ProximityScanner {self.MODEL} initialized")
            return True

    def read(self) -> Optional[ProximityReading]:
        if not self._initialized:
            return None
        with self._lock:
            # IR proximity (0–200cm)
            ir_dist = max(0.0, random.gauss(80.0, 20.0))

            if ir_dist < 45:
                zone = ProximityZone.INTIMATE
            elif ir_dist < 120:
                zone = ProximityZone.PERSONAL
            elif ir_dist < 360:
                zone = ProximityZone.SOCIAL
            else:
                zone = ProximityZone.PUBLIC

            # Simulated BLE scan
            devices = self._simulate_ble_scan()
            nearest_person = (min((d.estimated_distance_m for d in devices), default=10.0))

            privacy_alert = nearest_person < self.PRIVACY_DISTANCE_M
            in_path = ir_dist < 30.0   # object very close

            reading = ProximityReading(
                sensor_id=self.SENSOR_ID, timestamp=time.time(),
                ir_distance_cm=ir_dist, proximity_zone=zone,
                nearby_devices=devices, nearest_person_estimate_m=nearest_person,
                privacy_alert=privacy_alert,
                hologram_should_dim=privacy_alert,
                object_in_projection_path=in_path,
                confidence=0.85,
            )
            self._last_reading = reading
            self._read_count += 1
            return reading

    def _simulate_ble_scan(self) -> List[NearbyDevice]:
        """Simulate 0–3 BLE devices within range."""
        count = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
        devices = []
        for i in range(count):
            rssi = random.gauss(-65, 10)
            # RSSI → distance: d = 10 ^ ((TxPower - RSSI) / (10 * n))
            # TxPower ≈ -59 dBm, n ≈ 2.5
            dist = 10 ** ((-59 - rssi) / 25)
            devices.append(NearbyDevice(
                device_id=f"device_{i+1:04x}",
                device_type=random.choice(list(NearbyDeviceType)),
                rssi_dbm=rssi,
                estimated_distance_m=min(10.0, dist),
            ))
        return devices

    async def stream(self) -> AsyncIterator[ProximityReading]:
        import asyncio
        while self._running:
            r = self.read()
            if r: yield r
            await asyncio.sleep(0.5)

    def calibrate(self) -> bool:
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._running = self._initialized = False

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(sensor_id=self.SENSOR_ID, sensor_type=self.SENSOR_TYPE,
                          model=self.MODEL, manufacturer=self.MANUFACTURER,
                          firmware_version="1.0.0", hardware_version="rev-A",
                          capabilities={"ir_proximity": True, "ble_scan": True,
                                        "uwb_ranging": True, "privacy_alert": True})

    def get_status(self) -> SensorStatus:
        if not self._initialized: return SensorStatus.UNINITIALIZED
        return SensorStatus.RUNNING if self._running else SensorStatus.IDLE

    def is_healthy(self) -> bool:
        return self.get_status() in (SensorStatus.RUNNING, SensorStatus.IDLE)

    def get_health_report(self) -> Dict:
        return {"status": self.get_status().value, "read_count": self._read_count,
                "last_ir_cm": self._last_reading.ir_distance_cm if self._last_reading else None}

    def read_sync(self) -> Optional[ProximityReading]:
        return self.read()


def get_proximity_scanner() -> ProximityScanner:
    global _GLOBAL_PROX
    with _GLOBAL_PROX_LOCK:
        if _GLOBAL_PROX is None:
            _GLOBAL_PROX = ProximityScanner()
        return _GLOBAL_PROX


def run_proximity_scanner_tests() -> bool:
    ps = ProximityScanner()
    assert ps.initialize()
    r = ps.read()
    assert r is not None and r.ir_distance_cm >= 0
    ps.shutdown()
    logger.info("ProximityScanner tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_proximity_scanner_tests()
