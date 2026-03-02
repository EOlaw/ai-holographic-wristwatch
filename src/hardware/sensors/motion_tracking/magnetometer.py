"""
3-Axis Magnetometer Driver — AI Holographic Wristwatch

Implements a MEMS magnetometer (e.g., LIS3MDL) providing:
- Magnetic field strength on X/Y/Z axes (microtesla)
- Compass heading (azimuth) calculation
- Hard-iron and soft-iron calibration
- Magnetic anomaly detection
- Integration with gyroscope for tilt-compensated compass
- SensorInterface compliance
"""

from __future__ import annotations

import math
import threading
import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Deque, Dict, List, Optional, Tuple

from src.core.interfaces.sensor_interface import SensorInterface, SensorReading, SensorStatus, SensorInfo
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

# Earth's magnetic field reference (typical mid-latitude values)
EARTH_FIELD_UT = 50.0  # microtesla


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MagRange(Enum):
    """Full-scale magnetic range in gauss (1G = 100µT)."""
    G4  = 4    # ±4 gauss = ±400 µT
    G8  = 8
    G12 = 12
    G16 = 16


class MagOutputDataRate(Enum):
    """Output data rate in Hz."""
    ODR_0_625 = 0.625
    ODR_1_25  = 1.25
    ODR_2_5   = 2.5
    ODR_5     = 5.0
    ODR_10    = 10.0
    ODR_20    = 20.0
    ODR_40    = 40.0
    ODR_80    = 80.0
    ODR_155   = 155.0
    ODR_300   = 300.0
    ODR_560   = 560.0
    ODR_1000  = 1000.0


class CompassDirection(Enum):
    """16-point compass rose."""
    N   = "N"
    NNE = "NNE"
    NE  = "NE"
    ENE = "ENE"
    E   = "E"
    ESE = "ESE"
    SE  = "SE"
    SSE = "SSE"
    S   = "S"
    SSW = "SSW"
    SW  = "SW"
    WSW = "WSW"
    W   = "W"
    WNW = "WNW"
    NW  = "NW"
    NNW = "NNW"


class MagneticAnomalyType(Enum):
    NONE          = "none"
    HARD_IRON     = "hard_iron"     # constant offset — nearby permanent magnet
    SOFT_IRON     = "soft_iron"     # scale distortion — ferromagnetic material
    INTERFERENCE  = "interference"  # transient EM interference


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class MagneticVector:
    """Magnetic field in microtesla."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "MagneticVector":
        m = self.magnitude()
        if m < 1e-9:
            return MagneticVector(0.0, 0.0, 0.0)
        return MagneticVector(self.x / m, self.y / m, self.z / m)


@dataclass
class MagCalibration:
    """Hard-iron and soft-iron correction parameters."""
    hard_iron_offset: MagneticVector = field(default_factory=MagneticVector)
    soft_iron_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    calibrated: bool = False
    calibration_timestamp: float = 0.0
    sample_count: int = 0


@dataclass
class MagnetometerReading(SensorReading):
    """Full magnetometer measurement snapshot."""
    raw_field: MagneticVector = field(default_factory=MagneticVector)
    calibrated_field: MagneticVector = field(default_factory=MagneticVector)
    field_magnitude_ut: float = 0.0
    heading_deg: float = 0.0          # 0–360°, 0=North
    tilt_corrected_heading: float = 0.0
    compass_direction: CompassDirection = CompassDirection.N
    declination_correction_deg: float = 0.0
    anomaly_type: MagneticAnomalyType = MagneticAnomalyType.NONE
    anomaly_confidence: float = 0.0
    sample_rate_hz: float = 10.0
    range_gauss: int = 4


# ---------------------------------------------------------------------------
# Calibration and Processing
# ---------------------------------------------------------------------------

class HardIronCalibrator:
    """
    Online hard-iron calibration using the sphere-fitting approach.
    Collects field samples while user rotates device through all orientations,
    then estimates the center of the resulting ellipsoid.
    """

    MIN_SAMPLES = 200
    FIELD_VARIANCE_THRESHOLD = 100.0  # µT² — require diverse samples

    def __init__(self) -> None:
        self._samples: List[MagneticVector] = []
        self._calibration = MagCalibration()

    def add_sample(self, field: MagneticVector) -> None:
        self._samples.append(field)

    def compute(self) -> Optional[MagCalibration]:
        if len(self._samples) < self.MIN_SAMPLES:
            return None

        # Simple centroid estimation (full least-squares sphere fitting in production)
        xs = [s.x for s in self._samples]
        ys = [s.y for s in self._samples]
        zs = [s.z for s in self._samples]

        # Check for adequate variance
        def var(vals: List[float]) -> float:
            mean = sum(vals) / len(vals)
            return sum((v - mean)**2 for v in vals) / len(vals)

        if var(xs) < self.FIELD_VARIANCE_THRESHOLD:
            return None  # not enough motion — reject

        offset_x = (max(xs) + min(xs)) / 2.0
        offset_y = (max(ys) + min(ys)) / 2.0
        offset_z = (max(zs) + min(zs)) / 2.0

        self._calibration.hard_iron_offset = MagneticVector(offset_x, offset_y, offset_z)
        self._calibration.calibrated = True
        self._calibration.calibration_timestamp = time.time()
        self._calibration.sample_count = len(self._samples)
        return self._calibration

    def get_calibration(self) -> MagCalibration:
        return self._calibration


class TiltCompensator:
    """
    Applies tilt compensation using roll/pitch angles from accelerometer.
    Projects magnetic field onto horizontal plane for accurate heading.
    """

    def compensate(
        self,
        field: MagneticVector,
        roll_deg: float,
        pitch_deg: float,
    ) -> float:
        """Returns tilt-compensated heading in degrees [0, 360)."""
        roll  = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)

        # Rotate field to horizontal plane
        Xh = (field.x * math.cos(pitch)
              + field.y * math.sin(roll) * math.sin(pitch)
              + field.z * math.cos(roll) * math.sin(pitch))
        Yh = (field.y * math.cos(roll)
              - field.z * math.sin(roll))

        heading = math.degrees(math.atan2(-Yh, Xh))
        if heading < 0:
            heading += 360.0
        return heading


class AnomalyDetector:
    """Detects magnetic interference by monitoring field magnitude stability."""

    EXPECTED_MAGNITUDE_UT   = EARTH_FIELD_UT
    MAGNITUDE_TOLERANCE_PCT = 0.30  # ±30%

    def __init__(self) -> None:
        self._history: Deque[float] = deque(maxlen=20)

    def detect(self, magnitude: float) -> Tuple[MagneticAnomalyType, float]:
        self._history.append(magnitude)
        rel_error = abs(magnitude - self.EXPECTED_MAGNITUDE_UT) / self.EXPECTED_MAGNITUDE_UT

        if rel_error > self.MAGNITUDE_TOLERANCE_PCT:
            # Sudden spike → transient interference
            if len(self._history) >= 3:
                prev = list(self._history)[-3]
                if abs(magnitude - prev) > 20.0:
                    return MagneticAnomalyType.INTERFERENCE, min(1.0, rel_error)
            return MagneticAnomalyType.HARD_IRON, min(1.0, rel_error * 0.7)
        return MagneticAnomalyType.NONE, 0.0


def _heading_to_compass(heading: float) -> CompassDirection:
    """Map 0–360° heading to 16-point compass direction."""
    directions = [
        CompassDirection.N, CompassDirection.NNE, CompassDirection.NE,
        CompassDirection.ENE, CompassDirection.E, CompassDirection.ESE,
        CompassDirection.SE, CompassDirection.SSE, CompassDirection.S,
        CompassDirection.SSW, CompassDirection.SW, CompassDirection.WSW,
        CompassDirection.W, CompassDirection.WNW, CompassDirection.NW,
        CompassDirection.NNW,
    ]
    idx = round(heading / 22.5) % 16
    return directions[idx]


# ---------------------------------------------------------------------------
# Magnetometer Driver
# ---------------------------------------------------------------------------

_GLOBAL_MAG: Optional["Magnetometer"] = None
_GLOBAL_MAG_LOCK = threading.Lock()


class Magnetometer(SensorInterface):
    """
    3-Axis MEMS Magnetometer driver (LIS3MDL).
    Provides compass heading with tilt compensation and anomaly detection.
    """

    SENSOR_ID    = "motion.magnetometer"
    SENSOR_TYPE  = "magnetometer"
    MODEL        = "LIS3MDL"
    MANUFACTURER = "STMicroelectronics"

    # Magnetic declination offset for the device's home location (degrees)
    # Positive = East, Negative = West (configured per geographic location)
    DEFAULT_DECLINATION_DEG = -3.5  # Chicago, IL approx.

    def __init__(
        self,
        mag_range: MagRange = MagRange.G4,
        odr: MagOutputDataRate = MagOutputDataRate.ODR_10,
        declination_deg: float = DEFAULT_DECLINATION_DEG,
    ) -> None:
        self._range = mag_range
        self._odr   = odr
        self._declination = declination_deg

        self._calibrator      = HardIronCalibrator()
        self._tilt_compensator = TiltCompensator()
        self._anomaly_detector = AnomalyDetector()
        self._calibration     = MagCalibration()

        self._roll_deg  = 0.0
        self._pitch_deg = 0.0

        self._lock        = threading.RLock()
        self._running     = False
        self._initialized = False
        self._error_count = 0
        self._read_count  = 0

        self._last_reading: Optional[MagnetometerReading] = None
        self._history: Deque[MagnetometerReading] = deque(maxlen=200)

    # ------------------------------------------------------------------
    # SensorInterface
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        with self._lock:
            try:
                logger.info(f"Initializing {self.MODEL} magnetometer "
                            f"(range=±{self._range.value}G, odr={self._odr.value}Hz)")
                self._initialized = True
                self._running = True
                logger.info("Magnetometer initialized")
                return True
            except Exception as exc:
                logger.error(f"Magnetometer init failed: {exc}")
                return False

    def read(self) -> Optional[MagnetometerReading]:
        if not self._initialized:
            return None
        with self._lock:
            raw = self._read_hardware()
            if raw is None:
                self._error_count += 1
                return None
            reading = self._process(raw, time.time())
            self._last_reading = reading
            self._history.append(reading)
            self._read_count += 1
            return reading

    async def stream(self) -> AsyncIterator[MagnetometerReading]:
        interval = 1.0 / self._odr.value
        while self._running:
            reading = self.read()
            if reading:
                yield reading
            import asyncio
            await asyncio.sleep(interval)

    def calibrate(self) -> bool:
        """Prompts rotating device; collects sphere samples for hard-iron calibration."""
        with self._lock:
            logger.info("Magnetometer calibration — rotate device slowly in a figure-8 "
                        "for 30 seconds")
            self._calibrator = HardIronCalibrator()
            deadline = time.time() + 30.0
            while time.time() < deadline:
                raw = self._read_hardware()
                if raw:
                    self._calibrator.add_sample(raw)
                time.sleep(0.1)

            result = self._calibrator.compute()
            if result:
                self._calibration = result
                logger.info(f"Magnetometer calibrated: offset={self._calibration.hard_iron_offset}")
                return True
            logger.warning("Calibration insufficient — not enough motion detected")
            return False

    def shutdown(self) -> None:
        with self._lock:
            self._running = False
            self._initialized = False
            logger.info("Magnetometer shut down")

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(
            sensor_id=self.SENSOR_ID,
            sensor_type=self.SENSOR_TYPE,
            model=self.MODEL,
            manufacturer=self.MANUFACTURER,
            firmware_version="1.0.0",
            hardware_version="rev-A",
            capabilities={
                "max_range_gauss": 16,
                "max_odr_hz": 1000,
                "hard_iron_calibration": True,
                "tilt_compensation": True,
                "anomaly_detection": True,
            },
        )

    def get_status(self) -> SensorStatus:
        with self._lock:
            if not self._initialized:
                return SensorStatus.UNINITIALIZED
            if self._error_count > 10:
                return SensorStatus.ERROR
            return SensorStatus.RUNNING if self._running else SensorStatus.IDLE

    def is_healthy(self) -> bool:
        return self.get_status() in (SensorStatus.RUNNING, SensorStatus.IDLE)

    def get_health_report(self) -> Dict:
        with self._lock:
            return {
                "status": self.get_status().value,
                "read_count": self._read_count,
                "error_count": self._error_count,
                "calibrated": self._calibration.calibrated,
                "declination_deg": self._declination,
            }

    def read_sync(self) -> Optional[MagnetometerReading]:
        return self.read()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_hardware(self) -> Optional[MagneticVector]:
        """Simulate hardware read. Replace with I2C driver calls."""
        # Earth-like field with small noise
        noise = 1.0
        offset = self._calibration.hard_iron_offset
        return MagneticVector(
            x=random.gauss(20.0 - offset.x, noise),
            y=random.gauss(-10.0 - offset.y, noise),
            z=random.gauss(40.0 - offset.z, noise),
        )

    def _process(self, raw: MagneticVector, ts: float) -> MagnetometerReading:
        # Apply hard-iron correction
        o = self._calibration.hard_iron_offset
        s = self._calibration.soft_iron_scale
        calibrated = MagneticVector(
            x=(raw.x - o.x) * s[0],
            y=(raw.y - o.y) * s[1],
            z=(raw.z - o.z) * s[2],
        )

        # Heading
        magnitude = calibrated.magnitude()
        raw_heading = math.degrees(math.atan2(-calibrated.y, calibrated.x))
        if raw_heading < 0:
            raw_heading += 360.0

        tilt_heading = self._tilt_compensator.compensate(
            calibrated, self._roll_deg, self._pitch_deg
        )
        true_heading = (tilt_heading + self._declination) % 360.0

        anomaly, confidence = self._anomaly_detector.detect(magnitude)
        compass = _heading_to_compass(true_heading)

        return MagnetometerReading(
            sensor_id=self.SENSOR_ID,
            timestamp=ts,
            raw_field=raw,
            calibrated_field=calibrated,
            field_magnitude_ut=magnitude,
            heading_deg=raw_heading,
            tilt_corrected_heading=true_heading,
            compass_direction=compass,
            declination_correction_deg=self._declination,
            anomaly_type=anomaly,
            anomaly_confidence=confidence,
            sample_rate_hz=self._odr.value,
            range_gauss=self._range.value,
            confidence=0.9 if not anomaly else 0.5,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_tilt_angles(self, roll_deg: float, pitch_deg: float) -> None:
        """Update roll/pitch from accelerometer for tilt compensation."""
        with self._lock:
            self._roll_deg  = roll_deg
            self._pitch_deg = pitch_deg

    def get_heading(self) -> float:
        """Returns the most recent tilt-corrected true heading in degrees."""
        if self._last_reading:
            return self._last_reading.tilt_corrected_heading
        return 0.0

    def get_compass_direction(self) -> CompassDirection:
        if self._last_reading:
            return self._last_reading.compass_direction
        return CompassDirection.N


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

def get_magnetometer(
    mag_range: MagRange = MagRange.G4,
    odr: MagOutputDataRate = MagOutputDataRate.ODR_10,
) -> Magnetometer:
    global _GLOBAL_MAG
    with _GLOBAL_MAG_LOCK:
        if _GLOBAL_MAG is None:
            _GLOBAL_MAG = Magnetometer(mag_range=mag_range, odr=odr)
        return _GLOBAL_MAG


def reset_magnetometer() -> None:
    global _GLOBAL_MAG
    with _GLOBAL_MAG_LOCK:
        if _GLOBAL_MAG is not None:
            _GLOBAL_MAG.shutdown()
        _GLOBAL_MAG = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_magnetometer_tests() -> bool:
    logger.info("=== Magnetometer self-test ===")
    mag = Magnetometer()
    assert mag.initialize(), "Init failed"
    reading = mag.read()
    assert reading is not None, "Read returned None"
    assert 10.0 < reading.field_magnitude_ut < 150.0, f"Unexpected magnitude: {reading.field_magnitude_ut}"
    assert 0.0 <= reading.heading_deg < 360.0
    assert mag.is_healthy()
    mag.shutdown()
    logger.info("Magnetometer self-test PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_magnetometer_tests()
    sensor = get_magnetometer()
    sensor.initialize()
    for _ in range(5):
        r = sensor.read()
        print(f"  field={r.calibrated_field}  mag={r.field_magnitude_ut:.1f}µT  "
              f"heading={r.tilt_corrected_heading:.1f}°  dir={r.compass_direction.value}")
        time.sleep(0.1)
    sensor.shutdown()
