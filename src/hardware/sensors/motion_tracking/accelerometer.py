"""
3-Axis Accelerometer Driver — AI Holographic Wristwatch

Implements a high-resolution MEMS accelerometer (e.g., LSM6DSO) providing:
- Raw g-force readings on X/Y/Z axes at up to 6.66 kHz
- Configurable measurement ranges: ±2g / ±4g / ±8g / ±16g
- Low-pass filtering and noise reduction
- Orientation and tilt detection
- Free-fall and tap detection events
- Linear acceleration (gravity-compensated)
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
from enum import Enum, auto
from typing import AsyncIterator, Deque, Dict, List, Optional, Tuple

from src.core.interfaces.sensor_interface import SensorInterface, SensorReading, SensorStatus, SensorInfo
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AccelRange(Enum):
    """Measurement range in g-force."""
    G2  = 2    # ±2g  — most sensitive, lowest noise
    G4  = 4    # ±4g
    G8  = 8    # ±8g
    G16 = 16   # ±16g — least sensitive, highest range


class AccelOutputDataRate(Enum):
    """Output data rate in Hz."""
    ODR_12_5  = 12.5
    ODR_26    = 26.0
    ODR_52    = 52.0
    ODR_104   = 104.0
    ODR_208   = 208.0
    ODR_416   = 416.0
    ODR_833   = 833.0
    ODR_1666  = 1666.0
    ODR_3333  = 3333.0
    ODR_6666  = 6666.0


class Orientation(Enum):
    """Device physical orientation inferred from gravity vector."""
    FACE_UP    = "face_up"
    FACE_DOWN  = "face_down"
    PORTRAIT   = "portrait"
    PORTRAIT_INVERTED = "portrait_inverted"
    LANDSCAPE_LEFT    = "landscape_left"
    LANDSCAPE_RIGHT   = "landscape_right"
    UNKNOWN    = "unknown"


class TapType(Enum):
    NONE   = "none"
    SINGLE = "single"
    DOUBLE = "double"


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class Vector3:
    """3D vector for sensor data."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Vector3":
        m = self.magnitude()
        if m < 1e-9:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(self.x / m, self.y / m, self.z / m)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z


@dataclass
class AccelerometerReading(SensorReading):
    """Full accelerometer measurement snapshot."""
    raw_accel: Vector3 = field(default_factory=Vector3)          # raw g-force
    linear_accel: Vector3 = field(default_factory=Vector3)       # gravity removed
    gravity: Vector3 = field(default_factory=Vector3)            # estimated gravity
    orientation: Orientation = Orientation.UNKNOWN
    tap_type: TapType = TapType.NONE
    is_free_fall: bool = False
    magnitude_g: float = 0.0
    temperature_c: float = 25.0                                  # on-die temp
    sample_rate_hz: float = 104.0
    range_g: int = 4


@dataclass
class AccelCalibration:
    """Bias and scale calibration parameters."""
    bias: Vector3 = field(default_factory=Vector3)
    scale: Vector3 = field(default_factory=lambda: Vector3(1.0, 1.0, 1.0))
    calibrated: bool = False
    calibration_timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Signal Processing
# ---------------------------------------------------------------------------

class GravityEstimator:
    """
    Low-pass IIR filter to separate gravity from linear acceleration.
    α controls the filter time constant: higher α → slower response.
    """
    ALPHA = 0.8  # ≈ 200ms at 104 Hz

    def __init__(self) -> None:
        self._gravity = Vector3(0.0, 0.0, 1.0)  # start: device face-up

    def update(self, raw: Vector3) -> Tuple[Vector3, Vector3]:
        """Returns (gravity, linear_accel)."""
        a = self.ALPHA
        self._gravity = Vector3(
            a * self._gravity.x + (1 - a) * raw.x,
            a * self._gravity.y + (1 - a) * raw.y,
            a * self._gravity.z + (1 - a) * raw.z,
        )
        linear = raw - self._gravity
        return self._gravity, linear

    def reset(self) -> None:
        self._gravity = Vector3(0.0, 0.0, 1.0)


class OrientationDetector:
    """Classifies device orientation from gravity vector."""

    THRESHOLD = 0.7  # cos(45°) ≈ 0.707

    def classify(self, gravity: Vector3) -> Orientation:
        g = gravity.normalize()
        if g.z > self.THRESHOLD:
            return Orientation.FACE_UP
        if g.z < -self.THRESHOLD:
            return Orientation.FACE_DOWN
        if g.y > self.THRESHOLD:
            return Orientation.PORTRAIT
        if g.y < -self.THRESHOLD:
            return Orientation.PORTRAIT_INVERTED
        if g.x > self.THRESHOLD:
            return Orientation.LANDSCAPE_RIGHT
        if g.x < -self.THRESHOLD:
            return Orientation.LANDSCAPE_LEFT
        return Orientation.UNKNOWN


class TapDetector:
    """Single and double-tap detection via threshold crossing."""

    THRESHOLD_G      = 2.0    # linear accel magnitude to qualify as tap
    DOUBLE_TAP_MS    = 400    # max gap between taps for double-tap

    def __init__(self) -> None:
        self._last_tap_time: float = 0.0
        self._pending_single: bool = False

    def update(self, linear: Vector3, timestamp: float) -> TapType:
        mag = linear.magnitude()
        if mag < self.THRESHOLD_G:
            return TapType.NONE

        now = timestamp
        dt_ms = (now - self._last_tap_time) * 1000.0
        self._last_tap_time = now

        if self._pending_single and dt_ms < self.DOUBLE_TAP_MS:
            self._pending_single = False
            return TapType.DOUBLE

        self._pending_single = True
        return TapType.SINGLE


class FreeFallDetector:
    """Detects free-fall when total acceleration < 0.3g for ≥100ms."""

    THRESHOLD_G   = 0.3
    DURATION_SEC  = 0.1

    def __init__(self) -> None:
        self._start_time: Optional[float] = None

    def update(self, raw: Vector3, timestamp: float) -> bool:
        if raw.magnitude() < self.THRESHOLD_G:
            if self._start_time is None:
                self._start_time = timestamp
            elif timestamp - self._start_time >= self.DURATION_SEC:
                return True
        else:
            self._start_time = None
        return False


# ---------------------------------------------------------------------------
# Accelerometer Driver
# ---------------------------------------------------------------------------

_GLOBAL_ACCEL: Optional["Accelerometer"] = None
_GLOBAL_ACCEL_LOCK = threading.Lock()


class Accelerometer(SensorInterface):
    """
    3-Axis MEMS Accelerometer driver.

    Simulates LSM6DSO hardware behaviour. In production, the read_hardware()
    method would communicate with the physical chip via SPI/I2C.
    """

    SENSOR_ID   = "motion.accelerometer"
    SENSOR_TYPE = "accelerometer"
    MODEL       = "LSM6DSO"
    MANUFACTURER = "STMicroelectronics"

    def __init__(
        self,
        accel_range: AccelRange = AccelRange.G4,
        odr: AccelOutputDataRate = AccelOutputDataRate.ODR_104,
        enable_tap_detection: bool = True,
        enable_freefall: bool = True,
    ) -> None:
        self._range   = accel_range
        self._odr     = odr
        self._tap_det = enable_tap_detection
        self._ff_det  = enable_freefall

        self._gravity_est     = GravityEstimator()
        self._orientation_det = OrientationDetector()
        self._tap_detector    = TapDetector()
        self._ff_detector     = FreeFallDetector()
        self._calibration     = AccelCalibration()

        self._lock    = threading.RLock()
        self._running = False
        self._initialized = False
        self._error_count = 0
        self._read_count  = 0

        self._last_reading: Optional[AccelerometerReading] = None
        self._history: Deque[AccelerometerReading] = deque(maxlen=500)

    # ------------------------------------------------------------------
    # SensorInterface implementation
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        with self._lock:
            try:
                logger.info(f"Initializing {self.MODEL} accelerometer "
                            f"(range=±{self._range.value}g, odr={self._odr.value}Hz)")
                # Hardware: write config registers here
                self._gravity_est.reset()
                self._initialized = True
                self._running = True
                logger.info("Accelerometer initialized")
                return True
            except Exception as exc:
                logger.error(f"Accelerometer init failed: {exc}")
                return False

    def read(self) -> Optional[AccelerometerReading]:
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

    async def stream(self) -> AsyncIterator[AccelerometerReading]:
        interval = 1.0 / self._odr.value
        while self._running:
            reading = self.read()
            if reading:
                yield reading
            import asyncio
            await asyncio.sleep(interval)

    def calibrate(self) -> bool:
        """Flat calibration: average 100 samples, measure bias vs (0,0,1g)."""
        with self._lock:
            logger.info("Calibrating accelerometer — keep device flat and still")
            samples: List[Vector3] = []
            for _ in range(100):
                raw = self._read_hardware()
                if raw:
                    samples.append(raw)
                time.sleep(0.01)

            if len(samples) < 50:
                logger.warning("Insufficient samples for calibration")
                return False

            mean_x = sum(s.x for s in samples) / len(samples)
            mean_y = sum(s.y for s in samples) / len(samples)
            mean_z = sum(s.z for s in samples) / len(samples)

            self._calibration.bias = Vector3(mean_x, mean_y, mean_z - 1.0)
            self._calibration.calibrated = True
            self._calibration.calibration_timestamp = time.time()
            logger.info(f"Calibration complete — bias={self._calibration.bias}")
            return True

    def shutdown(self) -> None:
        with self._lock:
            self._running = False
            self._initialized = False
            logger.info("Accelerometer shut down")

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(
            sensor_id=self.SENSOR_ID,
            sensor_type=self.SENSOR_TYPE,
            model=self.MODEL,
            manufacturer=self.MANUFACTURER,
            firmware_version="1.0.0",
            hardware_version="rev-A",
            capabilities={
                "max_range_g": 16,
                "max_odr_hz": 6666,
                "tap_detection": True,
                "freefall_detection": True,
                "on_die_temperature": True,
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
                "initialized": self._initialized,
                "read_count": self._read_count,
                "error_count": self._error_count,
                "calibrated": self._calibration.calibrated,
                "range_g": self._range.value,
                "odr_hz": self._odr.value,
            }

    def read_sync(self) -> Optional[AccelerometerReading]:
        return self.read()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_hardware(self) -> Optional[Vector3]:
        """Simulate hardware SPI/I2C read. Replace with real driver calls."""
        t = time.time()
        # Simulate gravity + small noise + occasional arm movement
        base_z = 1.0  # device mostly face-up on wrist
        noise  = 0.02
        x = random.gauss(0.0,  noise)
        y = random.gauss(0.05, noise)  # slight wrist tilt
        z = random.gauss(base_z, noise)

        # Apply calibration bias correction
        b = self._calibration.bias
        return Vector3(x - b.x, y - b.y, z - b.z)

    def _process(self, raw: Vector3, ts: float) -> AccelerometerReading:
        gravity, linear = self._gravity_est.update(raw)
        orientation = self._orientation_det.classify(gravity)
        tap_type = self._tap_detector.update(linear, ts) if self._tap_det else TapType.NONE
        is_ff    = self._ff_detector.update(raw, ts) if self._ff_det else False

        return AccelerometerReading(
            sensor_id=self.SENSOR_ID,
            timestamp=ts,
            raw_accel=raw,
            linear_accel=linear,
            gravity=gravity,
            orientation=orientation,
            tap_type=tap_type,
            is_free_fall=is_ff,
            magnitude_g=raw.magnitude(),
            temperature_c=random.gauss(27.0, 0.3),
            sample_rate_hz=self._odr.value,
            range_g=self._range.value,
            confidence=0.97,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_last_reading(self) -> Optional[AccelerometerReading]:
        return self._last_reading

    def get_current_orientation(self) -> Orientation:
        if self._last_reading:
            return self._last_reading.orientation
        return Orientation.UNKNOWN

    def get_history(self) -> List[AccelerometerReading]:
        with self._lock:
            return list(self._history)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

def get_accelerometer(
    accel_range: AccelRange = AccelRange.G4,
    odr: AccelOutputDataRate = AccelOutputDataRate.ODR_104,
) -> Accelerometer:
    global _GLOBAL_ACCEL
    with _GLOBAL_ACCEL_LOCK:
        if _GLOBAL_ACCEL is None:
            _GLOBAL_ACCEL = Accelerometer(accel_range=accel_range, odr=odr)
        return _GLOBAL_ACCEL


def reset_accelerometer() -> None:
    global _GLOBAL_ACCEL
    with _GLOBAL_ACCEL_LOCK:
        if _GLOBAL_ACCEL is not None:
            _GLOBAL_ACCEL.shutdown()
        _GLOBAL_ACCEL = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_accelerometer_tests() -> bool:
    logger.info("=== Accelerometer self-test ===")
    accel = Accelerometer()
    assert accel.initialize(), "Init failed"
    reading = accel.read()
    assert reading is not None, "Read returned None"
    assert 0.5 < reading.magnitude_g < 1.5, f"Unexpected magnitude: {reading.magnitude_g}"
    assert accel.is_healthy(), "Not healthy"
    info = accel.get_sensor_info()
    assert info.sensor_id == Accelerometer.SENSOR_ID
    accel.shutdown()
    logger.info("Accelerometer self-test PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_accelerometer_tests()
    sensor = get_accelerometer()
    sensor.initialize()
    for _ in range(5):
        r = sensor.read()
        print(f"  accel={r.raw_accel}  orientation={r.orientation.value}  mag={r.magnitude_g:.3f}g")
        time.sleep(0.1)
    sensor.shutdown()
