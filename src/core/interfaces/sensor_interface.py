"""
Sensor Interface Contracts for AI Holographic Wristwatch System

Defines the abstract base classes and protocols that every sensor
implementation must satisfy. By programming against these interfaces,
the rest of the system can swap hardware drivers, inject mocks for testing,
or run in simulator mode without changing any calling code.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ..constants import SensorConstants, SensorStatus, AlertSeverity
from ..exceptions import SensorError, SensorTimeoutError


# ============================================================================
# Enumerations
# ============================================================================

class SensorType(Enum):
    """Canonical sensor type identifiers."""
    HEART_RATE = "heart_rate"
    SPO2 = "spo2"
    ECG = "ecg"
    SKIN_TEMPERATURE = "skin_temperature"
    GALVANIC_SKIN_RESPONSE = "galvanic_skin_response"
    BLOOD_PRESSURE = "blood_pressure"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    BAROMETER = "barometer"
    AMBIENT_LIGHT = "ambient_light"
    UV_INDEX = "uv_index"
    AIR_QUALITY = "air_quality"
    HUMIDITY_TEMPERATURE = "humidity_temperature"
    MICROPHONE = "microphone"
    GESTURE = "gesture"
    FUSION = "fusion"
    UNKNOWN = "unknown"


class SensorReadingQuality(Enum):
    """Quality tier for a sensor reading."""
    EXCELLENT = "excellent"   # > 95% confidence
    GOOD = "good"             # 80–95%
    FAIR = "fair"             # 60–80%
    POOR = "poor"             # < 60%, use with caution
    INVALID = "invalid"       # Discard


# ============================================================================
# Data Containers
# ============================================================================

@dataclass
class SensorInfo:
    """Static metadata describing a sensor hardware unit."""
    sensor_id: str
    sensor_type: SensorType
    model: str
    manufacturer: str
    firmware_version: str
    sampling_rate_hz: float
    resolution_bits: int
    measurement_range: Tuple[float, float]   # (min, max) in native units
    units: str
    supports_streaming: bool = True
    supports_calibration: bool = True
    power_consumption_mw: float = 0.0
    accuracy_percent: float = 99.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorReading:
    """A single timestamped reading from one sensor."""
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    value: Any = None
    raw_value: Any = None
    quality: SensorReadingQuality = SensorReadingQuality.GOOD
    confidence: float = 1.0
    units: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sequence_number: int = 0

    def is_reliable(self) -> bool:
        """True if quality and confidence are within acceptable thresholds."""
        return (self.quality not in (SensorReadingQuality.POOR,
                                     SensorReadingQuality.INVALID)
                and self.confidence >= SensorConstants.MIN_SIGNAL_QUALITY_PERCENT / 100.0)


@dataclass
class CalibrationResult:
    """Result of a sensor calibration procedure."""
    sensor_id: str
    success: bool
    calibration_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calibration_data: Dict[str, Any] = field(default_factory=dict)
    accuracy_improvement_percent: float = 0.0
    notes: str = ""
    next_calibration_due: Optional[datetime] = None


@dataclass
class FusedSensorData:
    """Output from multi-sensor fusion."""
    fusion_id: str
    source_sensors: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fused_value: Any = None
    confidence: float = 1.0
    quality: SensorReadingQuality = SensorReadingQuality.GOOD
    algorithm: str = "kalman"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorHealthReport:
    """Operational health metrics for a sensor."""
    sensor_id: str
    status: SensorStatus
    uptime_seconds: float
    reading_count: int
    error_count: int
    last_error: Optional[str]
    average_reading_quality: float
    calibration_age_hours: float
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# Core Sensor Interface
# ============================================================================

class SensorInterface(ABC):
    """
    Abstract contract for all sensor implementations.

    Every concrete sensor driver — hardware, simulation, or mock — must
    implement every method here. Thread-safety is mandatory in all
    implementations.
    """

    @property
    @abstractmethod
    def sensor_id(self) -> str:
        """Unique identifier for this sensor instance."""
        ...

    @property
    @abstractmethod
    def sensor_type(self) -> SensorType:
        """Type of physical quantity this sensor measures."""
        ...

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize hardware and prepare for reading.

        Returns:
            True on success.
        Raises:
            SensorError: If sensor is absent or critically broken.
        """
        ...

    @abstractmethod
    async def read(self) -> SensorReading:
        """Acquire a single reading.

        Returns:
            SensorReading with current measurement.
        Raises:
            SensorReadError, SensorTimeoutError
        """
        ...

    @abstractmethod
    async def calibrate(self) -> CalibrationResult:
        """Run the calibration procedure.

        Returns:
            CalibrationResult with success status.
        Raises:
            SensorCalibrationError
        """
        ...

    @abstractmethod
    async def stream(self) -> AsyncIterator[SensorReading]:
        """Yield readings continuously at the native sampling rate.

        Yields:
            SensorReading at each sample interval.
        """
        ...

    @abstractmethod
    def get_sensor_info(self) -> SensorInfo:
        """Return static hardware metadata."""
        ...

    @abstractmethod
    def get_status(self) -> SensorStatus:
        """Return current operational status."""
        ...

    @abstractmethod
    def is_healthy(self) -> bool:
        """Quick health check — True if sensor is operational."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully release hardware resources."""
        ...

    @abstractmethod
    def get_health_report(self) -> SensorHealthReport:
        """Generate detailed health/diagnostics report."""
        ...

    def read_sync(self,
                  timeout_seconds: float = SensorConstants.SENSOR_TIMEOUT_SECONDS
                  ) -> SensorReading:
        """Synchronous convenience wrapper around async read()."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                asyncio.wait_for(self.read(), timeout=timeout_seconds)
            )
        except asyncio.TimeoutError:
            raise SensorTimeoutError(
                f"Sensor {self.sensor_id} timed out after {timeout_seconds}s",
                sensor_type=self.sensor_type.value,
                sensor_id=self.sensor_id,
                timeout_seconds=timeout_seconds,
            )

    def get_sampling_rate(self) -> float:
        """Return nominal sampling rate in Hz."""
        return self.get_sensor_info().sampling_rate_hz


# ============================================================================
# Sensor Fusion Interface
# ============================================================================

class SensorFusionInterface(ABC):
    """Contract for multi-sensor data fusion algorithms."""

    @abstractmethod
    async def fuse(self, readings: List[SensorReading]) -> FusedSensorData:
        """Fuse simultaneous readings from multiple sensors.

        Args:
            readings: SensorReading list from different sensors.
        Returns:
            FusedSensorData with the combined estimate.
        Raises:
            SensorFusionError
        """
        ...

    @abstractmethod
    def get_supported_sensor_types(self) -> List[SensorType]:
        """Return sensor types this fusion algorithm supports."""
        ...

    @abstractmethod
    async def reset(self) -> None:
        """Reset filter state (covariance matrices, etc.)."""
        ...

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return algorithm identifier (e.g., 'Kalman', 'EKF')."""
        ...

    @abstractmethod
    def get_fusion_confidence(self) -> float:
        """Return current fusion confidence (0.0–1.0)."""
        ...


# ============================================================================
# Sensor Registry Interface
# ============================================================================

class SensorRegistryInterface(ABC):
    """Contract for managing a collection of heterogeneous sensors."""

    @abstractmethod
    def register(self, sensor: SensorInterface) -> None:
        """Register a sensor. Raises ValueError on duplicate sensor_id."""
        ...

    @abstractmethod
    def unregister(self, sensor_id: str) -> bool:
        """Remove a sensor by ID. Returns True if found and removed."""
        ...

    @abstractmethod
    def get_sensor(self, sensor_id: str) -> Optional[SensorInterface]:
        """Retrieve sensor by ID, or None if not registered."""
        ...

    @abstractmethod
    def get_sensors_by_type(self, sensor_type: SensorType) -> List[SensorInterface]:
        """Return all registered sensors of the given type."""
        ...

    @abstractmethod
    def list_all(self) -> List[SensorInfo]:
        """Return metadata for all registered sensors."""
        ...

    @abstractmethod
    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all sensors concurrently. Returns {sensor_id: success}."""
        ...

    @abstractmethod
    async def shutdown_all(self) -> None:
        """Shut down all registered sensors gracefully."""
        ...

    @abstractmethod
    def get_overall_health(self) -> Dict[str, SensorHealthReport]:
        """Return health reports for all registered sensors."""
        ...


# ============================================================================
# Sensor Event Handler Interface
# ============================================================================

class SensorEventHandlerInterface(ABC):
    """Contract for reacting to asynchronous sensor events."""

    @abstractmethod
    async def on_reading(self, reading: SensorReading) -> None:
        """Called for every new reading during streaming."""
        ...

    @abstractmethod
    async def on_threshold_crossed(self, reading: SensorReading,
                                   threshold_name: str,
                                   threshold_value: float,
                                   severity: AlertSeverity) -> None:
        """Called when a sensor value crosses a configured threshold."""
        ...

    @abstractmethod
    async def on_quality_degraded(self, sensor_id: str,
                                  old_quality: SensorReadingQuality,
                                  new_quality: SensorReadingQuality) -> None:
        """Called when signal quality falls below acceptable level."""
        ...

    @abstractmethod
    async def on_sensor_error(self, sensor_id: str, error: SensorError) -> None:
        """Called when an unrecoverable sensor error occurs."""
        ...

    @abstractmethod
    async def on_sensor_recovered(self, sensor_id: str) -> None:
        """Called when a previously errored sensor returns to healthy state."""
        ...


# ============================================================================
# Calibration Strategy Interface
# ============================================================================

class CalibrationStrategyInterface(ABC):
    """Strategy pattern for swappable sensor calibration algorithms."""

    @abstractmethod
    async def calibrate(self, sensor: SensorInterface,
                        reference_data: Optional[Dict[str, Any]] = None
                        ) -> CalibrationResult:
        """Execute calibration on the given sensor."""
        ...

    @abstractmethod
    def should_recalibrate(self, last_calibration: CalibrationResult) -> bool:
        """Return True if recalibration is recommended."""
        ...

    @abstractmethod
    def get_calibration_requirements(self) -> Dict[str, Any]:
        """Return conditions required for calibration to succeed."""
        ...


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "SensorType", "SensorReadingQuality",
    "SensorInfo", "SensorReading", "CalibrationResult",
    "FusedSensorData", "SensorHealthReport",
    "SensorInterface", "SensorFusionInterface",
    "SensorRegistryInterface", "SensorEventHandlerInterface",
    "CalibrationStrategyInterface",
]
