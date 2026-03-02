"""
Complete Exception Hierarchy for AI Holographic Wristwatch System

This module defines every typed exception used across the entire system.
No module should raise a bare Exception — always use a typed exception
from this hierarchy. Each exception carries structured metadata including
a correlation ID, severity level, recovery hint, and optional context dict
to make debugging and observability far richer than plain string messages.

Exception tree:
    AIWristwatchError (root)
    ├── HardwareError
    │   ├── SensorError
    │   │   ├── SensorCalibrationError
    │   │   ├── SensorReadError
    │   │   ├── SensorTimeoutError
    │   │   └── SensorFusionError
    │   ├── HolographicDisplayError
    │   │   ├── ProjectorError
    │   │   ├── RenderError
    │   │   └── DisplayCalibrationError
    │   ├── PowerManagementError
    │   │   ├── BatteryError
    │   │   └── ChargingError
    │   └── CommunicationError
    │       ├── BluetoothError
    │       ├── WiFiError
    │       └── NearFieldError
    ├── AISystemError
    │   ├── ConversationError
    │   │   ├── IntentRecognitionError
    │   │   └── DialogueError
    │   ├── ModelError
    │   │   ├── ModelLoadError
    │   │   ├── InferenceError
    │   │   └── ModelTimeoutError
    │   ├── SafetyViolationError
    │   ├── PersonalityError
    │   ├── KnowledgeError
    │   └── LearningError
    ├── SecurityError
    │   ├── AuthenticationError
    │   ├── AuthorizationError
    │   ├── TamperDetectionError
    │   ├── EncryptionError
    │   └── PrivacyViolationError
    ├── DataError
    │   ├── DataCorruptionError
    │   ├── SchemaValidationError
    │   ├── IntegrityError
    │   ├── SerializationError
    │   └── StorageError
    ├── NetworkError
    │   ├── RequestTimeoutError
    │   ├── ConnectionError
    │   ├── RateLimitError
    │   ├── AuthenticationFailedError
    │   └── ServiceUnavailableError
    ├── ConfigurationError
    │   ├── MissingConfigError
    │   ├── InvalidConfigError
    │   └── ConfigLoadError
    └── ApplicationError
        ├── SyncError
        ├── UIError
        ├── NotificationError
        └── UpdateError
"""

import uuid
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum


# ============================================================================
# Exception Metadata Enumerations
# ============================================================================

class ExceptionSeverity(Enum):
    """Triage severity for exceptions — drives alerting and logging level."""
    LOW = "low"           # Degraded but functional
    MEDIUM = "medium"     # Feature impaired, user impacted
    HIGH = "high"         # Critical path broken
    CRITICAL = "critical" # System unstable or safety risk
    FATAL = "fatal"       # Unrecoverable; restart required


class RecoveryStrategy(Enum):
    """Suggested recovery action for the catching code."""
    RETRY = "retry"
    FALLBACK = "fallback"
    USER_ACTION_REQUIRED = "user_action_required"
    RESTART_SUBSYSTEM = "restart_subsystem"
    RESTART_DEVICE = "restart_device"
    CONTACT_SUPPORT = "contact_support"
    NONE = "none"


# ============================================================================
# Root Exception
# ============================================================================

class AIWristwatchError(Exception):
    """
    Root exception for the entire AI Holographic Wristwatch system.

    All system exceptions must inherit from this class. It carries structured
    metadata for observability, logging, and incident response.

    Args:
        message:          Human-readable description of the failure.
        severity:         How critical is this failure.
        recovery:         Suggested recovery strategy.
        context:          Arbitrary key-value pairs for debugging context.
        correlation_id:   Optional trace/correlation ID; auto-generated if None.
        cause:            The original exception that triggered this one.
    """

    def __init__(
        self,
        message: str,
        severity: ExceptionSeverity = ExceptionSeverity.MEDIUM,
        recovery: RecoveryStrategy = RecoveryStrategy.NONE,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.recovery = recovery
        self.context: Dict[str, Any] = context or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.error_code = self._build_error_code()

        # Capture original traceback if a cause is provided
        self.original_traceback: Optional[str] = (
            "".join(traceback.format_exception(type(cause), cause, cause.__traceback__))
            if cause else None
        )

    def _build_error_code(self) -> str:
        """Build a machine-readable error code from the class hierarchy."""
        parts = [cls.__name__.replace("Error", "").upper() for cls in type(self).__mro__
                 if issubclass(cls, AIWristwatchError) and cls is not AIWristwatchError]
        return ".".join(reversed(parts)) or "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exception to a structured dictionary for logging."""
        return {
            "error_type": type(self).__name__,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "recovery": self.recovery.value,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"severity={self.severity.value}, "
            f"correlation_id={self.correlation_id!r})"
        )

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.message} (id={self.correlation_id[:8]})"


# ============================================================================
# Hardware Exceptions
# ============================================================================

class HardwareError(AIWristwatchError):
    """Base exception for all hardware-related failures."""

    def __init__(self, message: str, hardware_component: str = "unknown",
                 **kwargs):
        super().__init__(message, **kwargs)
        self.hardware_component = hardware_component
        self.context["hardware_component"] = hardware_component


# ── Sensor Exceptions ────────────────────────────────────────────────────────

class SensorError(HardwareError):
    """Base exception for sensor failures."""

    def __init__(self, message: str, sensor_type: str = "unknown",
                 sensor_id: Optional[str] = None, **kwargs):
        kwargs.setdefault("hardware_component", f"sensor:{sensor_type}")
        super().__init__(message, **kwargs)
        self.sensor_type = sensor_type
        self.sensor_id = sensor_id
        self.context.update({"sensor_type": sensor_type, "sensor_id": sensor_id})


class SensorCalibrationError(SensorError):
    """Sensor calibration failed or calibration data is invalid."""

    def __init__(self, message: str, calibration_step: Optional[str] = None,
                 **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)
        self.calibration_step = calibration_step
        self.context["calibration_step"] = calibration_step


class SensorReadError(SensorError):
    """Failed to read data from a sensor."""

    def __init__(self, message: str, read_attempt: int = 1, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)
        self.read_attempt = read_attempt
        self.context["read_attempt"] = read_attempt


class SensorTimeoutError(SensorError):
    """Sensor did not respond within the expected time window."""

    def __init__(self, message: str, timeout_seconds: float = 0.0, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.RESTART_SUBSYSTEM)
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.context["timeout_seconds"] = timeout_seconds


class SensorFusionError(SensorError):
    """Multi-sensor data fusion failed."""

    def __init__(self, message: str, fused_sensors: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(message, sensor_type="fusion", **kwargs)
        self.fused_sensors = fused_sensors or []
        self.context["fused_sensors"] = self.fused_sensors


# ── Holographic Display Exceptions ───────────────────────────────────────────

class HolographicDisplayError(HardwareError):
    """Base exception for holographic display subsystem failures."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("hardware_component", "holographic_display")
        super().__init__(message, **kwargs)


class ProjectorError(HolographicDisplayError):
    """Laser projector hardware failure."""

    def __init__(self, message: str, projector_id: Optional[str] = None,
                 laser_channel: Optional[str] = None, **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        super().__init__(message, **kwargs)
        self.projector_id = projector_id
        self.laser_channel = laser_channel
        self.context.update({"projector_id": projector_id, "laser_channel": laser_channel})


class EyeSafetyViolationError(ProjectorError):
    """Laser power exceeds eye-safety limits — safety interlock triggered."""

    def __init__(self, message: str, measured_power_mw: float = 0.0,
                 safe_limit_mw: float = 0.0, **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.FATAL)
        kwargs.setdefault("recovery", RecoveryStrategy.RESTART_DEVICE)
        super().__init__(message, **kwargs)
        self.measured_power_mw = measured_power_mw
        self.safe_limit_mw = safe_limit_mw
        self.context.update({"measured_power_mw": measured_power_mw,
                              "safe_limit_mw": safe_limit_mw})


class RenderError(HolographicDisplayError):
    """Holographic rendering pipeline failure."""

    def __init__(self, message: str, render_stage: Optional[str] = None,
                 frame_number: Optional[int] = None, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)
        self.render_stage = render_stage
        self.frame_number = frame_number
        self.context.update({"render_stage": render_stage, "frame_number": frame_number})


class DisplayCalibrationError(HolographicDisplayError):
    """Holographic display calibration failed."""

    def __init__(self, message: str, calibration_type: Optional[str] = None,
                 **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)
        self.calibration_type = calibration_type
        self.context["calibration_type"] = calibration_type


# ── Power Management Exceptions ──────────────────────────────────────────────

class PowerManagementError(HardwareError):
    """Base exception for power and battery failures."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("hardware_component", "power_management")
        super().__init__(message, **kwargs)


class BatteryError(PowerManagementError):
    """Battery health, state, or measurement error."""

    def __init__(self, message: str, battery_soc: Optional[float] = None,
                 battery_voltage: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.battery_soc = battery_soc
        self.battery_voltage = battery_voltage
        self.context.update({"battery_soc": battery_soc, "battery_voltage": battery_voltage})


class ChargingError(PowerManagementError):
    """Charging system fault."""

    def __init__(self, message: str, charging_method: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.charging_method = charging_method
        self.context["charging_method"] = charging_method


class ThermalThrottleError(PowerManagementError):
    """Device exceeded thermal limits and must throttle or shut down."""

    def __init__(self, message: str, temperature_c: float = 0.0,
                 limit_c: float = 0.0, **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        kwargs.setdefault("recovery", RecoveryStrategy.RESTART_SUBSYSTEM)
        super().__init__(message, **kwargs)
        self.temperature_c = temperature_c
        self.limit_c = limit_c
        self.context.update({"temperature_c": temperature_c, "limit_c": limit_c})


# ── Communication Exceptions ─────────────────────────────────────────────────

class CommunicationError(HardwareError):
    """Base exception for wireless communication failures."""

    def __init__(self, message: str, protocol: str = "unknown", **kwargs):
        kwargs.setdefault("hardware_component", f"comm:{protocol}")
        super().__init__(message, **kwargs)
        self.protocol = protocol
        self.context["protocol"] = protocol


class BluetoothError(CommunicationError):
    """Bluetooth LE communication failure."""

    def __init__(self, message: str, device_address: Optional[str] = None,
                 **kwargs):
        super().__init__(message, protocol="bluetooth_le", **kwargs)
        self.device_address = device_address
        self.context["device_address"] = device_address


class WiFiError(CommunicationError):
    """Wi-Fi communication failure."""

    def __init__(self, message: str, ssid: Optional[str] = None, **kwargs):
        super().__init__(message, protocol="wifi", **kwargs)
        self.ssid = ssid
        self.context["ssid"] = ssid


class NearFieldError(CommunicationError):
    """NFC communication failure."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, protocol="nfc", **kwargs)


# ============================================================================
# AI System Exceptions
# ============================================================================

class AISystemError(AIWristwatchError):
    """Base exception for AI subsystem failures."""

    def __init__(self, message: str, ai_component: str = "unknown",
                 model_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.ai_component = ai_component
        self.model_id = model_id
        self.context.update({"ai_component": ai_component, "model_id": model_id})


# ── Conversation Exceptions ───────────────────────────────────────────────────

class ConversationError(AISystemError):
    """Failure in the conversational AI pipeline."""

    def __init__(self, message: str, conversation_id: Optional[str] = None,
                 turn_number: Optional[int] = None, **kwargs):
        kwargs.setdefault("ai_component", "conversational_ai")
        super().__init__(message, **kwargs)
        self.conversation_id = conversation_id
        self.turn_number = turn_number
        self.context.update({"conversation_id": conversation_id,
                              "turn_number": turn_number})


class IntentRecognitionError(ConversationError):
    """Intent could not be recognized with sufficient confidence."""

    def __init__(self, message: str, confidence: float = 0.0,
                 user_input_length: int = 0, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.FALLBACK)
        super().__init__(message, **kwargs)
        self.confidence = confidence
        self.context.update({"confidence": confidence,
                              "user_input_length": user_input_length})


class DialogueError(ConversationError):
    """Dialogue management state machine encountered an invalid state."""

    def __init__(self, message: str, current_state: Optional[str] = None,
                 attempted_transition: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.current_state = current_state
        self.attempted_transition = attempted_transition
        self.context.update({"current_state": current_state,
                              "attempted_transition": attempted_transition})


# ── Model Exceptions ──────────────────────────────────────────────────────────

class ModelError(AISystemError):
    """Base exception for AI model failures."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("ai_component", "model")
        super().__init__(message, **kwargs)


class ModelLoadError(ModelError):
    """Model could not be loaded from storage or network."""

    def __init__(self, message: str, model_path: Optional[str] = None,
                 **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        kwargs.setdefault("recovery", RecoveryStrategy.FALLBACK)
        super().__init__(message, **kwargs)
        self.model_path = model_path
        self.context["model_path"] = model_path


class InferenceError(ModelError):
    """Model inference failed at runtime."""

    def __init__(self, message: str, input_shape: Optional[Any] = None,
                 **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)
        self.input_shape = input_shape
        self.context["input_shape"] = str(input_shape)


class ModelTimeoutError(ModelError):
    """Model inference exceeded the allowed time budget."""

    def __init__(self, message: str, timeout_ms: float = 0.0,
                 actual_ms: float = 0.0, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.FALLBACK)
        super().__init__(message, **kwargs)
        self.timeout_ms = timeout_ms
        self.actual_ms = actual_ms
        self.context.update({"timeout_ms": timeout_ms, "actual_ms": actual_ms})


# ── Safety & Ethics Exceptions ────────────────────────────────────────────────

class SafetyViolationError(AISystemError):
    """AI output violated safety or ethical constraints."""

    def __init__(self, message: str, violation_type: str = "unknown",
                 safety_score: float = 0.0, **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        kwargs.setdefault("recovery", RecoveryStrategy.FALLBACK)
        kwargs.setdefault("ai_component", "safety_filter")
        super().__init__(message, **kwargs)
        self.violation_type = violation_type
        self.safety_score = safety_score
        self.context.update({"violation_type": violation_type,
                              "safety_score": safety_score})


class PersonalityError(AISystemError):
    """Personality engine encountered an inconsistent or invalid state."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("ai_component", "personality_engine")
        super().__init__(message, **kwargs)


class KnowledgeError(AISystemError):
    """Knowledge base query or update failure."""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        kwargs.setdefault("ai_component", "knowledge_system")
        super().__init__(message, **kwargs)
        self.query = query
        self.context["query"] = query[:100] if query else None  # truncate for safety


class LearningError(AISystemError):
    """Online learning or model update failure."""

    def __init__(self, message: str, learning_algorithm: Optional[str] = None,
                 **kwargs):
        kwargs.setdefault("ai_component", "learning_system")
        super().__init__(message, **kwargs)
        self.learning_algorithm = learning_algorithm
        self.context["learning_algorithm"] = learning_algorithm


# ============================================================================
# Security Exceptions
# ============================================================================

class SecurityError(AIWristwatchError):
    """Base exception for security-related failures."""

    def __init__(self, message: str, security_domain: str = "unknown",
                 **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        super().__init__(message, **kwargs)
        self.security_domain = security_domain
        self.context["security_domain"] = security_domain


class AuthenticationError(SecurityError):
    """User or device authentication failed."""

    def __init__(self, message: str, auth_method: str = "unknown",
                 attempt_number: int = 1, **kwargs):
        kwargs.setdefault("security_domain", "authentication")
        super().__init__(message, **kwargs)
        self.auth_method = auth_method
        self.attempt_number = attempt_number
        self.context.update({"auth_method": auth_method,
                              "attempt_number": attempt_number})


class AuthorizationError(SecurityError):
    """Authenticated principal lacks the required permissions."""

    def __init__(self, message: str, required_permission: Optional[str] = None,
                 resource: Optional[str] = None, **kwargs):
        kwargs.setdefault("security_domain", "authorization")
        super().__init__(message, **kwargs)
        self.required_permission = required_permission
        self.resource = resource
        self.context.update({"required_permission": required_permission,
                              "resource": resource})


class TamperDetectionError(SecurityError):
    """Physical or logical tamper event detected."""

    def __init__(self, message: str, tamper_type: str = "unknown",
                 **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.FATAL)
        kwargs.setdefault("recovery", RecoveryStrategy.RESTART_DEVICE)
        kwargs.setdefault("security_domain", "tamper_detection")
        super().__init__(message, **kwargs)
        self.tamper_type = tamper_type
        self.context["tamper_type"] = tamper_type


class EncryptionError(SecurityError):
    """Cryptographic operation failed."""

    def __init__(self, message: str, operation: str = "unknown", **kwargs):
        kwargs.setdefault("security_domain", "encryption")
        super().__init__(message, **kwargs)
        self.operation = operation
        self.context["crypto_operation"] = operation


class PrivacyViolationError(SecurityError):
    """Operation would violate user's privacy settings or consent."""

    def __init__(self, message: str, data_category: Optional[str] = None,
                 required_consent: Optional[str] = None, **kwargs):
        kwargs.setdefault("security_domain", "privacy")
        kwargs.setdefault("recovery", RecoveryStrategy.USER_ACTION_REQUIRED)
        super().__init__(message, **kwargs)
        self.data_category = data_category
        self.required_consent = required_consent
        self.context.update({"data_category": data_category,
                              "required_consent": required_consent})


# ============================================================================
# Data Exceptions
# ============================================================================

class DataError(AIWristwatchError):
    """Base exception for data integrity and processing failures."""

    def __init__(self, message: str, data_source: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.data_source = data_source
        self.context["data_source"] = data_source


class DataCorruptionError(DataError):
    """Data corruption detected during read, write, or transfer."""

    def __init__(self, message: str, expected_checksum: Optional[str] = None,
                 actual_checksum: Optional[str] = None, **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        kwargs.setdefault("recovery", RecoveryStrategy.FALLBACK)
        super().__init__(message, **kwargs)
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum
        self.context.update({"expected_checksum": expected_checksum,
                              "actual_checksum": actual_checksum})


class SchemaValidationError(DataError):
    """Data does not match the expected schema."""

    def __init__(self, message: str, schema_name: Optional[str] = None,
                 field_errors: Optional[Dict[str, List[str]]] = None,
                 **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.USER_ACTION_REQUIRED)
        super().__init__(message, **kwargs)
        self.schema_name = schema_name
        self.field_errors: Dict[str, List[str]] = field_errors or {}
        self.context.update({"schema_name": schema_name,
                              "field_errors": self.field_errors})


class IntegrityError(DataError):
    """Data integrity verification failed (checksum, HMAC, etc.)."""

    def __init__(self, message: str, integrity_check_type: str = "checksum",
                 **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        super().__init__(message, **kwargs)
        self.integrity_check_type = integrity_check_type
        self.context["integrity_check_type"] = integrity_check_type


class SerializationError(DataError):
    """Data serialization or deserialization failed."""

    def __init__(self, message: str, format_type: Optional[str] = None,
                 direction: str = "serialize", **kwargs):
        super().__init__(message, **kwargs)
        self.format_type = format_type
        self.direction = direction
        self.context.update({"format_type": format_type, "direction": direction})


class StorageError(DataError):
    """Storage read/write/delete operation failed."""

    def __init__(self, message: str, storage_backend: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)
        self.storage_backend = storage_backend
        self.operation = operation
        self.context.update({"storage_backend": storage_backend,
                              "operation": operation})


# ============================================================================
# Network Exceptions
# ============================================================================

class NetworkError(AIWristwatchError):
    """Base exception for network and API failures."""

    def __init__(self, message: str, url: Optional[str] = None,
                 status_code: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.url = url
        self.status_code = status_code
        self.context.update({"url": url, "status_code": status_code})


class RequestTimeoutError(NetworkError):
    """HTTP/WebSocket request timed out."""

    def __init__(self, message: str, timeout_seconds: float = 0.0, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.context["timeout_seconds"] = timeout_seconds


class ConnectionError(NetworkError):
    """Cannot establish network connection."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)


class RateLimitError(NetworkError):
    """API rate limit exceeded."""

    def __init__(self, message: str, retry_after_seconds: float = 60.0,
                 **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.MEDIUM)
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)
        self.retry_after_seconds = retry_after_seconds
        self.context["retry_after_seconds"] = retry_after_seconds


class AuthenticationFailedError(NetworkError):
    """API authentication credentials rejected."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        kwargs.setdefault("recovery", RecoveryStrategy.USER_ACTION_REQUIRED)
        super().__init__(message, status_code=401, **kwargs)


class ServiceUnavailableError(NetworkError):
    """Backend service temporarily unavailable."""

    def __init__(self, message: str, service_name: Optional[str] = None,
                 **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, status_code=503, **kwargs)
        self.service_name = service_name
        self.context["service_name"] = service_name


# ============================================================================
# Configuration Exceptions
# ============================================================================

class ConfigurationError(AIWristwatchError):
    """Base exception for configuration failures."""

    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_file: Optional[str] = None, **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_file = config_file
        self.context.update({"config_key": config_key, "config_file": config_file})


class MissingConfigError(ConfigurationError):
    """Required configuration key is absent."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.USER_ACTION_REQUIRED)
        super().__init__(message, **kwargs)


class InvalidConfigError(ConfigurationError):
    """Configuration value failed validation."""

    def __init__(self, message: str, expected_type: Optional[str] = None,
                 actual_value: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.expected_type = expected_type
        self.actual_value = actual_value
        self.context.update({"expected_type": expected_type})


class ConfigLoadError(ConfigurationError):
    """Configuration file could not be loaded."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.FALLBACK)
        super().__init__(message, **kwargs)


# ============================================================================
# Application Exceptions
# ============================================================================

class ApplicationError(AIWristwatchError):
    """Base exception for application-layer failures."""

    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.component = component
        self.context["component"] = component


class SyncError(ApplicationError):
    """Data synchronization between device and backend failed."""

    def __init__(self, message: str, sync_direction: str = "bidirectional",
                 records_affected: int = 0, **kwargs):
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, component="sync_engine", **kwargs)
        self.sync_direction = sync_direction
        self.records_affected = records_affected
        self.context.update({"sync_direction": sync_direction,
                              "records_affected": records_affected})


class UIError(ApplicationError):
    """User interface rendering or interaction failure."""

    def __init__(self, message: str, view_name: Optional[str] = None,
                 **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.LOW)
        super().__init__(message, component="ui", **kwargs)
        self.view_name = view_name
        self.context["view_name"] = view_name


class NotificationError(ApplicationError):
    """Notification delivery failed."""

    def __init__(self, message: str, notification_type: Optional[str] = None,
                 **kwargs):
        super().__init__(message, component="notification", **kwargs)
        self.notification_type = notification_type
        self.context["notification_type"] = notification_type


class UpdateError(ApplicationError):
    """Firmware or app update process failed."""

    def __init__(self, message: str, update_version: Optional[str] = None,
                 update_stage: Optional[str] = None, **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.HIGH)
        kwargs.setdefault("recovery", RecoveryStrategy.RETRY)
        super().__init__(message, component="update_system", **kwargs)
        self.update_version = update_version
        self.update_stage = update_stage
        self.context.update({"update_version": update_version,
                              "update_stage": update_stage})


# ============================================================================
# Health Monitoring Exceptions
# ============================================================================

class HealthMonitoringError(AIWristwatchError):
    """Failure in health monitoring or medical alert pipeline."""

    def __init__(self, message: str, health_domain: str = "general", **kwargs):
        super().__init__(message, **kwargs)
        self.health_domain = health_domain
        self.context["health_domain"] = health_domain


class CriticalHealthAlert(HealthMonitoringError):
    """A critical health threshold was exceeded — user intervention needed."""

    def __init__(self, message: str, vital_sign: str = "unknown",
                 measured_value: Optional[float] = None,
                 critical_threshold: Optional[float] = None, **kwargs):
        kwargs.setdefault("severity", ExceptionSeverity.CRITICAL)
        kwargs.setdefault("recovery", RecoveryStrategy.USER_ACTION_REQUIRED)
        super().__init__(message, health_domain=vital_sign, **kwargs)
        self.vital_sign = vital_sign
        self.measured_value = measured_value
        self.critical_threshold = critical_threshold
        self.context.update({"vital_sign": vital_sign,
                              "measured_value": measured_value,
                              "critical_threshold": critical_threshold})


# ============================================================================
# Utility Functions
# ============================================================================

def wrap_exception(
    exc: Exception,
    wrapper_class: type,
    message: Optional[str] = None,
    **kwargs
) -> AIWristwatchError:
    """
    Wrap a non-system exception in the appropriate AIWristwatchError subclass.

    Args:
        exc:           The original exception to wrap.
        wrapper_class: The AIWristwatchError subclass to use.
        message:       Override message; defaults to str(exc).
        **kwargs:      Additional keyword arguments for the wrapper class.

    Returns:
        Instance of wrapper_class with the original exc as cause.
    """
    msg = message or str(exc)
    return wrapper_class(msg, cause=exc, **kwargs)


def get_recovery_hint(exception: AIWristwatchError) -> str:
    """
    Return a human-readable recovery hint for presenting to users or operators.

    Args:
        exception: Any AIWristwatchError instance.

    Returns:
        A short string hint about what to do next.
    """
    hints = {
        RecoveryStrategy.RETRY: "The operation can be retried. Please try again.",
        RecoveryStrategy.FALLBACK: "Using a fallback mechanism. Some features may be limited.",
        RecoveryStrategy.USER_ACTION_REQUIRED: "User action is required to resolve this issue.",
        RecoveryStrategy.RESTART_SUBSYSTEM: "A subsystem needs to be restarted.",
        RecoveryStrategy.RESTART_DEVICE: "Device restart is required.",
        RecoveryStrategy.CONTACT_SUPPORT: "Please contact support for assistance.",
        RecoveryStrategy.NONE: "No automatic recovery available.",
    }
    return hints.get(exception.recovery, "Unknown recovery strategy.")


def is_retryable(exception: Exception) -> bool:
    """
    Return True if the exception represents a potentially transient failure
    that is worth retrying.

    Args:
        exception: Any exception instance.

    Returns:
        True if retrying the operation may succeed.
    """
    if isinstance(exception, AIWristwatchError):
        return exception.recovery == RecoveryStrategy.RETRY
    # For unexpected non-system exceptions, default to not retrying
    return False


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    # Metadata enums
    "ExceptionSeverity", "RecoveryStrategy",

    # Root
    "AIWristwatchError",

    # Hardware
    "HardwareError",
    "SensorError", "SensorCalibrationError", "SensorReadError",
    "SensorTimeoutError", "SensorFusionError",
    "HolographicDisplayError", "ProjectorError", "EyeSafetyViolationError",
    "RenderError", "DisplayCalibrationError",
    "PowerManagementError", "BatteryError", "ChargingError", "ThermalThrottleError",
    "CommunicationError", "BluetoothError", "WiFiError", "NearFieldError",

    # AI System
    "AISystemError",
    "ConversationError", "IntentRecognitionError", "DialogueError",
    "ModelError", "ModelLoadError", "InferenceError", "ModelTimeoutError",
    "SafetyViolationError", "PersonalityError", "KnowledgeError", "LearningError",

    # Security
    "SecurityError",
    "AuthenticationError", "AuthorizationError", "TamperDetectionError",
    "EncryptionError", "PrivacyViolationError",

    # Data
    "DataError",
    "DataCorruptionError", "SchemaValidationError", "IntegrityError",
    "SerializationError", "StorageError",

    # Network
    "NetworkError",
    "RequestTimeoutError", "ConnectionError", "RateLimitError",
    "AuthenticationFailedError", "ServiceUnavailableError",

    # Configuration
    "ConfigurationError", "MissingConfigError", "InvalidConfigError", "ConfigLoadError",

    # Application
    "ApplicationError", "SyncError", "UIError", "NotificationError", "UpdateError",

    # Health
    "HealthMonitoringError", "CriticalHealthAlert",

    # Utilities
    "wrap_exception", "get_recovery_hint", "is_retryable",
]


if __name__ == "__main__":
    print("AI Holographic Wristwatch — Exception Hierarchy Module")
    print("=" * 55)

    # Demonstrate structured exceptions
    try:
        raise SensorReadError(
            "Heart rate sensor failed to return data",
            sensor_type="heart_rate",
            sensor_id="hr-001",
            read_attempt=2,
            severity=ExceptionSeverity.MEDIUM,
        )
    except SensorError as exc:
        print(f"Caught: {exc}")
        print(f"  Error code:  {exc.error_code}")
        print(f"  Retryable:   {is_retryable(exc)}")
        print(f"  Recovery:    {get_recovery_hint(exc)}")
        print(f"  Context:     {exc.context}")

    try:
        raise SafetyViolationError(
            "AI response contains potentially harmful content",
            violation_type="harmful_content",
            safety_score=0.3,
        )
    except AISystemError as exc:
        print(f"\nCaught: {exc}")
        print(f"  Severity: {exc.severity.value}")

    # Test wrapping
    original = ValueError("bad value")
    wrapped = wrap_exception(original, DataError, "Sensor data out of range",
                             data_source="accelerometer")
    print(f"\nWrapped: {wrapped}")
    print(f"  Cause: {wrapped.cause}")

    print("\nAll exception tests passed.")
