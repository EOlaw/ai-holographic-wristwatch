"""
Security Configuration for AI Holographic Wristwatch

Typed configuration for authentication, encryption, privacy management,
and regulatory compliance. Security configuration is treated as highest-
sensitivity data: never logged in plaintext, validated strictly on startup,
and loaded with secret injection from a vault or hardware secure element.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..constants import SecurityConstants, PrivacyConstants
from .base_config import BaseConfiguration, ConfigValidationResult


# ============================================================================
# Enumerations
# ============================================================================

class AuthMethod(Enum):
    """Supported authentication methods (in decreasing preference)."""
    BIOMETRIC_FACE = "biometric_face"
    BIOMETRIC_FINGERPRINT = "biometric_fingerprint"
    BIOMETRIC_HEARTBEAT = "biometric_heartbeat"
    PIN = "pin"
    PASSPHRASE = "passphrase"
    DEVICE_PAIRING = "device_pairing"
    EMERGENCY_CODE = "emergency_code"


class EncryptionAlgorithm(Enum):
    """Supported symmetric encryption algorithms."""
    AES_256_GCM = "AES-256-GCM"
    AES_256_CBC = "AES-256-CBC"
    CHACHA20_POLY1305 = "ChaCha20-Poly1305"


class KeyStoreBackend(Enum):
    """Where encryption keys are stored."""
    SECURE_ELEMENT = "secure_element"  # Dedicated hardware security chip
    TRUSTED_EXECUTION_ENVIRONMENT = "tee"
    SOFTWARE_KEYSTORE = "software"     # Development/simulation only


class ComplianceRegion(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    PIPEDA = "pipeda"
    PDPA = "pdpa"


# ============================================================================
# Sub-configurations
# ============================================================================

@dataclass
class AuthenticationConfig(BaseConfiguration):
    """User and device authentication settings."""
    primary_method: str = AuthMethod.BIOMETRIC_FACE.value
    fallback_methods: List[str] = field(default_factory=lambda: [
        AuthMethod.BIOMETRIC_FINGERPRINT.value,
        AuthMethod.PIN.value,
    ])
    biometric_confidence_threshold: float = SecurityConstants.BIOMETRIC_CONFIDENCE_THRESHOLD
    anti_spoofing_enabled: bool = True
    anti_spoofing_threshold: float = SecurityConstants.ANTI_SPOOFING_THRESHOLD
    pin_min_length: int = SecurityConstants.PIN_MIN_LENGTH
    pin_max_attempts: int = SecurityConstants.PIN_MAX_ATTEMPTS
    lockout_duration_seconds: int = SecurityConstants.LOCKOUT_DURATION_SECONDS
    progressive_lockout: bool = True        # Increase lockout on repeated failures
    require_auth_on_wrist_remove: bool = True
    auth_timeout_minutes: int = 5           # Re-auth after idle
    device_pairing_required: bool = True    # Companion app must be paired
    secure_enrollment_flow: bool = True

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_required(result, "primary_method", self.primary_method)
        self._check_range(result, "biometric_confidence_threshold",
                          self.biometric_confidence_threshold, 0.5, 1.0)
        if self.pin_min_length < 4:
            result.add_issue("pin_min_length", "out_of_range",
                             "PIN must be at least 4 digits for security")
        if self.pin_max_attempts < 1:
            result.add_issue("pin_max_attempts", "out_of_range",
                             "Must allow at least 1 attempt")
        return result


@dataclass
class EncryptionConfig(BaseConfiguration):
    """Data-at-rest and data-in-transit encryption settings."""
    algorithm: str = EncryptionAlgorithm.AES_256_GCM.value
    key_size_bits: int = SecurityConstants.AES_KEY_SIZE_BITS
    key_store_backend: str = KeyStoreBackend.SECURE_ELEMENT.value
    key_rotation_interval_days: int = SecurityConstants.KEY_ROTATION_INTERVAL_DAYS
    session_key_ttl_hours: int = SecurityConstants.SESSION_KEY_TTL_HOURS
    pbkdf2_iterations: int = SecurityConstants.PBKDF2_ITERATIONS
    salt_size_bytes: int = SecurityConstants.SALT_SIZE_BYTES
    enable_at_rest_encryption: bool = True
    enable_in_transit_encryption: bool = True
    tls_min_version: str = SecurityConstants.TLS_MIN_VERSION
    certificate_pinning_enabled: bool = True
    hsm_key_label_prefix: str = SecurityConstants.HSM_KEY_LABEL_PREFIX

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        valid_key_sizes = (128, 192, 256)
        if self.key_size_bits not in valid_key_sizes:
            result.add_issue("key_size_bits", "invalid",
                             f"Key size must be one of {valid_key_sizes}")
        if self.pbkdf2_iterations < 100_000:
            result.add_issue("pbkdf2_iterations", "out_of_range",
                             "PBKDF2 iterations must be >= 100,000 for security",
                             is_fatal=False)
            result.add_warning("PBKDF2 iterations below recommended minimum (600,000)")
        if self.salt_size_bytes < 16:
            result.add_issue("salt_size_bytes", "out_of_range",
                             "Salt must be at least 16 bytes")
        if not self.enable_in_transit_encryption:
            result.add_issue("enable_in_transit_encryption", "invalid",
                             "In-transit encryption cannot be disabled in production")
        return result


@dataclass
class PrivacyConfig(BaseConfiguration):
    """User data privacy and consent management settings."""
    health_data_retention_days: int = PrivacyConstants.HEALTH_DATA_RETENTION_DAYS
    conversation_retention_days: int = PrivacyConstants.CONVERSATION_RETENTION_DAYS
    analytics_retention_days: int = PrivacyConstants.ANALYTICS_RETENTION_DAYS
    security_log_retention_days: int = PrivacyConstants.SECURITY_LOG_RETENTION_DAYS
    differential_privacy_enabled: bool = True
    differential_privacy_epsilon: float = PrivacyConstants.DIFFERENTIAL_PRIVACY_EPSILON
    k_anonymity_factor: int = PrivacyConstants.ANONYMIZATION_K_FACTOR
    consent_required_for_analytics: bool = True
    consent_required_for_cloud_sync: bool = True
    consent_required_for_research: bool = True
    pii_fields_to_scrub: List[str] = field(
        default_factory=lambda: list(PrivacyConstants.PII_SCRUB_FIELDS)
    )
    enable_right_to_deletion: bool = True
    enable_data_portability: bool = True
    data_export_max_days: int = PrivacyConstants.DATA_EXPORT_MAX_DAYS
    breach_notification_hours: int = PrivacyConstants.BREACH_NOTIFICATION_HOURS

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        if self.health_data_retention_days < 1:
            result.add_issue("health_data_retention_days", "out_of_range",
                             "Retention must be at least 1 day")
        if self.differential_privacy_epsilon <= 0:
            result.add_issue("differential_privacy_epsilon", "out_of_range",
                             "Epsilon must be positive")
        if self.k_anonymity_factor < 2:
            result.add_issue("k_anonymity_factor", "out_of_range",
                             "k-anonymity factor must be at least 2")
        return result


@dataclass
class TamperDetectionConfig(BaseConfiguration):
    """Physical and logical tamper detection settings."""
    enabled: bool = True
    secure_boot_enabled: bool = True
    boot_hash_algorithm: str = SecurityConstants.SECURE_BOOT_HASH_ALGORITHM
    integrity_check_interval_seconds: int = SecurityConstants.INTEGRITY_CHECK_INTERVAL_SECONDS
    tamper_event_log_size: int = SecurityConstants.TAMPER_EVENT_LOG_SIZE
    wipe_on_tamper: bool = False    # Wipe sensitive data on physical tamper
    alert_on_tamper: bool = True
    geofencing_tamper_alert: bool = False

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        if not self.secure_boot_enabled:
            result.add_issue("secure_boot_enabled", "invalid",
                             "Secure boot must be enabled in production")
        return result


@dataclass
class ComplianceConfig(BaseConfiguration):
    """Regulatory compliance settings."""
    enabled_regions: List[str] = field(default_factory=lambda: [
        ComplianceRegion.GDPR.value,
        ComplianceRegion.CCPA.value,
    ])
    medical_device_classification: str = "wellness"  # "wellness" or "medical_grade"
    audit_trail_enabled: bool = True
    audit_log_retention_days: int = PrivacyConstants.SECURITY_LOG_RETENTION_DAYS
    data_residency_region: Optional[str] = None     # None = no restriction
    third_party_data_sharing_consent: bool = False
    children_data_protection_enabled: bool = True   # COPPA
    accessibility_compliance: bool = True

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        valid_classifications = ("wellness", "medical_grade")
        if self.medical_device_classification not in valid_classifications:
            result.add_issue("medical_device_classification", "invalid",
                             f"Must be one of {valid_classifications}")
        return result


# ============================================================================
# Root Security Configuration
# ============================================================================

@dataclass
class SecurityConfig(BaseConfiguration):
    """Aggregate root for all security and privacy configuration."""

    authentication: AuthenticationConfig = field(default_factory=AuthenticationConfig)
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    tamper_detection: TamperDetectionConfig = field(default_factory=TamperDetectionConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)

    # Top-level security flags
    debug_mode_allowed: bool = False       # MUST be False in production
    allow_unencrypted_storage: bool = False
    allow_untrusted_certificates: bool = False
    enable_security_telemetry: bool = True
    max_active_sessions: int = SecurityConstants.MAX_ACTIVE_SESSIONS

    def __post_init__(self):
        _map = {
            "authentication": AuthenticationConfig,
            "encryption": EncryptionConfig,
            "privacy": PrivacyConfig,
            "tamper_detection": TamperDetectionConfig,
            "compliance": ComplianceConfig,
        }
        for attr, cls in _map.items():
            value = getattr(self, attr)
            if isinstance(value, dict):
                setattr(self, attr, cls(value))

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)

        subsections = [
            ("authentication", self.authentication),
            ("encryption", self.encryption),
            ("privacy", self.privacy),
            ("tamper_detection", self.tamper_detection),
            ("compliance", self.compliance),
        ]
        for section_name, section in subsections:
            sub = section.validate()
            for issue in sub.issues:
                result.add_issue(f"{section_name}.{issue.field_path}",
                                 issue.issue_type, issue.message, issue.is_fatal)
            result.warnings.extend(sub.warnings)

        if self.allow_unencrypted_storage:
            result.add_issue("allow_unencrypted_storage", "invalid",
                             "Unencrypted storage is not allowed in production")
        if self.allow_untrusted_certificates:
            result.add_issue("allow_untrusted_certificates", "invalid",
                             "Untrusted certificates are not allowed in production")

        return result


# ============================================================================
# Convenience factory
# ============================================================================

def create_security_config(config_dict: Optional[Dict[str, Any]] = None) -> SecurityConfig:
    """Create a validated SecurityConfig."""
    from ..exceptions import InvalidConfigError
    config = SecurityConfig(config_dict or {})
    result = config.validate()
    if not result.is_valid:
        issues_str = "; ".join(f"{i.field_path}: {i.message}"
                                for i in result.fatal_issues)
        raise InvalidConfigError(f"Security configuration is invalid: {issues_str}",
                                  config_key="security")
    return config


# ============================================================================
# Tests
# ============================================================================

def run_security_config_tests() -> None:
    print("Testing security configuration...")
    config = SecurityConfig()
    result = config.validate()
    assert result.is_valid, f"Default config invalid: {result.issues}"

    assert config.encryption.key_size_bits == 256
    assert config.authentication.biometric_confidence_threshold >= 0.9
    assert config.tamper_detection.secure_boot_enabled is True

    bad = SecurityConfig({"encryption": {"pbkdf2_iterations": 1000}})
    bad_result = bad.validate()
    assert len(bad_result.warnings) > 0

    bad_enc = EncryptionConfig({"enable_in_transit_encryption": False})
    enc_result = bad_enc.validate()
    assert not enc_result.is_valid

    print("  Security config tests passed.")


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "AuthMethod", "EncryptionAlgorithm", "KeyStoreBackend", "ComplianceRegion",
    "AuthenticationConfig", "EncryptionConfig", "PrivacyConfig",
    "TamperDetectionConfig", "ComplianceConfig", "SecurityConfig",
    "create_security_config",
]


if __name__ == "__main__":
    print("AI Holographic Wristwatch — Security Configuration Module")
    print("=" * 55)
    run_security_config_tests()
