"""
Configuration Package for AI Holographic Wristwatch

Exports every configuration class, factory function, and infrastructure
primitive from the config sub-system. Consumers should import exclusively
from this package rather than from individual config modules.

Usage:
    from src.core.config import AppConfig, get_config
    from src.core.config import AIConfig, DeviceConfig, SecurityConfig
    from src.core.config import get_feature_flags, FlagState
"""

# ── Base infrastructure ───────────────────────────────────────────────────────
from .base_config import (
    Environment,
    ConfigFormat,
    ConfigValidationIssue,
    ConfigValidationResult,
    ConfigSnapshot,
    BaseConfiguration,
    ConfigLoader,
    ConfigWatcher,
    AppConfig,
    get_config,
    reset_global_config,
    config_override,
)

# ── AI configuration ──────────────────────────────────────────────────────────
from .ai_config import (
    ModelBackend,
    PersonalityProfile,
    ModelConfig,
    PersonalityConfig,
    ConversationConfig,
    LearningConfig,
    SafetyConfig,
    AIConfig,
    create_ai_config,
)

# ── Device / hardware configuration ──────────────────────────────────────────
from .device_config import (
    WristOrientation,
    DisplayOrientation,
    WirelessStandard,
    HolographicConfig,
    OLEDConfig,
    BiometricSensorConfig,
    MotionSensorConfig,
    EnvironmentalSensorConfig,
    AudioConfig,
    PowerConfig,
    BluetoothConfig,
    WiFiConfig,
    NFCConfig,
    WristwatchFormFactorConfig,
    DeviceConfig,
    create_device_config,
)

# ── Security & privacy configuration ─────────────────────────────────────────
from .security_config import (
    AuthMethod,
    EncryptionAlgorithm,
    KeyStoreBackend,
    ComplianceRegion,
    AuthenticationConfig,
    EncryptionConfig,
    PrivacyConfig,
    TamperDetectionConfig,
    ComplianceConfig,
    SecurityConfig,
    create_security_config,
)

# ── Feature flags ─────────────────────────────────────────────────────────────
from .feature_flags import (
    FlagState,
    FlagCategory,
    OverrideSource,
    FeatureFlag,
    FlagEvaluationEvent,
    FlagOverrideRecord,
    FeatureFlagRegistry,
    FeatureFlagsConfig,
    FeatureFlagManager,
    get_feature_flags,
    reset_feature_flags,
)


__version__ = "1.0.0"
__all__ = [
    # Base
    "Environment", "ConfigFormat",
    "ConfigValidationIssue", "ConfigValidationResult", "ConfigSnapshot",
    "BaseConfiguration", "ConfigLoader", "ConfigWatcher",
    "AppConfig", "get_config", "reset_global_config", "config_override",

    # AI
    "ModelBackend", "PersonalityProfile",
    "ModelConfig", "PersonalityConfig", "ConversationConfig",
    "LearningConfig", "SafetyConfig", "AIConfig",
    "create_ai_config",

    # Device
    "WristOrientation", "DisplayOrientation", "WirelessStandard",
    "HolographicConfig", "OLEDConfig", "BiometricSensorConfig",
    "MotionSensorConfig", "EnvironmentalSensorConfig", "AudioConfig",
    "PowerConfig", "BluetoothConfig", "WiFiConfig", "NFCConfig",
    "WristwatchFormFactorConfig", "DeviceConfig",
    "create_device_config",

    # Security
    "AuthMethod", "EncryptionAlgorithm", "KeyStoreBackend", "ComplianceRegion",
    "AuthenticationConfig", "EncryptionConfig", "PrivacyConfig",
    "TamperDetectionConfig", "ComplianceConfig", "SecurityConfig",
    "create_security_config",

    # Feature flags
    "FlagState", "FlagCategory", "OverrideSource",
    "FeatureFlag", "FlagEvaluationEvent", "FlagOverrideRecord",
    "FeatureFlagRegistry", "FeatureFlagsConfig", "FeatureFlagManager",
    "get_feature_flags", "reset_feature_flags",
]
