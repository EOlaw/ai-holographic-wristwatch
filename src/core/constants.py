"""
System-Wide Constants for AI Holographic Wristwatch

This module is the single source of truth for every magic number, threshold,
limit, string key, and enumeration shared across the entire system. No other
module should define constants that belong here. Import from this module only
— never hard-code values inline.

All classes are namespace-only (no instantiation needed) unless stated otherwise.
"""

from enum import Enum, IntEnum, Flag, auto
from typing import Final, Tuple


# ============================================================================
# System Identity
# ============================================================================

class SystemConstants:
    """Project-wide identity and version constants."""
    PROJECT_NAME: Final[str] = "AI Holographic Wristwatch"
    SYSTEM_ID: Final[str] = "ai-holographic-wristwatch"
    FIRMWARE_VERSION: Final[str] = "1.0.0"
    API_VERSION: Final[str] = "v1"
    PROTOCOL_VERSION: Final[int] = 1
    BUILD_TARGET: Final[str] = "production"

    # Timing
    STARTUP_TIMEOUT_SECONDS: Final[float] = 30.0
    SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 10.0
    WATCHDOG_INTERVAL_SECONDS: Final[float] = 5.0

    # Resource limits
    MAX_CONCURRENT_SESSIONS: Final[int] = 4
    MAX_BACKGROUND_TASKS: Final[int] = 16
    MAX_EVENT_QUEUE_SIZE: Final[int] = 1024


# ============================================================================
# Display & Holographic Constants
# ============================================================================

class DisplayConstants:
    """Holographic and OLED display specifications."""

    # Holographic projection
    HOLO_PROJECTION_WIDTH_MM: Final[float] = 200.0       # max horizontal span
    HOLO_PROJECTION_HEIGHT_MM: Final[float] = 150.0      # max vertical span
    HOLO_PROJECTION_DEPTH_MM: Final[float] = 100.0       # max depth range
    HOLO_SAFE_DISTANCE_MM: Final[float] = 100.0          # min eye distance
    HOLO_FRAME_RATE_HZ: Final[int] = 60                  # hologram refresh rate
    HOLO_RESOLUTION_H: Final[int] = 1920                 # horizontal pixels
    HOLO_RESOLUTION_V: Final[int] = 1080                 # vertical pixels
    HOLO_COLOR_DEPTH_BITS: Final[int] = 24               # bits per pixel
    HOLO_MAX_BRIGHTNESS_MW: Final[float] = 50.0          # laser power ceiling
    HOLO_MIN_BRIGHTNESS_MW: Final[float] = 0.5           # minimum visible power
    HOLO_LASER_WAVELENGTHS_NM: Final[Tuple[int, ...]] = (638, 520, 450)  # R, G, B

    # OLED watchface display
    OLED_WIDTH_PX: Final[int] = 396
    OLED_HEIGHT_PX: Final[int] = 484
    OLED_PPI: Final[int] = 326
    OLED_REFRESH_RATE_HZ: Final[int] = 60
    OLED_AOD_REFRESH_RATE_HZ: Final[int] = 1             # always-on display
    OLED_MAX_BRIGHTNESS_NITS: Final[int] = 2000
    OLED_MIN_BRIGHTNESS_NITS: Final[int] = 1

    # Rendering pipeline
    TARGET_RENDER_TIME_MS: Final[float] = 16.67          # 60 fps budget
    MAX_RENDER_TIME_MS: Final[float] = 33.3              # 30 fps fallback
    DEPTH_BUFFER_BITS: Final[int] = 24
    ANTI_ALIASING_SAMPLES: Final[int] = 4

    # Eye safety (IEC 60825-1 Class 1 limits)
    MAX_SAFE_IRRADIANCE_MW_CM2: Final[float] = 1.0
    BLINK_REFLEX_OVERRIDE_THRESHOLD_MS: Final[float] = 250.0


# ============================================================================
# Sensor Constants
# ============================================================================

class SensorConstants:
    """Sampling rates, ranges, and calibration parameters for all sensors."""

    # Heart rate (PPG)
    HR_SAMPLING_RATE_HZ: Final[int] = 100
    HR_MIN_BPM: Final[int] = 20
    HR_MAX_BPM: Final[int] = 300
    HR_FILTER_CUTOFF_HZ: Final[float] = 4.0
    HR_WINDOW_SECONDS: Final[int] = 10
    HR_CONFIDENCE_THRESHOLD: Final[float] = 0.7

    # SpO2
    SPO2_SAMPLING_RATE_HZ: Final[int] = 25
    SPO2_MIN_PERCENT: Final[float] = 70.0
    SPO2_MAX_PERCENT: Final[float] = 100.0
    SPO2_ALERT_THRESHOLD_PERCENT: Final[float] = 90.0
    SPO2_LED_WAVELENGTHS_NM: Final[Tuple[int, int]] = (660, 940)  # red, IR

    # ECG
    ECG_SAMPLING_RATE_HZ: Final[int] = 512
    ECG_NOTCH_FILTER_HZ: Final[int] = 50                # or 60 Hz by locale
    ECG_HIGH_PASS_CUTOFF_HZ: Final[float] = 0.5
    ECG_LOW_PASS_CUTOFF_HZ: Final[float] = 150.0
    ECG_R_PEAK_MIN_AMPLITUDE_MV: Final[float] = 0.1

    # Skin temperature
    SKIN_TEMP_SAMPLING_RATE_HZ: Final[float] = 1.0
    SKIN_TEMP_MIN_C: Final[float] = 25.0
    SKIN_TEMP_MAX_C: Final[float] = 42.0
    SKIN_TEMP_RESOLUTION_C: Final[float] = 0.01

    # Accelerometer
    ACCEL_SAMPLING_RATE_HZ: Final[int] = 200
    ACCEL_RANGE_G: Final[float] = 16.0                  # ±16g
    ACCEL_RESOLUTION_BITS: Final[int] = 14
    ACCEL_STEP_THRESHOLD_G: Final[float] = 0.15
    ACCEL_FALL_THRESHOLD_G: Final[float] = 3.0
    ACCEL_IMPACT_THRESHOLD_G: Final[float] = 8.0

    # Gyroscope
    GYRO_SAMPLING_RATE_HZ: Final[int] = 200
    GYRO_RANGE_DPS: Final[float] = 2000.0               # ±2000 deg/s
    GYRO_RESOLUTION_BITS: Final[int] = 16
    GYRO_NOISE_RMS_DPS: Final[float] = 0.07

    # Magnetometer
    MAG_SAMPLING_RATE_HZ: Final[int] = 100
    MAG_RANGE_UT: Final[float] = 4900.0                 # ±4900 µT
    MAG_RESOLUTION_BITS: Final[int] = 16

    # Environmental
    BARO_SAMPLING_RATE_HZ: Final[float] = 1.0
    BARO_RANGE_HPA: Final[Tuple[float, float]] = (300.0, 1100.0)
    BARO_ALTITUDE_ACCURACY_M: Final[float] = 0.5
    UV_SAMPLING_RATE_HZ: Final[float] = 0.2             # every 5 seconds

    # Microphone
    MIC_SAMPLING_RATE_HZ: Final[int] = 16000
    MIC_BIT_DEPTH: Final[int] = 16
    MIC_CHANNELS: Final[int] = 2                        # stereo for beamforming
    MIC_WAKE_WORD_TIMEOUT_MS: Final[int] = 2000
    MIC_VAD_THRESHOLD_DB: Final[float] = -40.0

    # Sensor fusion
    FUSION_RATE_HZ: Final[int] = 100
    KALMAN_PROCESS_NOISE: Final[float] = 1e-5
    KALMAN_MEASUREMENT_NOISE: Final[float] = 1e-4
    COMPLEMENTARY_FILTER_ALPHA: Final[float] = 0.98     # gyro trust factor

    # Quality thresholds
    MIN_SIGNAL_QUALITY_PERCENT: Final[float] = 60.0
    SENSOR_TIMEOUT_SECONDS: Final[float] = 5.0
    MAX_CALIBRATION_ATTEMPTS: Final[int] = 3
    RECALIBRATION_INTERVAL_HOURS: Final[float] = 24.0


# ============================================================================
# AI System Constants
# ============================================================================

class AIConstants:
    """AI model parameters, context limits, and decision thresholds."""

    # Language model
    MAX_CONTEXT_TOKENS: Final[int] = 4096
    MAX_OUTPUT_TOKENS: Final[int] = 512
    DEFAULT_TEMPERATURE: Final[float] = 0.7
    MIN_TEMPERATURE: Final[float] = 0.0
    MAX_TEMPERATURE: Final[float] = 2.0
    TOP_P_DEFAULT: Final[float] = 0.95
    TOP_K_DEFAULT: Final[int] = 40
    FREQUENCY_PENALTY: Final[float] = 0.0
    PRESENCE_PENALTY: Final[float] = 0.0

    # Conversation
    MAX_CONVERSATION_HISTORY: Final[int] = 50           # turns
    CONVERSATION_TIMEOUT_SECONDS: Final[int] = 300      # 5-minute idle
    MIN_RESPONSE_CONFIDENCE: Final[float] = 0.6
    CLARIFICATION_THRESHOLD: Final[float] = 0.4         # ask for clarification below
    MAX_CLARIFICATION_ATTEMPTS: Final[int] = 2

    # Intent recognition
    INTENT_CONFIDENCE_THRESHOLD: Final[float] = 0.75
    ENTITY_CONFIDENCE_THRESHOLD: Final[float] = 0.70
    SENTIMENT_NEUTRAL_RANGE: Final[Tuple[float, float]] = (-0.1, 0.1)

    # Personality engine
    PERSONALITY_ADAPTATION_RATE: Final[float] = 0.01   # slow drift
    EMOTIONAL_DECAY_RATE: Final[float] = 0.95          # per interaction
    MAX_PERSONALITY_DRIFT: Final[float] = 0.3          # from baseline
    EMPATHY_RESPONSE_WEIGHT: Final[float] = 0.4

    # Memory systems
    EPISODIC_MEMORY_CAPACITY: Final[int] = 10000        # episodes
    SEMANTIC_MEMORY_CAPACITY: Final[int] = 100000       # facts
    SHORT_TERM_MEMORY_SECONDS: Final[int] = 300         # working memory TTL
    LONG_TERM_MEMORY_RETENTION_DAYS: Final[int] = 365
    MEMORY_CONSOLIDATION_INTERVAL_HOURS: Final[int] = 8 # during sleep

    # On-device inference
    EDGE_INFERENCE_TIMEOUT_MS: Final[float] = 200.0    # max latency on-device
    CLOUD_INFERENCE_TIMEOUT_MS: Final[float] = 2000.0  # max latency cloud
    INFERENCE_BATCH_SIZE: Final[int] = 1               # real-time = no batching
    MODEL_CACHE_SIZE_MB: Final[int] = 512

    # Safety
    SAFETY_SCORE_THRESHOLD: Final[float] = 0.85        # block below this
    BIAS_DETECTION_THRESHOLD: Final[float] = 0.15
    HALLUCINATION_DETECTION_CONFIDENCE: Final[float] = 0.6
    MAX_RETRY_ON_SAFETY_VIOLATION: Final[int] = 2


# ============================================================================
# Network & Communication Constants
# ============================================================================

class NetworkConstants:
    """Timeout, retry, protocol, and endpoint constants."""

    # General HTTP
    HTTP_CONNECT_TIMEOUT_SECONDS: Final[float] = 5.0
    HTTP_READ_TIMEOUT_SECONDS: Final[float] = 30.0
    HTTP_MAX_RETRIES: Final[int] = 3
    HTTP_RETRY_BACKOFF_BASE: Final[float] = 2.0         # exponential backoff
    HTTP_RETRY_BACKOFF_MAX_SECONDS: Final[float] = 32.0
    HTTP_MAX_CONNECTIONS: Final[int] = 20
    HTTP_KEEPALIVE_TIMEOUT_SECONDS: Final[int] = 30

    # WebSocket
    WS_PING_INTERVAL_SECONDS: Final[float] = 20.0
    WS_PING_TIMEOUT_SECONDS: Final[float] = 10.0
    WS_MAX_MESSAGE_SIZE_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB
    WS_RECONNECT_MAX_ATTEMPTS: Final[int] = 10
    WS_RECONNECT_DELAY_SECONDS: Final[float] = 1.0

    # Bluetooth LE
    BLE_SCAN_TIMEOUT_SECONDS: Final[float] = 10.0
    BLE_CONNECTION_TIMEOUT_SECONDS: Final[float] = 10.0
    BLE_MTU_SIZE_BYTES: Final[int] = 512
    BLE_SUPERVISION_TIMEOUT_MS: Final[int] = 4000
    BLE_CONNECTION_INTERVAL_MS: Final[float] = 15.0
    BLE_SERVICE_UUID: Final[str] = "00001234-0000-1000-8000-00805F9B34FB"

    # Wi-Fi
    WIFI_RSSI_EXCELLENT_DBM: Final[int] = -50
    WIFI_RSSI_GOOD_DBM: Final[int] = -65
    WIFI_RSSI_FAIR_DBM: Final[int] = -75
    WIFI_RSSI_POOR_DBM: Final[int] = -85

    # Sync
    SYNC_INTERVAL_SECONDS: Final[int] = 60
    SYNC_BATCH_SIZE_RECORDS: Final[int] = 100
    SYNC_MAX_PAYLOAD_BYTES: Final[int] = 1024 * 1024    # 1 MB
    SYNC_CONFLICT_RESOLUTION_STRATEGY: Final[str] = "server_wins"

    # Rate limiting
    API_RATE_LIMIT_PER_MINUTE: Final[int] = 60
    AI_RATE_LIMIT_PER_MINUTE: Final[int] = 20
    SENSOR_UPLOAD_RATE_LIMIT_PER_MINUTE: Final[int] = 10


# ============================================================================
# Security Constants
# ============================================================================

class SecurityConstants:
    """Cryptographic parameters, session management, and access control."""

    # Encryption
    AES_KEY_SIZE_BITS: Final[int] = 256
    RSA_KEY_SIZE_BITS: Final[int] = 4096
    ECDSA_CURVE: Final[str] = "secp256r1"
    PBKDF2_ITERATIONS: Final[int] = 600_000
    BCRYPT_ROUNDS: Final[int] = 12
    SALT_SIZE_BYTES: Final[int] = 32
    IV_SIZE_BYTES: Final[int] = 16
    GCM_TAG_SIZE_BYTES: Final[int] = 16
    ENCRYPTION_ALGORITHM: Final[str] = "AES-256-GCM"

    # Key management
    KEY_ROTATION_INTERVAL_DAYS: Final[int] = 90
    SESSION_KEY_TTL_HOURS: Final[int] = 24
    REFRESH_TOKEN_TTL_DAYS: Final[int] = 30
    ACCESS_TOKEN_TTL_MINUTES: Final[int] = 60
    MAX_ACTIVE_SESSIONS: Final[int] = 5

    # Authentication
    BIOMETRIC_CONFIDENCE_THRESHOLD: Final[float] = 0.90
    PIN_MIN_LENGTH: Final[int] = 6
    PIN_MAX_ATTEMPTS: Final[int] = 10
    LOCKOUT_DURATION_SECONDS: Final[int] = 300
    ANTI_SPOOFING_THRESHOLD: Final[float] = 0.85

    # Secure storage
    SECURE_ELEMENT_SLOT_COUNT: Final[int] = 16
    HSM_KEY_LABEL_PREFIX: Final[str] = "ai_wristwatch"

    # Communication security
    TLS_MIN_VERSION: Final[str] = "TLSv1.3"
    CERTIFICATE_VALIDITY_DAYS: Final[int] = 365
    HSTS_MAX_AGE_SECONDS: Final[int] = 31536000

    # Tamper detection
    TAMPER_EVENT_LOG_SIZE: Final[int] = 1000
    SECURE_BOOT_HASH_ALGORITHM: Final[str] = "SHA-256"
    INTEGRITY_CHECK_INTERVAL_SECONDS: Final[int] = 3600


# ============================================================================
# Health & Wellness Constants
# ============================================================================

class HealthConstants:
    """Clinical reference ranges, alert thresholds, and wellness parameters."""

    # Heart rate
    HR_RESTING_NORMAL_LOW: Final[int] = 50
    HR_RESTING_NORMAL_HIGH: Final[int] = 100
    HR_BRADYCARDIA_THRESHOLD: Final[int] = 50
    HR_TACHYCARDIA_THRESHOLD: Final[int] = 100
    HR_CRITICAL_LOW: Final[int] = 40
    HR_CRITICAL_HIGH: Final[int] = 180

    # Blood oxygen
    SPO2_NORMAL_LOW: Final[float] = 95.0
    SPO2_HYPOXIA_THRESHOLD: Final[float] = 90.0
    SPO2_SEVERE_HYPOXIA_THRESHOLD: Final[float] = 85.0

    # Blood pressure (mmHg)
    BP_SYSTOLIC_NORMAL_LOW: Final[int] = 90
    BP_SYSTOLIC_NORMAL_HIGH: Final[int] = 120
    BP_SYSTOLIC_HIGH_WARNING: Final[int] = 140
    BP_SYSTOLIC_CRISIS_THRESHOLD: Final[int] = 180
    BP_DIASTOLIC_NORMAL_LOW: Final[int] = 60
    BP_DIASTOLIC_NORMAL_HIGH: Final[int] = 80
    BP_DIASTOLIC_HIGH_WARNING: Final[int] = 90

    # Body temperature (°C)
    BODY_TEMP_NORMAL_LOW_C: Final[float] = 36.1
    BODY_TEMP_NORMAL_HIGH_C: Final[float] = 37.2
    BODY_TEMP_FEVER_THRESHOLD_C: Final[float] = 38.0
    BODY_TEMP_HIGH_FEVER_C: Final[float] = 39.5
    BODY_TEMP_HYPOTHERMIA_C: Final[float] = 35.0

    # Respiratory rate
    RESP_RATE_NORMAL_LOW: Final[int] = 12
    RESP_RATE_NORMAL_HIGH: Final[int] = 20
    RESP_RATE_TACHYPNEA: Final[int] = 20
    RESP_RATE_BRADYPNEA: Final[int] = 12

    # Activity
    DAILY_STEP_GOAL: Final[int] = 10000
    ACTIVE_MINUTES_GOAL_PER_DAY: Final[int] = 30
    SEDENTARY_ALERT_INTERVAL_MINUTES: Final[int] = 60
    VO2_MAX_EXCELLENT_ML_KG_MIN: Final[float] = 55.0

    # Sleep
    SLEEP_DETECTION_THRESHOLD_MINUTES: Final[int] = 20  # min duration
    DEEP_SLEEP_TARGET_PERCENT: Final[float] = 20.0
    REM_SLEEP_TARGET_PERCENT: Final[float] = 20.0
    SLEEP_QUALITY_EXCELLENT_THRESHOLD: Final[float] = 85.0

    # Stress (via HRV)
    HRV_RMSSD_STRESSED_MS: Final[float] = 20.0          # low HRV = stressed
    HRV_RMSSD_RECOVERED_MS: Final[float] = 50.0
    STRESS_ALERT_CONSECUTIVE_READINGS: Final[int] = 5


# ============================================================================
# Power & Battery Constants
# ============================================================================

class BatteryConstants:
    """Battery state thresholds and power mode parameters."""

    BATTERY_CAPACITY_MAH: Final[int] = 600              # device capacity
    BATTERY_FULL_VOLTAGE_V: Final[float] = 4.35
    BATTERY_EMPTY_VOLTAGE_V: Final[float] = 3.0
    BATTERY_CRITICAL_VOLTAGE_V: Final[float] = 3.2

    # State of charge thresholds
    SOC_FULL_PERCENT: Final[int] = 100
    SOC_HIGH_PERCENT: Final[int] = 80
    SOC_NORMAL_PERCENT: Final[int] = 50
    SOC_LOW_PERCENT: Final[int] = 20
    SOC_CRITICAL_PERCENT: Final[int] = 10
    SOC_SHUTDOWN_PERCENT: Final[int] = 3

    # Charging
    MAX_CHARGE_CURRENT_MA: Final[int] = 1000
    TRICKLE_CHARGE_CURRENT_MA: Final[int] = 50
    WIRELESS_CHARGE_MAX_WATTS: Final[float] = 15.0
    SOLAR_HARVEST_MAX_MW: Final[float] = 50.0

    # Thermal
    BATTERY_TEMP_MIN_CHARGE_C: Final[float] = 0.0
    BATTERY_TEMP_MAX_CHARGE_C: Final[float] = 45.0
    BATTERY_TEMP_WARNING_C: Final[float] = 40.0
    BATTERY_TEMP_CRITICAL_C: Final[float] = 50.0

    # Power modes (mW estimated draw)
    POWER_MODE_ACTIVE_MW: Final[float] = 200.0
    POWER_MODE_NORMAL_MW: Final[float] = 100.0
    POWER_MODE_LOW_MW: Final[float] = 30.0
    POWER_MODE_ULTRA_LOW_MW: Final[float] = 10.0
    POWER_MODE_STANDBY_MW: Final[float] = 2.0

    # Estimated runtimes (hours)
    RUNTIME_ACTIVE_HOURS: Final[float] = 3.0
    RUNTIME_NORMAL_HOURS: Final[float] = 6.0
    RUNTIME_LOW_POWER_HOURS: Final[float] = 20.0


# ============================================================================
# Privacy & Compliance Constants
# ============================================================================

class PrivacyConstants:
    """Data retention, consent tiers, and regulatory compliance settings."""

    # Data retention defaults (days)
    HEALTH_DATA_RETENTION_DAYS: Final[int] = 3650       # 10 years
    CONVERSATION_RETENTION_DAYS: Final[int] = 365
    ANALYTICS_RETENTION_DAYS: Final[int] = 90
    SECURITY_LOG_RETENTION_DAYS: Final[int] = 180
    DEBUG_LOG_RETENTION_DAYS: Final[int] = 30
    CRASH_REPORT_RETENTION_DAYS: Final[int] = 60

    # Anonymization
    ANONYMIZATION_K_FACTOR: Final[int] = 5              # k-anonymity
    DIFFERENTIAL_PRIVACY_EPSILON: Final[float] = 1.0
    PII_SCRUB_FIELDS: Final[Tuple[str, ...]] = (
        "name", "email", "phone", "address", "ip_address",
        "device_id", "biometric_hash", "location"
    )

    # Consent levels
    CONSENT_REQUIRED_FOR_CLOUD_SYNC: Final[bool] = True
    CONSENT_REQUIRED_FOR_ANALYTICS: Final[bool] = True
    CONSENT_REQUIRED_FOR_RESEARCH: Final[bool] = True
    CONSENT_REQUIRED_FOR_EMERGENCY: Final[bool] = False  # implied consent

    # GDPR / CCPA
    DATA_EXPORT_MAX_DAYS: Final[int] = 30               # respond within 30 days
    DATA_DELETION_MAX_DAYS: Final[int] = 30
    BREACH_NOTIFICATION_HOURS: Final[int] = 72


# ============================================================================
# UI / UX Constants
# ============================================================================

class UIConstants:
    """User interface timing, haptic patterns, and display thresholds."""

    # Animation
    ANIMATION_DURATION_MS: Final[int] = 250
    TRANSITION_DURATION_MS: Final[int] = 300
    FADE_DURATION_MS: Final[int] = 150

    # Touch / gesture
    TAP_MAX_DURATION_MS: Final[int] = 200
    LONG_PRESS_DURATION_MS: Final[int] = 500
    SWIPE_MIN_VELOCITY_PX_S: Final[float] = 300.0
    SWIPE_MIN_DISTANCE_PX: Final[int] = 50
    DOUBLE_TAP_MAX_GAP_MS: Final[int] = 300

    # Haptic patterns (duration in ms, intensity 0.0–1.0)
    HAPTIC_LIGHT_DURATION_MS: Final[int] = 10
    HAPTIC_MEDIUM_DURATION_MS: Final[int] = 20
    HAPTIC_STRONG_DURATION_MS: Final[int] = 50
    HAPTIC_ALERT_PATTERN_MS: Final[Tuple[int, ...]] = (50, 100, 50, 100, 50)

    # Notification
    NOTIFICATION_DISPLAY_SECONDS: Final[int] = 5
    NOTIFICATION_MAX_QUEUED: Final[int] = 20
    CRITICAL_ALERT_REPEAT_SECONDS: Final[int] = 60

    # Voice
    VOICE_RESPONSE_MAX_WORDS: Final[int] = 50           # brevity on watch
    TTS_SPEAKING_RATE_WPM: Final[int] = 170
    TTS_PITCH_DEFAULT: Final[float] = 1.0

    # Screen timeout
    SCREEN_ON_WRIST_RAISE_SECONDS: Final[int] = 5
    SCREEN_ON_TOUCH_SECONDS: Final[int] = 10
    SCREEN_OFF_AMBIENT_SECONDS: Final[int] = 0          # AOD stays on


# ============================================================================
# Application Layer Constants
# ============================================================================

class AppConstants:
    """Application-level configuration, feature keys, and storage keys."""

    # Storage keys
    KEY_USER_PROFILE: Final[str] = "user_profile"
    KEY_DEVICE_CONFIG: Final[str] = "device_config"
    KEY_AI_PERSONALITY: Final[str] = "ai_personality"
    KEY_HEALTH_BASELINE: Final[str] = "health_baseline"
    KEY_AUTH_TOKEN: Final[str] = "auth_token"
    KEY_SYNC_TIMESTAMP: Final[str] = "sync_timestamp"

    # Feature flag keys (canonical names)
    FEATURE_HOLOGRAPHIC_DISPLAY: Final[str] = "feature.holographic_display"
    FEATURE_AI_HEALTH_COACHING: Final[str] = "feature.ai_health_coaching"
    FEATURE_SMART_HOME_CONTROL: Final[str] = "feature.smart_home_control"
    FEATURE_ADVANCED_BIOMETRICS: Final[str] = "feature.advanced_biometrics"
    FEATURE_CLOUD_AI: Final[str] = "feature.cloud_ai"
    FEATURE_FEDERATED_LEARNING: Final[str] = "feature.federated_learning"
    FEATURE_EMERGENCY_SOS: Final[str] = "feature.emergency_sos"

    # Cache namespaces
    CACHE_NS_AI_RESPONSES: Final[str] = "ai:responses"
    CACHE_NS_SENSOR_DATA: Final[str] = "sensor:data"
    CACHE_NS_HEALTH_METRICS: Final[str] = "health:metrics"
    CACHE_NS_USER_PREFS: Final[str] = "user:prefs"

    # Worker pool sizing
    IO_THREAD_POOL_SIZE: Final[int] = 8
    CPU_THREAD_POOL_SIZE: Final[int] = 4
    ASYNC_EVENT_LOOP_COUNT: Final[int] = 1


# ============================================================================
# Enumerations
# ============================================================================

class DeviceState(Enum):
    """Overall device operating state machine."""
    BOOTING = "booting"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    LOW_POWER = "low_power"
    ULTRA_LOW_POWER = "ultra_low_power"
    CHARGING = "charging"
    UPDATING = "updating"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class PowerMode(Enum):
    """Device power operating modes."""
    HIGH_PERFORMANCE = "high_performance"
    NORMAL = "normal"
    BALANCED = "balanced"
    LOW_POWER = "low_power"
    ULTRA_LOW_POWER = "ultra_low_power"
    EMERGENCY_RESERVE = "emergency_reserve"


class ConnectivityState(Enum):
    """Network and Bluetooth connectivity states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    PAIRED = "paired"
    SYNCING = "syncing"
    ERROR = "error"


class PrivacyLevel(IntEnum):
    """User privacy preference tiers."""
    OPEN = 0            # all features, full cloud sync
    STANDARD = 1        # cloud sync with anonymization
    ENHANCED = 2        # on-device only, opt-in sharing
    MAXIMUM = 3         # fully offline, no data leaves device


class SensorStatus(Enum):
    """Sensor operational status."""
    UNINITIALIZED = "uninitialized"
    CALIBRATING = "calibrating"
    ACTIVE = "active"
    DEGRADED = "degraded"
    ERROR = "error"
    OFFLINE = "offline"


class AIMode(Enum):
    """AI assistant operating modes."""
    CONVERSATIONAL = "conversational"
    TASK_EXECUTION = "task_execution"
    HEALTH_MONITORING = "health_monitoring"
    EMERGENCY = "emergency"
    SILENT = "silent"
    LEARNING = "learning"


class AlertSeverity(Enum):
    """Alert severity for notifications and health events."""
    INFO = "info"
    WARNING = "warning"
    URGENT = "urgent"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DataCategory(Flag):
    """Data categories for privacy tagging and consent management."""
    NONE = 0
    HEALTH_VITALS = auto()
    HEALTH_ACTIVITY = auto()
    HEALTH_SLEEP = auto()
    LOCATION = auto()
    CONVERSATION = auto()
    BIOMETRIC = auto()
    BEHAVIORAL = auto()
    DEVICE_DIAGNOSTICS = auto()
    ALL = (HEALTH_VITALS | HEALTH_ACTIVITY | HEALTH_SLEEP |
           LOCATION | CONVERSATION | BIOMETRIC |
           BEHAVIORAL | DEVICE_DIAGNOSTICS)


# ============================================================================
# Validation Helpers
# ============================================================================

def is_valid_heart_rate(bpm: float) -> bool:
    """Return True if heart rate value is within physiologically plausible range."""
    return SensorConstants.HR_MIN_BPM <= bpm <= SensorConstants.HR_MAX_BPM


def is_valid_spo2(percent: float) -> bool:
    """Return True if SpO2 value is within valid range."""
    return (HealthConstants.SPO2_NORMAL_LOW - 30) <= percent <= 100.0


def is_valid_battery_soc(percent: float) -> bool:
    """Return True if battery state-of-charge is valid."""
    return 0.0 <= percent <= 100.0


def get_alert_severity_for_hr(bpm: float) -> AlertSeverity:
    """Classify heart rate value into alert severity tier."""
    if bpm <= HealthConstants.HR_CRITICAL_LOW or bpm >= HealthConstants.HR_CRITICAL_HIGH:
        return AlertSeverity.CRITICAL
    elif bpm < HealthConstants.HR_BRADYCARDIA_THRESHOLD or bpm > HealthConstants.HR_TACHYCARDIA_THRESHOLD:
        return AlertSeverity.WARNING
    return AlertSeverity.INFO


def get_alert_severity_for_spo2(percent: float) -> AlertSeverity:
    """Classify SpO2 value into alert severity tier."""
    if percent < HealthConstants.SPO2_SEVERE_HYPOXIA_THRESHOLD:
        return AlertSeverity.EMERGENCY
    elif percent < HealthConstants.SPO2_HYPOXIA_THRESHOLD:
        return AlertSeverity.CRITICAL
    elif percent < HealthConstants.SPO2_NORMAL_LOW:
        return AlertSeverity.WARNING
    return AlertSeverity.INFO


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    # Namespace classes
    "SystemConstants", "DisplayConstants", "SensorConstants",
    "AIConstants", "NetworkConstants", "SecurityConstants",
    "HealthConstants", "BatteryConstants", "PrivacyConstants",
    "UIConstants", "AppConstants",

    # Enumerations
    "DeviceState", "PowerMode", "ConnectivityState", "PrivacyLevel",
    "SensorStatus", "AIMode", "AlertSeverity", "DataCategory",

    # Validation helpers
    "is_valid_heart_rate", "is_valid_spo2", "is_valid_battery_soc",
    "get_alert_severity_for_hr", "get_alert_severity_for_spo2",
]

if __name__ == "__main__":
    print("AI Holographic Wristwatch — Constants Module")
    print("=" * 55)
    print(f"System: {SystemConstants.PROJECT_NAME} v{SystemConstants.FIRMWARE_VERSION}")
    print(f"Display: {DisplayConstants.HOLO_RESOLUTION_H}x{DisplayConstants.HOLO_RESOLUTION_V}"
          f" @ {DisplayConstants.HOLO_FRAME_RATE_HZ}fps")
    print(f"Heart rate range: {SensorConstants.HR_MIN_BPM}–{SensorConstants.HR_MAX_BPM} BPM")
    print(f"Battery capacity: {BatteryConstants.BATTERY_CAPACITY_MAH} mAh")
    print(f"Encryption: {SecurityConstants.ENCRYPTION_ALGORITHM} "
          f"({SecurityConstants.AES_KEY_SIZE_BITS}-bit)")

    # Validate helpers
    assert is_valid_heart_rate(72)
    assert not is_valid_heart_rate(5)
    assert get_alert_severity_for_hr(200) == AlertSeverity.CRITICAL
    assert get_alert_severity_for_spo2(98.0) == AlertSeverity.INFO
    print("\nAll constant validation checks passed.")
