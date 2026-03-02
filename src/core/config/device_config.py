"""
Device Configuration for AI Holographic Wristwatch

Typed configuration classes for all device hardware subsystems: holographic
display, sensor array, power management, communication protocols, and physical
form factor. Used by hardware abstraction drivers at initialization time.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..constants import (
    DisplayConstants, SensorConstants, BatteryConstants,
    NetworkConstants, PowerMode
)
from .base_config import BaseConfiguration, ConfigValidationResult


# ============================================================================
# Enumerations
# ============================================================================

class WristOrientation(Enum):
    """Which wrist and orientation the device is worn on."""
    LEFT_CROWN_RIGHT = "left_crown_right"
    LEFT_CROWN_LEFT = "left_crown_left"
    RIGHT_CROWN_RIGHT = "right_crown_right"
    RIGHT_CROWN_LEFT = "right_crown_left"


class DisplayOrientation(Enum):
    """Physical display orientation."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class WirelessStandard(Enum):
    """Wi-Fi standard."""
    WIFI_6 = "802.11ax"
    WIFI_5 = "802.11ac"
    WIFI_4 = "802.11n"


# ============================================================================
# Sub-configurations
# ============================================================================

@dataclass
class HolographicConfig(BaseConfiguration):
    """Holographic projection subsystem configuration."""
    enabled: bool = True
    projection_width_mm: float = DisplayConstants.HOLO_PROJECTION_WIDTH_MM
    projection_height_mm: float = DisplayConstants.HOLO_PROJECTION_HEIGHT_MM
    projection_depth_mm: float = DisplayConstants.HOLO_PROJECTION_DEPTH_MM
    safe_distance_mm: float = DisplayConstants.HOLO_SAFE_DISTANCE_MM
    target_fps: int = DisplayConstants.HOLO_FRAME_RATE_HZ
    resolution_h: int = DisplayConstants.HOLO_RESOLUTION_H
    resolution_v: int = DisplayConstants.HOLO_RESOLUTION_V
    max_brightness_mw: float = DisplayConstants.HOLO_MAX_BRIGHTNESS_MW
    min_brightness_mw: float = DisplayConstants.HOLO_MIN_BRIGHTNESS_MW
    auto_brightness: bool = True
    eye_tracking_enabled: bool = True
    depth_sensing_enabled: bool = True
    eye_safety_interlock: bool = True       # MUST always be True in production
    auto_off_on_removal: bool = True
    startup_calibration: bool = True

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_range(result, "projection_width_mm",
                          self.projection_width_mm, 50.0, 500.0)
        self._check_range(result, "target_fps", float(self.target_fps), 24.0, 120.0)
        self._check_range(result, "max_brightness_mw",
                          self.max_brightness_mw, 0.1,
                          DisplayConstants.HOLO_MAX_BRIGHTNESS_MW)
        if not self.eye_safety_interlock:
            result.add_issue("eye_safety_interlock", "invalid",
                             "Eye safety interlock must be enabled in production",
                             is_fatal=True)
        if self.min_brightness_mw >= self.max_brightness_mw:
            result.add_issue("brightness_range", "invalid",
                             "min_brightness_mw must be less than max_brightness_mw")
        return result


@dataclass
class OLEDConfig(BaseConfiguration):
    """OLED watchface display configuration."""
    enabled: bool = True
    width_px: int = DisplayConstants.OLED_WIDTH_PX
    height_px: int = DisplayConstants.OLED_HEIGHT_PX
    ppi: int = DisplayConstants.OLED_PPI
    refresh_rate_hz: int = DisplayConstants.OLED_REFRESH_RATE_HZ
    always_on_display: bool = True
    aod_refresh_rate_hz: int = DisplayConstants.OLED_AOD_REFRESH_RATE_HZ
    max_brightness_nits: int = DisplayConstants.OLED_MAX_BRIGHTNESS_NITS
    min_brightness_nits: int = DisplayConstants.OLED_MIN_BRIGHTNESS_NITS
    adaptive_brightness: bool = True
    auto_off_seconds: int = 10
    orientation: str = DisplayOrientation.PORTRAIT.value

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_range(result, "refresh_rate_hz", float(self.refresh_rate_hz),
                          1.0, 120.0)
        if self.min_brightness_nits >= self.max_brightness_nits:
            result.add_issue("brightness_range", "invalid",
                             "min_brightness_nits must be less than max_brightness_nits")
        return result


@dataclass
class BiometricSensorConfig(BaseConfiguration):
    """Configuration for biometric sensor suite."""
    heart_rate_enabled: bool = True
    heart_rate_sampling_hz: int = SensorConstants.HR_SAMPLING_RATE_HZ
    heart_rate_continuous: bool = True
    spo2_enabled: bool = True
    spo2_sampling_hz: int = SensorConstants.SPO2_SAMPLING_RATE_HZ
    ecg_enabled: bool = True
    ecg_sampling_hz: int = SensorConstants.ECG_SAMPLING_RATE_HZ
    skin_temp_enabled: bool = True
    skin_temp_sampling_hz: float = SensorConstants.SKIN_TEMP_SAMPLING_RATE_HZ
    blood_pressure_enabled: bool = True
    gsr_enabled: bool = True
    auto_calibrate_on_startup: bool = True
    calibration_reminder_hours: float = SensorConstants.RECALIBRATION_INTERVAL_HOURS

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_range(result, "heart_rate_sampling_hz",
                          float(self.heart_rate_sampling_hz), 1.0, 1000.0)
        return result


@dataclass
class MotionSensorConfig(BaseConfiguration):
    """Accelerometer, gyroscope, and magnetometer configuration."""
    accel_enabled: bool = True
    accel_sampling_hz: int = SensorConstants.ACCEL_SAMPLING_RATE_HZ
    accel_range_g: float = SensorConstants.ACCEL_RANGE_G
    gyro_enabled: bool = True
    gyro_sampling_hz: int = SensorConstants.GYRO_SAMPLING_RATE_HZ
    gyro_range_dps: float = SensorConstants.GYRO_RANGE_DPS
    mag_enabled: bool = True
    mag_sampling_hz: int = SensorConstants.MAG_SAMPLING_RATE_HZ
    gesture_recognition: bool = True
    fall_detection: bool = True
    step_counting: bool = True
    activity_recognition: bool = True
    fusion_rate_hz: int = SensorConstants.FUSION_RATE_HZ

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_range(result, "accel_sampling_hz",
                          float(self.accel_sampling_hz), 1.0, 800.0)
        self._check_range(result, "gyro_sampling_hz",
                          float(self.gyro_sampling_hz), 1.0, 800.0)
        return result


@dataclass
class EnvironmentalSensorConfig(BaseConfiguration):
    """Environmental sensor suite configuration."""
    barometer_enabled: bool = True
    barometer_sampling_hz: float = SensorConstants.BARO_SAMPLING_RATE_HZ
    ambient_light_enabled: bool = True
    uv_sensor_enabled: bool = True
    air_quality_enabled: bool = True
    humidity_temp_enabled: bool = True

    def validate(self) -> ConfigValidationResult:
        return ConfigValidationResult(is_valid=True)


@dataclass
class AudioConfig(BaseConfiguration):
    """Microphone and speaker configuration."""
    microphone_enabled: bool = True
    mic_sampling_hz: int = SensorConstants.MIC_SAMPLING_RATE_HZ
    mic_channels: int = SensorConstants.MIC_CHANNELS
    wake_word_detection: bool = True
    noise_cancellation: bool = True
    beamforming: bool = True
    vad_threshold_db: float = SensorConstants.MIC_VAD_THRESHOLD_DB
    speaker_enabled: bool = True
    max_speaker_volume_db: float = 85.0     # hearing safety limit
    haptic_feedback_enabled: bool = True

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_range(result, "max_speaker_volume_db",
                          self.max_speaker_volume_db, 0.0, 90.0)
        return result


@dataclass
class PowerConfig(BaseConfiguration):
    """Battery and power management configuration."""
    battery_capacity_mah: int = BatteryConstants.BATTERY_CAPACITY_MAH
    default_power_mode: str = PowerMode.NORMAL.value
    low_battery_threshold_percent: int = BatteryConstants.SOC_LOW_PERCENT
    critical_battery_threshold_percent: int = BatteryConstants.SOC_CRITICAL_PERCENT
    shutdown_threshold_percent: int = BatteryConstants.SOC_SHUTDOWN_PERCENT
    wireless_charging_enabled: bool = True
    solar_harvesting_enabled: bool = True
    kinetic_harvesting_enabled: bool = True
    max_charge_temp_c: float = BatteryConstants.BATTERY_TEMP_MAX_CHARGE_C
    thermal_throttle_temp_c: float = BatteryConstants.BATTERY_TEMP_WARNING_C
    power_saving_auto_switch: bool = True
    background_sync_on_battery: bool = False

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        if (self.critical_battery_threshold_percent >=
                self.low_battery_threshold_percent):
            result.add_issue("battery_thresholds", "invalid",
                             "critical_battery must be less than low_battery threshold")
        if self.shutdown_threshold_percent >= self.critical_battery_threshold_percent:
            result.add_issue("battery_thresholds", "invalid",
                             "shutdown_threshold must be less than critical threshold")
        self._check_range(result, "max_charge_temp_c",
                          self.max_charge_temp_c,
                          BatteryConstants.BATTERY_TEMP_MIN_CHARGE_C,
                          BatteryConstants.BATTERY_TEMP_CRITICAL_C)
        return result


@dataclass
class BluetoothConfig(BaseConfiguration):
    """Bluetooth LE configuration."""
    enabled: bool = True
    scan_timeout_seconds: float = NetworkConstants.BLE_SCAN_TIMEOUT_SECONDS
    connection_timeout_seconds: float = NetworkConstants.BLE_CONNECTION_TIMEOUT_SECONDS
    mtu_bytes: int = NetworkConstants.BLE_MTU_SIZE_BYTES
    supervision_timeout_ms: int = NetworkConstants.BLE_SUPERVISION_TIMEOUT_MS
    connection_interval_ms: float = NetworkConstants.BLE_CONNECTION_INTERVAL_MS
    auto_reconnect: bool = True
    max_paired_devices: int = 5
    secure_pairing_required: bool = True

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_range(result, "mtu_bytes", float(self.mtu_bytes), 23.0, 517.0)
        return result


@dataclass
class WiFiConfig(BaseConfiguration):
    """Wi-Fi configuration."""
    enabled: bool = True
    standard: str = WirelessStandard.WIFI_6.value
    wpa3_required: bool = True
    fast_transition_enabled: bool = True
    max_tx_power_dbm: float = 20.0
    background_scan_enabled: bool = True
    connection_timeout_seconds: float = 15.0

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_range(result, "max_tx_power_dbm", self.max_tx_power_dbm,
                          0.0, 30.0)
        return result


@dataclass
class NFCConfig(BaseConfiguration):
    """Near-Field Communication configuration."""
    enabled: bool = True
    secure_element_enabled: bool = True
    payment_enabled: bool = True
    nfc_timeout_ms: int = 5000

    def validate(self) -> ConfigValidationResult:
        return ConfigValidationResult(is_valid=True)


@dataclass
class WristwatchFormFactorConfig(BaseConfiguration):
    """Physical device and form-factor settings."""
    device_name: str = "AI Holographic Wristwatch"
    wrist_orientation: str = WristOrientation.LEFT_CROWN_RIGHT.value
    water_resistance_atm: int = 5
    haptic_feedback_intensity: float = 0.7   # 0.0–1.0
    crown_sensitivity: float = 0.6
    touchpad_sensitivity: float = 0.7
    raise_to_wake: bool = True
    flip_to_mute: bool = True

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_required(result, "device_name", self.device_name)
        self._check_range(result, "haptic_feedback_intensity",
                          self.haptic_feedback_intensity, 0.0, 1.0)
        return result


# ============================================================================
# Root Device Configuration
# ============================================================================

@dataclass
class DeviceConfig(BaseConfiguration):
    """Aggregate root for all device hardware configuration."""

    # Sub-sections
    holographic: HolographicConfig = field(default_factory=HolographicConfig)
    oled: OLEDConfig = field(default_factory=OLEDConfig)
    biometric_sensors: BiometricSensorConfig = field(default_factory=BiometricSensorConfig)
    motion_sensors: MotionSensorConfig = field(default_factory=MotionSensorConfig)
    environmental_sensors: EnvironmentalSensorConfig = field(
        default_factory=EnvironmentalSensorConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    bluetooth: BluetoothConfig = field(default_factory=BluetoothConfig)
    wifi: WiFiConfig = field(default_factory=WiFiConfig)
    nfc: NFCConfig = field(default_factory=NFCConfig)
    form_factor: WristwatchFormFactorConfig = field(
        default_factory=WristwatchFormFactorConfig)

    # Device identity
    hardware_revision: str = "r1.0"
    serial_number: Optional[str] = None
    manufacturing_date: Optional[str] = None

    def __post_init__(self):
        _subsections = {
            "holographic": HolographicConfig,
            "oled": OLEDConfig,
            "biometric_sensors": BiometricSensorConfig,
            "motion_sensors": MotionSensorConfig,
            "environmental_sensors": EnvironmentalSensorConfig,
            "audio": AudioConfig,
            "power": PowerConfig,
            "bluetooth": BluetoothConfig,
            "wifi": WiFiConfig,
            "nfc": NFCConfig,
            "form_factor": WristwatchFormFactorConfig,
        }
        for attr, cls in _subsections.items():
            value = getattr(self, attr)
            if isinstance(value, dict):
                setattr(self, attr, cls(value))

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)

        subsections = [
            ("holographic", self.holographic),
            ("oled", self.oled),
            ("biometric_sensors", self.biometric_sensors),
            ("motion_sensors", self.motion_sensors),
            ("audio", self.audio),
            ("power", self.power),
            ("bluetooth", self.bluetooth),
            ("wifi", self.wifi),
            ("form_factor", self.form_factor),
        ]

        for section_name, section in subsections:
            sub_result = section.validate()
            for issue in sub_result.issues:
                result.add_issue(f"{section_name}.{issue.field_path}",
                                 issue.issue_type, issue.message, issue.is_fatal)
            result.warnings.extend(sub_result.warnings)

        return result


# ============================================================================
# Convenience factory
# ============================================================================

def create_device_config(config_dict: Optional[Dict[str, Any]] = None) -> DeviceConfig:
    """Create a validated DeviceConfig from a raw config dict."""
    from ..exceptions import InvalidConfigError
    config = DeviceConfig(config_dict or {})
    result = config.validate()
    if not result.is_valid:
        issues_str = "; ".join(f"{i.field_path}: {i.message}"
                                for i in result.fatal_issues)
        raise InvalidConfigError(f"Device configuration is invalid: {issues_str}",
                                  config_key="device")
    return config


# ============================================================================
# Tests
# ============================================================================

def run_device_config_tests() -> None:
    print("Testing device configuration...")
    config = DeviceConfig()
    result = config.validate()
    assert result.is_valid, f"Default config invalid: {result.issues}"

    assert config.holographic.enabled is True
    assert config.power.battery_capacity_mah == BatteryConstants.BATTERY_CAPACITY_MAH
    assert config.bluetooth.secure_pairing_required is True

    # Eye-safety interlock must be enforced
    bad = HolographicConfig({"eye_safety_interlock": False})
    bad_result = bad.validate()
    assert not bad_result.is_valid

    print("  Device config tests passed.")


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "WristOrientation", "DisplayOrientation", "WirelessStandard",
    "HolographicConfig", "OLEDConfig", "BiometricSensorConfig",
    "MotionSensorConfig", "EnvironmentalSensorConfig", "AudioConfig",
    "PowerConfig", "BluetoothConfig", "WiFiConfig", "NFCConfig",
    "WristwatchFormFactorConfig", "DeviceConfig",
    "create_device_config",
]


if __name__ == "__main__":
    print("AI Holographic Wristwatch — Device Configuration Module")
    print("=" * 55)
    run_device_config_tests()
