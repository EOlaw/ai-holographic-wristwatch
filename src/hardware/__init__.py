"""
Hardware Package — AI Holographic Wristwatch

Top-level hardware abstraction layer. Provides access to all hardware subsystems:

- sensors/     : Biometric, motion, environmental, audio, and sensor fusion
- holographic/ : Laser projector, rendering pipeline, hologram generation,
                 interaction tracking, and display calibration
- power_management/ : Battery, charging, thermal, and energy harvesting
- communication/    : BLE, Wi-Fi, NFC, cellular, UWB, and data sync
- wristwatch/       : Chassis, display system, input, miniaturization, and strap

Architecture:
    All hardware drivers implement SensorInterface or a hardware-specific ABC.
    Access is via singleton factories (get_X()) to ensure single hardware ownership.
    The SensorFusionEngine aggregates all sensor data into a unified FusedSensorReading.
"""

from .sensors import (
    # Biometric
    HeartRateMonitor, get_heart_rate_monitor,
    BloodOxygenMonitor, get_spo2_monitor,
    TemperatureSensor, get_temperature_sensor,
    StressDetector, get_stress_detector,
    SleepTracker, get_sleep_tracker,
    ActivityMonitor, get_activity_monitor,
    HydrationSensor, get_hydration_sensor,
    # Motion Tracking
    Accelerometer, get_accelerometer,
    Gyroscope, get_gyroscope,
    Magnetometer, get_magnetometer,
    GestureDetector, get_gesture_detector,
    get_step_counter,
    # Environmental
    AirQualityMonitor, get_air_quality_monitor,
    LightSensor, get_light_sensor,
    LocationSensor, get_location_sensor,
    NoiseLevelDetector, get_noise_level_detector,
    ProximityScanner, get_proximity_scanner,
    WeatherSensor, get_weather_sensor,
    # Audio
    MicrophoneArray, get_microphone_array,
    NoiseCancellation, get_noise_cancellation,
    AcousticProcessor, get_acoustic_processor,
    SpeakerIdentifier, get_speaker_identifier,
    VoiceIsolator, get_voice_isolator,
    # Fusion
    SensorFusionEngine, get_sensor_fusion_engine,
    CalibrationManager, get_calibration_manager,
    ContextAwarenessEngine, get_context_awareness_engine,
    # Key Types
    FusedSensorReading, FusedActivityState, FusionMode,
    ContextSnapshot, UserActivity,
)

__version__ = "1.0.0"

__all__ = [
    # Biometric sensors
    "HeartRateMonitor", "get_heart_rate_monitor",
    "BloodOxygenMonitor", "get_spo2_monitor",
    "TemperatureSensor", "get_temperature_sensor",
    "StressDetector", "get_stress_detector",
    "SleepTracker", "get_sleep_tracker",
    "ActivityMonitor", "get_activity_monitor",
    "HydrationSensor", "get_hydration_sensor",
    # Motion
    "Accelerometer", "get_accelerometer",
    "Gyroscope", "get_gyroscope",
    "Magnetometer", "get_magnetometer",
    "GestureDetector", "get_gesture_detector",
    "get_step_counter",
    # Environmental
    "AirQualityMonitor", "get_air_quality_monitor",
    "LightSensor", "get_light_sensor",
    "LocationSensor", "get_location_sensor",
    "NoiseLevelDetector", "get_noise_level_detector",
    "ProximityScanner", "get_proximity_scanner",
    "WeatherSensor", "get_weather_sensor",
    # Audio
    "MicrophoneArray", "get_microphone_array",
    "NoiseCancellation", "get_noise_cancellation",
    "AcousticProcessor", "get_acoustic_processor",
    "SpeakerIdentifier", "get_speaker_identifier",
    "VoiceIsolator", "get_voice_isolator",
    # Fusion
    "SensorFusionEngine", "get_sensor_fusion_engine",
    "CalibrationManager", "get_calibration_manager",
    "ContextAwarenessEngine", "get_context_awareness_engine",
    # Key types
    "FusedSensorReading", "FusedActivityState", "FusionMode",
    "ContextSnapshot", "UserActivity",
]
