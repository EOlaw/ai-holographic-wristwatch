"""
Hardware Sensor Package — AI Holographic Wristwatch

Aggregates all sensor subsystems:
- Biometric sensors (heart rate, SpO2, temperature, stress, sleep, activity, hydration)
- Motion tracking (accelerometer, gyroscope, magnetometer, gestures, step counting)
- Environmental sensors (air quality, light, location, noise, proximity, weather)
- Audio sensors (microphone array, ANC, acoustic processing, speaker ID, voice isolation)
- Sensor fusion (EKF, calibration manager, conflict resolution, context awareness)
"""

from .biometric import (
    # Heart Rate
    HRMonitorMode, PPGSignalQuality, RRInterval, HRVMetrics,
    HeartRateReading, PPGSignalProcessor, HRVAnalyzer,
    HeartRateMonitor, get_heart_rate_monitor,
    # SpO2
    SpO2MeasurementState, OxygenationLevel, SpO2Reading,
    SpO2Calculator, BloodOxygenMonitor, get_spo2_monitor,
    # Temperature
    TemperatureZone, TemperatureReading, CoreTempEstimator,
    TemperatureSensor, get_temperature_sensor,
    # Stress
    StressLevel, AutonomicState, GSRReading, StressReading,
    GSRProcessor, StressFusionEngine, StressDetector, get_stress_detector,
    # Sleep
    SleepStage, SleepQuality, SleepEpoch, SleepSession,
    SleepStageClassifier, SleepTracker, get_sleep_tracker,
    # Activity
    ActivityType, IntensityLevel, StepData, ActivityReading,
    StepCounter, ActivityClassifier, ActivityMonitor, get_activity_monitor,
    # Hydration
    HydrationStatus, BIAMeasurement, HydrationReading,
    BIAHydrationEstimator, HydrationSensor, get_hydration_sensor,
)

from .motion_tracking import (
    # Accelerometer
    AccelRange, AccelOutputDataRate, Orientation, TapType,
    Vector3, AccelerometerReading, AccelCalibration,
    Accelerometer, get_accelerometer,
    # Gyroscope
    GyroRange, GyroOutputDataRate, WristGesture,
    Quaternion, AngularVelocity, GyroscopeReading,
    Gyroscope, get_gyroscope,
    # Magnetometer
    MagRange, MagOutputDataRate, CompassDirection, MagneticAnomalyType,
    MagneticVector, MagnetometerReading,
    Magnetometer, get_magnetometer,
    # Gesture Detector
    GestureType, GestureConfidence, GestureContext,
    GestureReading, GestureEvent,
    GestureDetector, get_gesture_detector,
    # Step Counter
    WalkingSpeed, VerticalMovement, StepMilestone,
    DailyStepSession, StepCounterReading, StepCounterConfig,
    get_step_counter,
)

from .environmental import (
    # Air Quality
    AQICategory, VOCLevel, IndoorOutdoor, AirQualityReading,
    AirQualityMonitor, get_air_quality_monitor,
    # Light
    LightEnvironment, CircadianPhase, LightReading,
    LightSensor, get_light_sensor,
    # Location
    LocationSource, TravelMode, Coordinates, Geofence, LocationReading,
    LocationSensor, get_location_sensor,
    # Noise
    NoiseEnvironment, HearingRiskLevel, NoiseLevelReading,
    NoiseLevelDetector, get_noise_level_detector,
    # Proximity
    ProximityZone, NearbyDeviceType, NearbyDevice, ProximityReading,
    ProximityScanner, get_proximity_scanner,
    # Weather
    WeatherTrend, ComfortLevel, WeatherReading,
    WeatherSensor, get_weather_sensor,
)

from .audio import (
    AudioQuality, SoundEvent, AudioFrame, MicArrayReading,
    MicrophoneArray, get_microphone_array,
    ANCMode, NoiseType, ANCProcessingResult,
    NoiseCancellation, get_noise_cancellation,
    AudioClass, VocalEmotion, AcousticFeatures, AcousticProcessingReading,
    AcousticProcessor, get_acoustic_processor,
    VerificationDecision, SpeakerProfile, SpeakerIdReading,
    SpeakerIdentifier, get_speaker_identifier,
    IsolationMode, VoiceIsolationReading,
    VoiceIsolator, get_voice_isolator,
)

from .fusion import (
    FusionMode, FusedActivityState, SensorHealthLevel,
    AttitudeState, MotionState, EnvironmentState, BiometricState, FusedSensorReading,
    SensorFusionEngine, get_sensor_fusion_engine,
    CalibrationState, CalibrationTrigger, SensorCalibrationRecord,
    CalibrationManager, get_calibration_manager,
    ConflictType, ResolutionStrategy, SensorMeasurement, ConflictReport,
    ConflictResolver, get_resolver,
    UserActivity, LocationType, SocialContext, WearState, InteractionMode, TimeOfDay,
    ContextSnapshot, ContextAwarenessEngine, get_context_awareness_engine,
)

__version__ = "1.0.0"
