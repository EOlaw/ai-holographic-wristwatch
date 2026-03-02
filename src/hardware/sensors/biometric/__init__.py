"""
Biometric Sensor Package for AI Holographic Wristwatch

Exports all biometric sensor drivers: heart rate, SpO2, skin temperature,
stress (GSR), sleep staging, activity monitoring, and hydration sensing.
"""

from .heart_rate_monitor import (
    HRMonitorMode, PPGSignalQuality,
    RRInterval, HRVMetrics, HeartRateReading,
    PPGSignalProcessor, HRVAnalyzer,
    HeartRateMonitor, get_heart_rate_monitor,
)

from .blood_oxygen_monitor import (
    SpO2MeasurementState, OxygenationLevel,
    SpO2Reading, SpO2Calculator,
    BloodOxygenMonitor, get_spo2_monitor,
)

from .temperature_sensor import (
    TemperatureZone, TemperatureReading,
    CoreTempEstimator, TemperatureSensor,
    get_temperature_sensor,
)

from .stress_detector import (
    StressLevel, AutonomicState,
    GSRReading, StressReading,
    GSRProcessor, StressFusionEngine,
    StressDetector, get_stress_detector,
)

from .sleep_tracker import (
    SleepStage, SleepQuality,
    SleepEpoch, SleepSession,
    SleepStageClassifier, SleepTracker,
    get_sleep_tracker,
)

from .activity_monitor import (
    ActivityType, IntensityLevel,
    StepData, ActivityReading,
    StepCounter, ActivityClassifier,
    ActivityMonitor, get_activity_monitor,
)

from .hydration_sensor import (
    HydrationStatus, BIAMeasurement, HydrationReading,
    BIAHydrationEstimator, HydrationSensor,
    get_hydration_sensor,
)

__version__ = "1.0.0"
__all__ = [
    # Heart Rate
    "HRMonitorMode", "PPGSignalQuality", "RRInterval", "HRVMetrics",
    "HeartRateReading", "PPGSignalProcessor", "HRVAnalyzer",
    "HeartRateMonitor", "get_heart_rate_monitor",
    # SpO2
    "SpO2MeasurementState", "OxygenationLevel", "SpO2Reading",
    "SpO2Calculator", "BloodOxygenMonitor", "get_spo2_monitor",
    # Temperature
    "TemperatureZone", "TemperatureReading", "CoreTempEstimator",
    "TemperatureSensor", "get_temperature_sensor",
    # Stress
    "StressLevel", "AutonomicState", "GSRReading", "StressReading",
    "GSRProcessor", "StressFusionEngine", "StressDetector", "get_stress_detector",
    # Sleep
    "SleepStage", "SleepQuality", "SleepEpoch", "SleepSession",
    "SleepStageClassifier", "SleepTracker", "get_sleep_tracker",
    # Activity
    "ActivityType", "IntensityLevel", "StepData", "ActivityReading",
    "StepCounter", "ActivityClassifier", "ActivityMonitor", "get_activity_monitor",
    # Hydration
    "HydrationStatus", "BIAMeasurement", "HydrationReading",
    "BIAHydrationEstimator", "HydrationSensor", "get_hydration_sensor",
]
