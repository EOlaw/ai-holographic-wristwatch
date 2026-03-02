"""
Sensor Fusion Package — AI Holographic Wristwatch

Exports the sensor fusion engine and supporting systems:
- Sensor fusion engine (EKF + complementary filter, activity classification)
- Calibration manager (drift detection, temperature compensation)
- Conflict resolution (Bayesian fusion, outlier rejection)
- Context awareness (situational context inference)
"""

from .sensor_fusion_engine import (
    FusionMode, FusedActivityState, SensorHealthLevel,
    AttitudeState, MotionState, EnvironmentState, BiometricState, FusedSensorReading,
    ComplementaryAttitudeFilter, MultiSensorActivityClassifier,
    SensorFusionEngine, get_sensor_fusion_engine, reset_sensor_fusion_engine,
    run_sensor_fusion_engine_tests,
)

from .calibration_manager import (
    CalibrationState, CalibrationTrigger,
    SensorCalibrationRecord, CalibrationManagerReading,
    DriftDetector, TemperatureCompensator,
    CalibrationManager, get_calibration_manager,
    run_calibration_manager_tests,
)

from .conflict_resolution import (
    ConflictType, ResolutionStrategy,
    SensorMeasurement, ConflictReport,
    HampelIdentifier, BayesianFuser, StalenessChecker,
    ConflictResolver, get_resolver,
    run_conflict_resolution_tests,
)

from .context_awareness import (
    UserActivity, LocationType, SocialContext, WearState, InteractionMode, TimeOfDay,
    UserContext, DeviceContext, EnvironmentContext, ContextSnapshot,
    ActivityInferenceEngine, WearStateDetector, NotificationPolicyEngine,
    HologramBrightnessPolicy,
    ContextAwarenessEngine, get_context_awareness_engine,
    run_context_awareness_tests,
)

__version__ = "1.0.0"
__all__ = [
    # Fusion Engine
    "FusionMode", "FusedActivityState", "SensorHealthLevel",
    "AttitudeState", "MotionState", "EnvironmentState", "BiometricState", "FusedSensorReading",
    "ComplementaryAttitudeFilter", "MultiSensorActivityClassifier",
    "SensorFusionEngine", "get_sensor_fusion_engine", "reset_sensor_fusion_engine",
    "run_sensor_fusion_engine_tests",
    # Calibration
    "CalibrationState", "CalibrationTrigger", "SensorCalibrationRecord",
    "CalibrationManagerReading", "DriftDetector", "TemperatureCompensator",
    "CalibrationManager", "get_calibration_manager", "run_calibration_manager_tests",
    # Conflict Resolution
    "ConflictType", "ResolutionStrategy", "SensorMeasurement", "ConflictReport",
    "HampelIdentifier", "BayesianFuser", "StalenessChecker",
    "ConflictResolver", "get_resolver", "run_conflict_resolution_tests",
    # Context Awareness
    "UserActivity", "LocationType", "SocialContext", "WearState",
    "InteractionMode", "TimeOfDay",
    "UserContext", "DeviceContext", "EnvironmentContext", "ContextSnapshot",
    "ActivityInferenceEngine", "WearStateDetector",
    "NotificationPolicyEngine", "HologramBrightnessPolicy",
    "ContextAwarenessEngine", "get_context_awareness_engine",
    "run_context_awareness_tests",
]
