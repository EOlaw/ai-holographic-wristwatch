"""
Motion Tracking Sensor Package — AI Holographic Wristwatch

Exports all motion tracking drivers:
- 3-axis accelerometer (LSM6DSO)
- 3-axis gyroscope (LSM6DSO angular rate)
- 3-axis magnetometer / compass (LIS3MDL)
- Gesture detector (wrist gestures, tap, raise-to-wake)
- Dedicated step counter with cadence, distance, and calorie tracking
"""

from .accelerometer import (
    AccelRange, AccelOutputDataRate, Orientation, TapType,
    Vector3, AccelerometerReading, AccelCalibration,
    GravityEstimator, OrientationDetector, TapDetector, FreeFallDetector,
    Accelerometer, get_accelerometer, reset_accelerometer,
    run_accelerometer_tests,
)

from .gyroscope import (
    GyroRange, GyroOutputDataRate, WristGesture,
    Quaternion, AngularVelocity, GyroscopeReading, GyroBiasEstimate,
    AttitudeIntegrator, GyroBiasEstimator, WristGestureRecognizer,
    Gyroscope, get_gyroscope, reset_gyroscope,
    run_gyroscope_tests,
)

from .magnetometer import (
    MagRange, MagOutputDataRate, CompassDirection, MagneticAnomalyType,
    MagneticVector, MagCalibration, MagnetometerReading,
    HardIronCalibrator, TiltCompensator, AnomalyDetector,
    Magnetometer, get_magnetometer, reset_magnetometer,
    run_magnetometer_tests,
)

from .gesture_detector import (
    GestureType, GestureConfidence, GestureContext,
    GestureReading, GestureTemplate, GestureEvent,
    TapGestureRecognizer, WristMotionRecognizer, ShakeDetector, RaiseToWakeDetector,
    GestureDetector, get_gesture_detector, reset_gesture_detector,
    run_gesture_detector_tests,
)

from .step_counter import (
    WalkingSpeed, VerticalMovement, StepMilestone,
    DailyStepSession, StepCounterReading, StepCounterConfig,
    AdaptivePeakDetector, CadenceTracker, MetabolicCalculator,
    StepCounter, get_step_counter, reset_step_counter,
    run_step_counter_tests,
)

__version__ = "1.0.0"
__all__ = [
    # Accelerometer
    "AccelRange", "AccelOutputDataRate", "Orientation", "TapType",
    "Vector3", "AccelerometerReading", "AccelCalibration",
    "GravityEstimator", "OrientationDetector", "TapDetector", "FreeFallDetector",
    "Accelerometer", "get_accelerometer", "reset_accelerometer",
    "run_accelerometer_tests",
    # Gyroscope
    "GyroRange", "GyroOutputDataRate", "WristGesture",
    "Quaternion", "AngularVelocity", "GyroscopeReading", "GyroBiasEstimate",
    "AttitudeIntegrator", "GyroBiasEstimator", "WristGestureRecognizer",
    "Gyroscope", "get_gyroscope", "reset_gyroscope",
    "run_gyroscope_tests",
    # Magnetometer
    "MagRange", "MagOutputDataRate", "CompassDirection", "MagneticAnomalyType",
    "MagneticVector", "MagCalibration", "MagnetometerReading",
    "HardIronCalibrator", "TiltCompensator", "AnomalyDetector",
    "Magnetometer", "get_magnetometer", "reset_magnetometer",
    "run_magnetometer_tests",
    # Gesture Detector
    "GestureType", "GestureConfidence", "GestureContext",
    "GestureReading", "GestureTemplate", "GestureEvent",
    "TapGestureRecognizer", "WristMotionRecognizer", "ShakeDetector", "RaiseToWakeDetector",
    "GestureDetector", "get_gesture_detector", "reset_gesture_detector",
    "run_gesture_detector_tests",
    # Step Counter
    "WalkingSpeed", "VerticalMovement", "StepMilestone",
    "DailyStepSession", "StepCounterReading", "StepCounterConfig",
    "AdaptivePeakDetector", "CadenceTracker", "MetabolicCalculator",
    "StepCounter", "get_step_counter", "reset_step_counter",
    "run_step_counter_tests",
]
