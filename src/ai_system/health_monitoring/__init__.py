"""AI Health Monitoring package for the holographic wristwatch system."""
from __future__ import annotations

from src.ai_system.health_monitoring.vital_signs.continuous_monitoring import (
    VitalSignsMonitor,
    get_vital_signs_monitor,
)
from src.ai_system.health_monitoring.vital_signs.anomaly_detection import (
    AnomalyDetector,
    AnomalyType,
    get_anomaly_detector,
)
from src.ai_system.health_monitoring.vital_signs.emergency_detection import (
    EmergencyDetector,
    EmergencyLevel,
    get_emergency_detector,
)

__all__ = [
    "VitalSignsMonitor",
    "get_vital_signs_monitor",
    "AnomalyDetector",
    "AnomalyType",
    "get_anomaly_detector",
    "EmergencyDetector",
    "EmergencyLevel",
    "get_emergency_detector",
]
