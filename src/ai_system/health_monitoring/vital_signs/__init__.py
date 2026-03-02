"""Vital signs monitoring subpackage."""
from __future__ import annotations

from src.ai_system.health_monitoring.vital_signs.continuous_monitoring import (
    VitalSignsMonitor,
    VitalReading,
    get_vital_signs_monitor,
)
from src.ai_system.health_monitoring.vital_signs.anomaly_detection import (
    AnomalyDetector,
    AnomalyType,
    AnomalyEvent,
    get_anomaly_detector,
)
from src.ai_system.health_monitoring.vital_signs.trend_analysis import (
    TrendAnalyzer,
    TrendDirection,
    get_trend_analyzer,
)
from src.ai_system.health_monitoring.vital_signs.emergency_detection import (
    EmergencyDetector,
    EmergencyLevel,
    get_emergency_detector,
)

__all__ = [
    "VitalSignsMonitor",
    "VitalReading",
    "get_vital_signs_monitor",
    "AnomalyDetector",
    "AnomalyType",
    "AnomalyEvent",
    "get_anomaly_detector",
    "TrendAnalyzer",
    "TrendDirection",
    "get_trend_analyzer",
    "EmergencyDetector",
    "EmergencyLevel",
    "get_emergency_detector",
]
