"""
Environmental Sensor Package — AI Holographic Wristwatch

Exports all environmental sensor drivers:
- Air quality (VOC, eCO₂, PM2.5, AQI)
- Ambient light (lux, UV, color temperature, circadian)
- Location (GPS/Wi-Fi/cell fusion, geofencing)
- Noise level (SPL, dB-A, hearing risk, noise dose)
- Proximity (IR, BLE, UWB, privacy alert)
- Weather (temperature, humidity, pressure, forecast trend)
"""

from .air_quality_monitor import (
    AQICategory, VOCLevel, IndoorOutdoor,
    ParticulateMatter, AirQualityReading,
    AirQualityMonitor, get_air_quality_monitor,
    run_air_quality_monitor_tests,
)

from .light_sensor import (
    LightEnvironment, CircadianPhase,
    LightReading,
    LightSensor, get_light_sensor,
    run_light_sensor_tests,
)

from .location_sensor import (
    LocationSource, TravelMode, GeofenceEvent,
    Coordinates, Geofence, LocationReading,
    LocationSensor, get_location_sensor,
    run_location_sensor_tests,
)

from .noise_level_detector import (
    NoiseEnvironment, HearingRiskLevel,
    NoiseLevelReading,
    NoiseLevelDetector, get_noise_level_detector,
    run_noise_level_detector_tests,
)

from .proximity_scanner import (
    ProximityZone, NearbyDeviceType,
    NearbyDevice, ProximityReading,
    ProximityScanner, get_proximity_scanner,
    run_proximity_scanner_tests,
)

from .weather_sensor import (
    WeatherTrend, ComfortLevel,
    WeatherReading,
    WeatherSensor, get_weather_sensor,
    run_weather_sensor_tests,
)

__version__ = "1.0.0"
__all__ = [
    # Air Quality
    "AQICategory", "VOCLevel", "IndoorOutdoor",
    "ParticulateMatter", "AirQualityReading",
    "AirQualityMonitor", "get_air_quality_monitor", "run_air_quality_monitor_tests",
    # Light
    "LightEnvironment", "CircadianPhase", "LightReading",
    "LightSensor", "get_light_sensor", "run_light_sensor_tests",
    # Location
    "LocationSource", "TravelMode", "GeofenceEvent",
    "Coordinates", "Geofence", "LocationReading",
    "LocationSensor", "get_location_sensor", "run_location_sensor_tests",
    # Noise
    "NoiseEnvironment", "HearingRiskLevel", "NoiseLevelReading",
    "NoiseLevelDetector", "get_noise_level_detector", "run_noise_level_detector_tests",
    # Proximity
    "ProximityZone", "NearbyDeviceType", "NearbyDevice", "ProximityReading",
    "ProximityScanner", "get_proximity_scanner", "run_proximity_scanner_tests",
    # Weather
    "WeatherTrend", "ComfortLevel", "WeatherReading",
    "WeatherSensor", "get_weather_sensor", "run_weather_sensor_tests",
]
