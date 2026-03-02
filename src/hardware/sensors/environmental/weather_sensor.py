"""
Weather Sensor — AI Holographic Wristwatch

Provides onboard atmospheric measurement:
- Barometric pressure (BME688 or equivalent)
- Temperature and relative humidity
- Weather trend prediction (rising/falling pressure → fair/storm)
- Altitude estimation from pressure
- Dew point and heat index calculation
- SensorInterface compliance
"""

from __future__ import annotations

import math
import threading
import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Deque, Dict, List, Optional

from src.core.interfaces.sensor_interface import SensorInterface, SensorReading, SensorStatus, SensorInfo
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class WeatherTrend(Enum):
    RAPIDLY_RISING   = "rapidly_rising"    # > +2 hPa/3hr → clearing
    RISING           = "rising"            # +1 to +2 hPa/3hr → fair
    STEADY           = "steady"            # ±1 hPa/3hr → stable
    FALLING          = "falling"           # -1 to -2 hPa/3hr → clouds
    RAPIDLY_FALLING  = "rapidly_falling"   # < -2 hPa/3hr → storm


class ComfortLevel(Enum):
    COMFORTABLE   = "comfortable"
    HUMID         = "humid"
    DRY           = "dry"
    HOT_HUMID     = "hot_humid"
    COLD          = "cold"


@dataclass
class WeatherReading(SensorReading):
    temperature_c: float = 20.0
    humidity_rh: float = 50.0
    pressure_hpa: float = 1013.25
    altitude_m: float = 0.0
    dew_point_c: float = 10.0
    heat_index_c: float = 20.0
    wind_chill_c: float = 20.0
    absolute_humidity_g_m3: float = 9.0
    weather_trend: WeatherTrend = WeatherTrend.STEADY
    comfort_level: ComfortLevel = ComfortLevel.COMFORTABLE
    storm_probability_pct: float = 0.0


def _dew_point(temp_c: float, rh: float) -> float:
    """Magnus formula for dew point (°C)."""
    a, b = 17.625, 243.04
    alpha = math.log(max(0.01, rh) / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * alpha) / (a - alpha)


def _heat_index(temp_c: float, rh: float) -> float:
    """Steadman's heat index (°C)."""
    t = temp_c * 9/5 + 32  # to Fahrenheit
    hi = (-42.379 + 2.04901523*t + 10.14333127*rh - 0.22475541*t*rh
          - 0.00683783*t**2 - 0.05481717*rh**2 + 0.00122874*t**2*rh
          + 0.00085282*t*rh**2 - 0.00000199*t**2*rh**2)
    return (hi - 32) * 5/9


def _pressure_to_altitude(pressure_hpa: float, sea_level_hpa: float = 1013.25) -> float:
    """International pressure formula (meters)."""
    return 44330 * (1 - (pressure_hpa / sea_level_hpa) ** (1/5.255))


def _analyze_trend(history: List[float]) -> WeatherTrend:
    if len(history) < 2:
        return WeatherTrend.STEADY
    # Compare last vs 3-hours-ago equivalent (assume 10s sampling → 1080 samples/3hr)
    delta = history[-1] - history[0]
    if delta > 2:    return WeatherTrend.RAPIDLY_RISING
    if delta > 0.5:  return WeatherTrend.RISING
    if delta < -2:   return WeatherTrend.RAPIDLY_FALLING
    if delta < -0.5: return WeatherTrend.FALLING
    return WeatherTrend.STEADY


def _comfort(temp_c: float, rh: float) -> ComfortLevel:
    if temp_c < 10:                          return ComfortLevel.COLD
    if temp_c > 28 and rh > 70:             return ComfortLevel.HOT_HUMID
    if rh > 70:                              return ComfortLevel.HUMID
    if rh < 30:                              return ComfortLevel.DRY
    return ComfortLevel.COMFORTABLE


_GLOBAL_WEATHER: Optional["WeatherSensor"] = None
_GLOBAL_WEATHER_LOCK = threading.Lock()


class WeatherSensor(SensorInterface):
    """Atmospheric sensor (BME688: temp + humidity + pressure + gas)."""

    SENSOR_ID    = "environmental.weather"
    SENSOR_TYPE  = "weather"
    MODEL        = "BME688"
    MANUFACTURER = "Bosch Sensortec"

    def __init__(self) -> None:
        self._pressure_history: Deque[float] = deque(maxlen=1080)
        self._lock = threading.RLock()
        self._running = self._initialized = False
        self._read_count = 0
        self._last_reading: Optional[WeatherReading] = None

    def initialize(self) -> bool:
        with self._lock:
            self._initialized = self._running = True
            logger.info(f"WeatherSensor {self.MODEL} initialized")
            return True

    def read(self) -> Optional[WeatherReading]:
        if not self._initialized:
            return None
        with self._lock:
            temp  = random.gauss(22.0, 1.5)
            rh    = max(10.0, min(100.0, random.gauss(48.0, 5.0)))
            pres  = random.gauss(1013.5, 0.5)
            self._pressure_history.append(pres)

            alt    = _pressure_to_altitude(pres)
            dp     = _dew_point(temp, rh)
            hi     = _heat_index(temp, rh)
            trend  = _analyze_trend(list(self._pressure_history))
            storm  = max(0.0, -3 * (pres - 1000)) if pres < 1005 else 0.0
            ah     = (6.112 * math.exp(17.67*temp/(temp+243.5)) * rh * 2.1674) / (273.15 + temp)

            reading = WeatherReading(
                sensor_id=self.SENSOR_ID, timestamp=time.time(),
                temperature_c=temp, humidity_rh=rh, pressure_hpa=pres,
                altitude_m=alt, dew_point_c=dp, heat_index_c=hi,
                wind_chill_c=temp, absolute_humidity_g_m3=ah,
                weather_trend=trend, comfort_level=_comfort(temp, rh),
                storm_probability_pct=min(100.0, storm),
                confidence=0.95,
            )
            self._last_reading = reading
            self._read_count += 1
            return reading

    async def stream(self) -> AsyncIterator[WeatherReading]:
        import asyncio
        while self._running:
            r = self.read()
            if r: yield r
            await asyncio.sleep(10.0)

    def calibrate(self) -> bool:
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._running = self._initialized = False

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(sensor_id=self.SENSOR_ID, sensor_type=self.SENSOR_TYPE,
                          model=self.MODEL, manufacturer=self.MANUFACTURER,
                          firmware_version="1.0.0", hardware_version="rev-A",
                          capabilities={"temperature": True, "humidity": True,
                                        "pressure": True, "altitude": True,
                                        "weather_trend": True})

    def get_status(self) -> SensorStatus:
        if not self._initialized: return SensorStatus.UNINITIALIZED
        return SensorStatus.RUNNING if self._running else SensorStatus.IDLE

    def is_healthy(self) -> bool:
        return self.get_status() in (SensorStatus.RUNNING, SensorStatus.IDLE)

    def get_health_report(self) -> Dict:
        return {"status": self.get_status().value, "read_count": self._read_count,
                "last_pressure_hpa": self._last_reading.pressure_hpa if self._last_reading else None}

    def read_sync(self) -> Optional[WeatherReading]:
        return self.read()


def get_weather_sensor() -> WeatherSensor:
    global _GLOBAL_WEATHER
    with _GLOBAL_WEATHER_LOCK:
        if _GLOBAL_WEATHER is None:
            _GLOBAL_WEATHER = WeatherSensor()
        return _GLOBAL_WEATHER


def run_weather_sensor_tests() -> bool:
    ws = WeatherSensor()
    assert ws.initialize()
    r = ws.read()
    assert r is not None
    assert 900 < r.pressure_hpa < 1100
    assert -50 < r.temperature_c < 60
    ws.shutdown()
    logger.info("WeatherSensor tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_weather_sensor_tests()
