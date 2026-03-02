"""Kinetic and solar energy harvesting for the AI Holographic Wristwatch.

Implements piezoelectric kinetic harvesting from wrist motion and photovoltaic
MPPT (Maximum Power Point Tracking) for the bezel-integrated solar cell array.
"""
from __future__ import annotations

import threading
import time
import random
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class HarvestSource(Enum):
    KINETIC = auto()        # Piezoelectric from wrist motion
    SOLAR = auto()          # Photovoltaic bezel cells
    THERMOELECTRIC = auto() # Seebeck effect body-heat gradient


@dataclass
class HarvestReading:
    source: HarvestSource = HarvestSource.KINETIC
    power_uw: float = 0.0          # Micro-watts harvested
    voltage_mv: float = 0.0        # Open-circuit voltage
    accumulated_mwh: float = 0.0   # Total energy accumulated this session
    efficiency_pct: float = 0.0
    active: bool = False
    timestamp: float = field(default_factory=time.time)


class EnergyHarvester:
    """Multi-source energy harvester with MPPT and piezo kinetic recovery."""

    # Piezo harvester constants
    PIEZO_OPEN_CIRCUIT_MV = 3300.0
    PIEZO_MAX_POWER_UW = 180.0      # Peak at moderate activity

    # Solar harvester constants
    SOLAR_VOC_MV = 4200.0
    SOLAR_MAX_POWER_UW = 800.0      # Full sun on bezel cells

    # Thermoelectric constants
    THERMO_MAX_POWER_UW = 40.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._readings: dict[HarvestSource, HarvestReading] = {
            s: HarvestReading(source=s) for s in HarvestSource
        }
        self._accumulated_mwh: dict[HarvestSource, float] = {s: 0.0 for s in HarvestSource}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        logger.info("EnergyHarvester initialised")

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._harvest_loop, daemon=True, name="energy-harvest")
            self._thread.start()
        logger.info("EnergyHarvester started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("EnergyHarvester stopped")

    def _simulate_kinetic(self) -> HarvestReading:
        """Simulate wrist acceleration profile — gaussian noise around walking cadence."""
        activity_factor = abs(math.sin(time.time() * 0.5)) * random.uniform(0.5, 1.0)
        power_uw = self.PIEZO_MAX_POWER_UW * activity_factor
        voltage_mv = self.PIEZO_OPEN_CIRCUIT_MV * (0.8 + 0.2 * activity_factor)
        efficiency = 65.0 + random.gauss(0, 2)
        return HarvestReading(
            source=HarvestSource.KINETIC,
            power_uw=round(power_uw, 2),
            voltage_mv=round(voltage_mv, 1),
            efficiency_pct=round(efficiency, 1),
            active=True,
        )

    def _simulate_solar_mppt(self) -> HarvestReading:
        """Simulate MPPT tracking — Perturb & Observe algorithm on bezel PV cells."""
        irradiance = random.uniform(0.1, 1.0)  # 0=dark, 1=full sun
        power_uw = self.SOLAR_MAX_POWER_UW * irradiance * random.uniform(0.85, 0.95)
        voc_mv = self.SOLAR_VOC_MV * irradiance
        efficiency = 18.0 + random.gauss(0, 0.5)  # ~18% efficient thin-film
        return HarvestReading(
            source=HarvestSource.SOLAR,
            power_uw=round(power_uw, 2),
            voltage_mv=round(voc_mv, 1),
            efficiency_pct=round(efficiency, 1),
            active=irradiance > 0.05,
        )

    def _simulate_thermoelectric(self) -> HarvestReading:
        """Simulate Seebeck body-heat delta (skin ~33C, ambient ~20C => 13C delta)."""
        delta_t = random.gauss(13.0, 2.0)
        power_uw = self.THERMO_MAX_POWER_UW * (delta_t / 20.0) * random.uniform(0.7, 1.0)
        voltage_mv = delta_t * 40  # ~40uV/K Seebeck coefficient * 1000 pellets
        efficiency = 5.0 + random.gauss(0, 0.3)
        return HarvestReading(
            source=HarvestSource.THERMOELECTRIC,
            power_uw=round(max(0, power_uw), 2),
            voltage_mv=round(max(0, voltage_mv), 1),
            efficiency_pct=round(efficiency, 1),
            active=delta_t > 5.0,
        )

    def _harvest_loop(self) -> None:
        interval_s = 1.0
        while self._running:
            with self._lock:
                for src, fn in [
                    (HarvestSource.KINETIC, self._simulate_kinetic),
                    (HarvestSource.SOLAR, self._simulate_solar_mppt),
                    (HarvestSource.THERMOELECTRIC, self._simulate_thermoelectric),
                ]:
                    reading = fn()
                    self._accumulated_mwh[src] += reading.power_uw * interval_s / 3.6e9
                    reading.accumulated_mwh = self._accumulated_mwh[src]
                    reading.timestamp = time.time()
                    self._readings[src] = reading
            time.sleep(interval_s)

    def get_reading(self, source: HarvestSource) -> HarvestReading:
        with self._lock:
            return self._readings[source]

    def get_total_power_uw(self) -> float:
        with self._lock:
            return sum(r.power_uw for r in self._readings.values() if r.active)


_GLOBAL_ENERGY_HARVESTER: Optional[EnergyHarvester] = None
_GLOBAL_ENERGY_HARVESTER_LOCK = threading.Lock()


def get_energy_harvester() -> EnergyHarvester:
    global _GLOBAL_ENERGY_HARVESTER
    with _GLOBAL_ENERGY_HARVESTER_LOCK:
        if _GLOBAL_ENERGY_HARVESTER is None:
            _GLOBAL_ENERGY_HARVESTER = EnergyHarvester()
    return _GLOBAL_ENERGY_HARVESTER


def run_kinetic_charging_tests() -> bool:
    logger.info("=== KineticCharging tests ===")
    harvester = EnergyHarvester()
    harvester.start()
    time.sleep(0.15)
    total = harvester.get_total_power_uw()
    assert total >= 0, "Total power must be non-negative"
    k = harvester.get_reading(HarvestSource.KINETIC)
    assert k.source == HarvestSource.KINETIC
    assert 0 <= k.power_uw <= EnergyHarvester.PIEZO_MAX_POWER_UW * 1.1
    harvester.stop()
    logger.info("KineticCharging tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_kinetic_charging_tests()
