"""Charging systems sub-package for the AI Holographic Wristwatch.

Exports charging controllers, MPPT solar integration, kinetic harvesting,
wireless Qi charging receiver, and safety watchdog.
"""
from __future__ import annotations

from src.hardware.power_management.charging_systems.fast_charging import (
    ChargingMode,
    ChargingPhase,
    ChargingStatus,
    ChargingController,
    get_charging_controller,
)
from src.hardware.power_management.charging_systems.kinetic_charging import (
    HarvestSource,
    HarvestReading,
    EnergyHarvester,
    get_energy_harvester,
)
from src.hardware.power_management.charging_systems.solar_integration import (
    SolarState,
    SolarReading,
    SolarIntegration,
    get_solar_integration,
)
from src.hardware.power_management.charging_systems.wireless_charging import (
    WirelessChargingState,
    WirelessStatus,
    WirelessChargingReceiver,
    get_wireless_charging_receiver,
)
from src.hardware.power_management.charging_systems.charging_safety import (
    SafetyEvent,
    SafetyReading,
    ChargingSafety,
    get_charging_safety,
)

__all__ = [
    "ChargingMode",
    "ChargingPhase",
    "ChargingStatus",
    "ChargingController",
    "get_charging_controller",
    "HarvestSource",
    "HarvestReading",
    "EnergyHarvester",
    "get_energy_harvester",
    "SolarState",
    "SolarReading",
    "SolarIntegration",
    "get_solar_integration",
    "WirelessChargingState",
    "WirelessStatus",
    "WirelessChargingReceiver",
    "get_wireless_charging_receiver",
    "SafetyEvent",
    "SafetyReading",
    "ChargingSafety",
    "get_charging_safety",
]
