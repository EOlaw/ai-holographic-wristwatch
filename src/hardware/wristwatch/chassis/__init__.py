"""Chassis subpackage: frame, materials, shock, thermal, and waterproofing."""
from src.hardware.wristwatch.chassis.frame_controller import (
    FrameState, FrameReading, FrameController, get_frame_controller,
)
from src.hardware.wristwatch.chassis.material_specs import (
    MaterialType, MaterialSpec, MaterialSpecsDB, get_material_specs_db,
)
from src.hardware.wristwatch.chassis.shock_protection import (
    ImpactSeverity, ImpactEvent, ShockProtectionSystem, get_shock_protection_system,
)
from src.hardware.wristwatch.chassis.thermal_management import (
    ThermalPath, ThermalZone, ThermalReading, ChassisThermalManager,
    get_chassis_thermal_manager,
)
from src.hardware.wristwatch.chassis.waterproofing import (
    WaterResistanceLevel, SealStatus, SealIntegrityReading,
    WaterproofingMonitor, get_waterproofing_monitor,
)

__all__ = [
    "FrameState", "FrameReading", "FrameController", "get_frame_controller",
    "MaterialType", "MaterialSpec", "MaterialSpecsDB", "get_material_specs_db",
    "ImpactSeverity", "ImpactEvent", "ShockProtectionSystem", "get_shock_protection_system",
    "ThermalPath", "ThermalZone", "ThermalReading", "ChassisThermalManager",
    "get_chassis_thermal_manager",
    "WaterResistanceLevel", "SealStatus", "SealIntegrityReading",
    "WaterproofingMonitor", "get_waterproofing_monitor",
]
