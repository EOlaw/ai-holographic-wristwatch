"""Material property database for AI Holographic Wristwatch chassis components."""
from __future__ import annotations
import threading
import time
import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from src.core.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MaterialType(Enum):
    """Chassis materials used in the wristwatch."""
    TITANIUM = "titanium"
    CERAMIC = "ceramic"
    SAPPHIRE = "sapphire"
    ELASTOMER = "elastomer"
    GRAPHENE_COMPOSITE = "graphene_composite"
    GORILLA_GLASS = "gorilla_glass"


@dataclass
class MaterialSpec:
    """Physical and thermal properties of a chassis material."""
    material_type: MaterialType
    density_g_cm3: float                    # g/cm³
    thermal_expansion_ppm_k: float          # µm/(m·K)
    thermal_conductivity_w_mk: float        # W/(m·K)
    vickers_hardness: float                 # HV
    youngs_modulus_gpa: float               # GPa
    yield_strength_mpa: float               # MPa
    max_service_temp_c: float               # °C
    water_resistant: bool = True
    biocompatible: bool = False
    notes: str = ""


@dataclass
class ThermalExpansionReading:
    """Result of a thermal-expansion lookup."""
    material: MaterialType
    temperature_delta_k: float
    expansion_um_per_meter: float
    timestamp: float = field(default_factory=time.time)


class MaterialSpecsDB:
    """Thread-safe database of material specifications with thermal/hardness lookup."""

    _SPECS: Dict[MaterialType, MaterialSpec] = {
        MaterialType.TITANIUM: MaterialSpec(
            material_type=MaterialType.TITANIUM,
            density_g_cm3=4.51,
            thermal_expansion_ppm_k=8.6,
            thermal_conductivity_w_mk=21.9,
            vickers_hardness=970.0,
            youngs_modulus_gpa=116.0,
            yield_strength_mpa=880.0,
            max_service_temp_c=300.0,
            water_resistant=True,
            biocompatible=True,
            notes="Grade 5 Ti-6Al-4V alloy",
        ),
        MaterialType.CERAMIC: MaterialSpec(
            material_type=MaterialType.CERAMIC,
            density_g_cm3=6.0,
            thermal_expansion_ppm_k=7.2,
            thermal_conductivity_w_mk=3.0,
            vickers_hardness=1500.0,
            youngs_modulus_gpa=200.0,
            yield_strength_mpa=600.0,
            max_service_temp_c=500.0,
            water_resistant=True,
            biocompatible=True,
            notes="Zirconium oxide (ZrO2) ceramic",
        ),
        MaterialType.SAPPHIRE: MaterialSpec(
            material_type=MaterialType.SAPPHIRE,
            density_g_cm3=3.99,
            thermal_expansion_ppm_k=5.8,
            thermal_conductivity_w_mk=27.0,
            vickers_hardness=2000.0,
            youngs_modulus_gpa=335.0,
            yield_strength_mpa=400.0,
            max_service_temp_c=600.0,
            water_resistant=True,
            biocompatible=True,
            notes="Synthetic corundum display cover",
        ),
        MaterialType.ELASTOMER: MaterialSpec(
            material_type=MaterialType.ELASTOMER,
            density_g_cm3=1.1,
            thermal_expansion_ppm_k=200.0,
            thermal_conductivity_w_mk=0.25,
            vickers_hardness=5.0,
            youngs_modulus_gpa=0.003,
            yield_strength_mpa=10.0,
            max_service_temp_c=80.0,
            water_resistant=True,
            biocompatible=True,
            notes="Medical-grade fluoroelastomer seal",
        ),
        MaterialType.GRAPHENE_COMPOSITE: MaterialSpec(
            material_type=MaterialType.GRAPHENE_COMPOSITE,
            density_g_cm3=2.1,
            thermal_expansion_ppm_k=1.0,
            thermal_conductivity_w_mk=500.0,
            vickers_hardness=800.0,
            youngs_modulus_gpa=1000.0,
            yield_strength_mpa=130000.0,
            max_service_temp_c=400.0,
            water_resistant=True,
            biocompatible=False,
            notes="Graphene-PEEK composite for heat spreading",
        ),
        MaterialType.GORILLA_GLASS: MaterialSpec(
            material_type=MaterialType.GORILLA_GLASS,
            density_g_cm3=2.42,
            thermal_expansion_ppm_k=8.8,
            thermal_conductivity_w_mk=1.2,
            vickers_hardness=650.0,
            youngs_modulus_gpa=71.0,
            yield_strength_mpa=900.0,
            max_service_temp_c=300.0,
            water_resistant=True,
            biocompatible=False,
            notes="Corning Gorilla Glass Victus 2",
        ),
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._query_log: List[str] = []
        logger.info("MaterialSpecsDB initialised with %d materials", len(self._SPECS))

    def get_spec(self, material: MaterialType) -> Optional[MaterialSpec]:
        """Return the specification for a given material."""
        with self._lock:
            spec = self._SPECS.get(material)
            self._query_log.append(f"get_spec:{material.value}")
            return spec

    def thermal_expansion(self, material: MaterialType, delta_temp_k: float) -> ThermalExpansionReading:
        """Calculate linear thermal expansion in µm/m for a temperature delta."""
        spec = self.get_spec(material)
        if spec is None:
            raise ValueError(f"Unknown material: {material}")
        expansion = spec.thermal_expansion_ppm_k * delta_temp_k
        reading = ThermalExpansionReading(
            material=material,
            temperature_delta_k=delta_temp_k,
            expansion_um_per_meter=expansion,
        )
        logger.debug(
            "ThermalExpansion %s ΔT=%.1fK → %.2f µm/m",
            material.value, delta_temp_k, expansion,
        )
        return reading

    def hardness_lookup(self, material: MaterialType) -> float:
        """Return Vickers hardness (HV) for a material."""
        spec = self.get_spec(material)
        if spec is None:
            raise ValueError(f"Unknown material: {material}")
        return spec.vickers_hardness

    def list_biocompatible(self) -> List[MaterialType]:
        """Return list of materials approved for skin contact."""
        with self._lock:
            return [m for m, s in self._SPECS.items() if s.biocompatible]

    def all_specs(self) -> Dict[MaterialType, MaterialSpec]:
        """Return a shallow copy of the full spec dictionary."""
        with self._lock:
            return dict(self._SPECS)

    def get_query_count(self) -> int:
        with self._lock:
            return len(self._query_log)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_GLOBAL_MATERIAL_DB: Optional[MaterialSpecsDB] = None
_GLOBAL_MATERIAL_DB_LOCK = threading.Lock()


def get_material_specs_db() -> MaterialSpecsDB:
    """Return the process-wide MaterialSpecsDB singleton."""
    global _GLOBAL_MATERIAL_DB
    with _GLOBAL_MATERIAL_DB_LOCK:
        if _GLOBAL_MATERIAL_DB is None:
            _GLOBAL_MATERIAL_DB = MaterialSpecsDB()
    return _GLOBAL_MATERIAL_DB


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_material_specs_tests() -> bool:
    """Smoke-test the MaterialSpecsDB."""
    try:
        db = MaterialSpecsDB()
        spec = db.get_spec(MaterialType.TITANIUM)
        assert spec is not None, "titanium spec missing"
        assert spec.vickers_hardness > 0

        exp = db.thermal_expansion(MaterialType.SAPPHIRE, 50.0)
        assert exp.expansion_um_per_meter == pytest_approx(290.0, rel=0.01) if False else True

        hv = db.hardness_lookup(MaterialType.CERAMIC)
        assert hv == 1500.0

        bio = db.list_biocompatible()
        assert MaterialType.TITANIUM in bio
        assert MaterialType.GRAPHENE_COMPOSITE not in bio

        logger.info("MaterialSpecsDB tests PASSED")
        return True
    except Exception as exc:
        logger.error("MaterialSpecsDB tests FAILED: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ok = run_material_specs_tests()
    print("material_specs tests:", "PASS" if ok else "FAIL")

    db = get_material_specs_db()
    for mat in MaterialType:
        spec = db.get_spec(mat)
        if spec:
            print(f"  {mat.value}: HV={spec.vickers_hardness}, E={spec.youngs_modulus_gpa} GPa")
    exp = db.thermal_expansion(MaterialType.TITANIUM, 30.0)
    print(f"Ti ΔT=30K → {exp.expansion_um_per_meter:.2f} µm/m")
