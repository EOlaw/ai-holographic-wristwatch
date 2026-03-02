"""Projector control subpackage for the AI Holographic Wristwatch.

Exports all projector-control components: laser driver, beam focusing,
color calibration, portable optics, projection geometry, and safety protocols.
"""

from src.hardware.holographic.projector_control.laser_controller import (
    LaserColor,
    LaserState,
    LaserPowerReading,
    LaserController,
    get_laser_controller,
)
from src.hardware.holographic.projector_control.beam_focusing import (
    FocusMode,
    BeamState,
    BeamFocusController,
    get_beam_focus_controller,
)
from src.hardware.holographic.projector_control.color_calibration import (
    ColorProfile,
    ColorCalibration,
    ColorCalibrationEngine,
    get_color_calibration_engine,
)
from src.hardware.holographic.projector_control.portable_optics import (
    OpticsState,
    OpticsReading,
    PortableOptics,
    get_portable_optics,
)
from src.hardware.holographic.projector_control.projection_matrix import (
    ProjectionMatrix,
    ProjectionMatrixEngine,
    get_projection_matrix_engine,
)
from src.hardware.holographic.projector_control.safety_protocols import (
    SafetyLevel,
    HazardType,
    SafetyReading,
    SafetyProtocolManager,
    get_safety_protocol_manager,
)

__all__ = [
    "LaserColor", "LaserState", "LaserPowerReading", "LaserController", "get_laser_controller",
    "FocusMode", "BeamState", "BeamFocusController", "get_beam_focus_controller",
    "ColorProfile", "ColorCalibration", "ColorCalibrationEngine", "get_color_calibration_engine",
    "OpticsState", "OpticsReading", "PortableOptics", "get_portable_optics",
    "ProjectionMatrix", "ProjectionMatrixEngine", "get_projection_matrix_engine",
    "SafetyLevel", "HazardType", "SafetyReading", "SafetyProtocolManager",
    "get_safety_protocol_manager",
]
