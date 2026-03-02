"""
Holographic Display Interface Contracts for AI Holographic Wristwatch System

Defines the abstract contracts for the holographic projection subsystem and
the OLED watchface display. Concrete hardware drivers, software renderers, and
simulation backends all implement these interfaces, enabling clean dependency
injection and hardware-independent testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ..constants import DisplayConstants
from ..exceptions import HolographicDisplayError, RenderError


# ============================================================================
# Enumerations
# ============================================================================

class DisplayMode(Enum):
    """Operating mode of the holographic projector."""
    OFF = "off"
    STANDBY = "standby"
    ACTIVE = "active"
    CALIBRATING = "calibrating"
    ERROR = "error"


class RenderQuality(Enum):
    """Rendering quality vs. power trade-off setting."""
    ULTRA = "ultra"       # Full holographic quality, highest power
    HIGH = "high"         # Standard holographic quality
    MEDIUM = "medium"     # Reduced depth complexity
    LOW = "low"           # 2D projected on holographic plane
    MINIMAL = "minimal"   # Text/icons only for low-battery


class InteractionType(Enum):
    """Type of user interaction with holographic elements."""
    GAZE = "gaze"
    HAND_GESTURE = "hand_gesture"
    WRIST_GESTURE = "wrist_gesture"
    VOICE = "voice"
    TOUCH = "touch"


class ColorSpace(Enum):
    """Color space for holographic content."""
    SRGB = "srgb"
    DISPLAY_P3 = "display_p3"
    REC2020 = "rec2020"
    LINEAR = "linear"


# ============================================================================
# Data Containers
# ============================================================================

@dataclass
class DisplayStatus:
    """Current operational status of the display subsystem."""
    mode: DisplayMode
    brightness_level: float              # 0.0–1.0
    frame_rate_hz: float
    temperature_c: float
    power_consumption_mw: float
    is_eye_safe: bool
    last_calibration: Optional[datetime]
    error_message: Optional[str] = None


@dataclass
class HologramData:
    """Raw data for a single holographic frame."""
    hologram_id: str
    width_px: int
    height_px: int
    depth_layers: int
    pixel_data: bytes                   # Encoded frame data
    color_space: ColorSpace = ColorSpace.SRGB
    depth_map: Optional[bytes] = None   # Optional depth buffer
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RenderResult:
    """Outcome of a holographic render operation."""
    hologram_id: str
    success: bool
    render_time_ms: float
    frame_number: int
    quality_achieved: RenderQuality
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationResult:
    """Result of a display calibration procedure."""
    success: bool
    calibration_type: str               # "geometric", "color", "eye_tracking"
    correction_applied: Dict[str, Any] = field(default_factory=dict)
    accuracy_metric: float = 0.0        # 0.0–1.0
    notes: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UIElement:
    """A 3D UI widget rendered in holographic space."""
    element_id: str
    element_type: str                   # "button", "panel", "text", "chart"
    position_xyz: Tuple[float, float, float]   # meters from wrist origin
    size_xyz: Tuple[float, float, float]       # meters
    content: Dict[str, Any] = field(default_factory=dict)
    is_interactive: bool = True
    opacity: float = 1.0
    color_rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


@dataclass
class InteractionEvent:
    """User interaction with a holographic element."""
    event_id: str
    interaction_type: InteractionType
    element_id: Optional[str]
    position_xyz: Optional[Tuple[float, float, float]]
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionSpec:
    """Configuration for a holographic projection session."""
    projection_width_mm: float = DisplayConstants.HOLO_PROJECTION_WIDTH_MM
    projection_height_mm: float = DisplayConstants.HOLO_PROJECTION_HEIGHT_MM
    projection_depth_mm: float = DisplayConstants.HOLO_PROJECTION_DEPTH_MM
    safe_distance_mm: float = DisplayConstants.HOLO_SAFE_DISTANCE_MM
    target_fps: int = DisplayConstants.HOLO_FRAME_RATE_HZ
    quality: RenderQuality = RenderQuality.HIGH
    enable_depth: bool = True
    enable_interaction: bool = True


# ============================================================================
# Holographic Display Interface
# ============================================================================

class HolographicDisplayInterface(ABC):
    """
    Abstract contract for the holographic projection subsystem.

    All projection backends — laser holographic hardware, software simulation,
    light-field displays, or test mocks — implement this interface.

    Eye Safety:
        Implementations MUST enforce the eye-safety limits defined in
        DisplayConstants. Any operation that would exceed
        MAX_SAFE_IRRADIANCE_MW_CM2 must raise EyeSafetyViolationError
        immediately and power down the projector.
    """

    @abstractmethod
    async def initialize(self, spec: Optional[ProjectionSpec] = None) -> bool:
        """Initialize the holographic projection hardware.

        Args:
            spec: Optional projection configuration. Uses defaults if None.
        Returns:
            True on success.
        Raises:
            HolographicDisplayError if hardware is absent or broken.
        """
        ...

    @abstractmethod
    async def render_hologram(self, hologram: HologramData) -> RenderResult:
        """Render a holographic frame.

        Args:
            hologram: Complete hologram data for one frame.
        Returns:
            RenderResult with timing and quality metrics.
        Raises:
            RenderError on rendering pipeline failure.
            EyeSafetyViolationError if power limits would be exceeded.
        """
        ...

    @abstractmethod
    async def render_ui_element(self, element: UIElement) -> RenderResult:
        """Render a single 3D UI element into the holographic scene.

        Args:
            element: UIElement specification with position and content.
        Returns:
            RenderResult for this element.
        """
        ...

    @abstractmethod
    async def clear_scene(self) -> None:
        """Remove all rendered elements from the holographic scene."""
        ...

    @abstractmethod
    async def calibrate(self, calibration_type: str = "full") -> CalibrationResult:
        """Run display calibration.

        Args:
            calibration_type: One of "geometric", "color", "eye_tracking", "full".
        Returns:
            CalibrationResult with outcome.
        """
        ...

    @abstractmethod
    def set_brightness(self, level: float) -> None:
        """Set display brightness.

        Args:
            level: 0.0 (off) to 1.0 (maximum safe brightness).
        Raises:
            ValueError: If level is outside [0.0, 1.0].
        """
        ...

    @abstractmethod
    def get_brightness(self) -> float:
        """Return current brightness level (0.0–1.0)."""
        ...

    @abstractmethod
    def set_quality(self, quality: RenderQuality) -> None:
        """Set rendering quality mode."""
        ...

    @abstractmethod
    def get_display_status(self) -> DisplayStatus:
        """Return current operational status."""
        ...

    @abstractmethod
    async def power_on(self) -> bool:
        """Power on the holographic projector. Returns True on success."""
        ...

    @abstractmethod
    async def power_off(self) -> None:
        """Power off the holographic projector safely."""
        ...

    @abstractmethod
    async def emergency_shutdown(self) -> None:
        """Immediately cut power to all lasers. Used in safety interlocks."""
        ...

    @abstractmethod
    def is_eye_safe(self) -> bool:
        """Return True if current operating parameters are within eye-safety limits."""
        ...

    @abstractmethod
    async def stream_frames(self, frame_source: AsyncIterator[HologramData]
                            ) -> None:
        """Stream hologram frames from an async source at the target FPS.

        Args:
            frame_source: Async iterator providing HologramData at render rate.
        """
        ...


# ============================================================================
# Interaction Tracking Interface
# ============================================================================

class InteractionTrackingInterface(ABC):
    """Contract for holographic interaction detection (gaze, hand, gesture)."""

    @abstractmethod
    async def start_tracking(self) -> None:
        """Start the interaction tracking pipeline."""
        ...

    @abstractmethod
    async def stop_tracking(self) -> None:
        """Stop interaction tracking and release resources."""
        ...

    @abstractmethod
    async def get_interactions(self) -> AsyncIterator[InteractionEvent]:
        """Yield detected interaction events in real time.

        Yields:
            InteractionEvent as users interact with holographic elements.
        """
        ...

    @abstractmethod
    def register_element(self, element: UIElement) -> None:
        """Register a UI element for hit-testing.

        Args:
            element: UIElement with spatial bounds and ID.
        """
        ...

    @abstractmethod
    def unregister_element(self, element_id: str) -> None:
        """Remove a UI element from hit-testing."""
        ...

    @abstractmethod
    def get_gaze_point(self) -> Optional[Tuple[float, float, float]]:
        """Return estimated gaze intersection with holographic plane (m), or None."""
        ...

    @abstractmethod
    def get_hand_position(self) -> Optional[Tuple[float, float, float]]:
        """Return dominant hand fingertip position (m), or None if not detected."""
        ...


# ============================================================================
# Display Calibration Interface
# ============================================================================

class DisplayCalibrationInterface(ABC):
    """Contract for holographic display calibration procedures."""

    @abstractmethod
    async def run_geometric_calibration(self) -> CalibrationResult:
        """Correct lens distortion and geometric projection errors."""
        ...

    @abstractmethod
    async def run_color_calibration(self) -> CalibrationResult:
        """Calibrate per-channel laser intensities for color accuracy."""
        ...

    @abstractmethod
    async def run_eye_tracking_calibration(self, gaze_targets: int = 9
                                           ) -> CalibrationResult:
        """Calibrate the eye-tracking system for the current user.

        Args:
            gaze_targets: Number of gaze calibration points (default 9).
        """
        ...

    @abstractmethod
    def get_calibration_status(self) -> Dict[str, Any]:
        """Return calibration status and age for all calibration types."""
        ...

    @abstractmethod
    def needs_recalibration(self) -> bool:
        """Return True if any calibration is expired or missing."""
        ...


# ============================================================================
# OLED Watch Face Interface
# ============================================================================

class WatchFaceDisplayInterface(ABC):
    """Contract for the OLED watchface display (non-holographic)."""

    @abstractmethod
    async def render_watchface(self, watchface_data: Dict[str, Any]) -> bool:
        """Render the current watchface frame.

        Args:
            watchface_data: Dict with time, metrics, notifications, etc.
        Returns:
            True if rendering succeeded.
        """
        ...

    @abstractmethod
    def set_always_on_display(self, enabled: bool) -> None:
        """Enable or disable always-on display mode."""
        ...

    @abstractmethod
    def set_brightness(self, nits: int) -> None:
        """Set display brightness in nits.

        Args:
            nits: Target brightness within
                  [OLED_MIN_BRIGHTNESS_NITS, OLED_MAX_BRIGHTNESS_NITS].
        """
        ...

    @abstractmethod
    def get_display_info(self) -> Dict[str, Any]:
        """Return display hardware specifications (resolution, PPI, etc.)."""
        ...

    @abstractmethod
    async def show_notification(self, title: str, body: str,
                                duration_seconds: float = 5.0) -> None:
        """Display a transient notification on the watchface."""
        ...


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "DisplayMode", "RenderQuality", "InteractionType", "ColorSpace",
    "DisplayStatus", "HologramData", "RenderResult", "CalibrationResult",
    "UIElement", "InteractionEvent", "ProjectionSpec",
    "HolographicDisplayInterface", "InteractionTrackingInterface",
    "DisplayCalibrationInterface", "WatchFaceDisplayInterface",
]
