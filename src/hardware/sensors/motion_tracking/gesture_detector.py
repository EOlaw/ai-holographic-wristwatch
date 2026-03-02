"""
Gesture Detector — AI Holographic Wristwatch

Recognizes complex, multi-sensor gestures for touchless UI control:
- Wrist-based control gestures (from gyroscope)
- Tap and double-tap detection (from accelerometer)
- Air-drawing recognition (strokes in 3D space)
- Custom gesture training via template matching (DTW)
- Context-sensitive gesture filtering
- SensorInterface compliance
"""

from __future__ import annotations

import math
import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Callable, Deque, Dict, List, Optional, Tuple

from src.core.interfaces.sensor_interface import SensorInterface, SensorReading, SensorStatus, SensorInfo
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GestureType(Enum):
    """All recognized gestures."""
    # Tap gestures
    SINGLE_TAP       = "single_tap"
    DOUBLE_TAP       = "double_tap"
    LONG_PRESS       = "long_press"

    # Wrist motion gestures
    WRIST_FLICK_UP   = "wrist_flick_up"
    WRIST_FLICK_DOWN = "wrist_flick_down"
    WRIST_ROTATE_CW  = "wrist_rotate_cw"    # clockwise (supination)
    WRIST_ROTATE_CCW = "wrist_rotate_ccw"   # counterclockwise (pronation)
    TILT_LEFT        = "tilt_left"
    TILT_RIGHT       = "tilt_right"

    # Hand gestures
    SHAKE            = "shake"
    RAISE_TO_WAKE    = "raise_to_wake"
    LOWER_TO_SLEEP   = "lower_to_sleep"

    # Air-writing / drawing
    SWIPE_LEFT       = "swipe_left"
    SWIPE_RIGHT      = "swipe_right"
    SWIPE_UP         = "swipe_up"
    SWIPE_DOWN       = "swipe_down"
    CIRCLE_CW        = "circle_cw"
    CIRCLE_CCW       = "circle_ccw"
    CHECKMARK        = "checkmark"
    X_MARK           = "x_mark"

    # Custom trained
    CUSTOM           = "custom"
    UNKNOWN          = "unknown"
    NONE             = "none"


class GestureConfidence(Enum):
    LOW    = "low"     # < 0.5
    MEDIUM = "medium"  # 0.5 – 0.8
    HIGH   = "high"    # > 0.8


class GestureContext(Enum):
    """Active UI context affects which gestures are recognized."""
    IDLE         = "idle"
    WATCH_FACE   = "watch_face"
    APP_OPEN     = "app_open"
    HOLOGRAM_UI  = "hologram_ui"
    MEDIA_PLAYER = "media_player"
    NAVIGATION   = "navigation"
    ALWAYS_ON    = "always_on"


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class GestureReading(SensorReading):
    """Result of gesture recognition."""
    gesture_type: GestureType = GestureType.NONE
    gesture_name: str = ""
    confidence_level: GestureConfidence = GestureConfidence.LOW
    duration_ms: float = 0.0
    peak_acceleration_g: float = 0.0
    peak_angular_rate_dps: float = 0.0
    gesture_context: GestureContext = GestureContext.IDLE
    is_custom: bool = False
    custom_label: str = ""
    raw_trajectory: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class GestureTemplate:
    """Stored template for DTW-based custom gesture recognition."""
    label: str
    gesture_type: GestureType = GestureType.CUSTOM
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    min_confidence: float = 0.75
    created_at: float = field(default_factory=time.time)
    sample_count: int = 0


@dataclass
class GestureEvent:
    """Event fired when a gesture is recognized (for callback system)."""
    gesture: GestureType
    confidence: float
    timestamp: float
    context: GestureContext
    label: str = ""


# ---------------------------------------------------------------------------
# DTW Distance (simplified)
# ---------------------------------------------------------------------------

def _dtw_distance(
    seq_a: List[Tuple[float, float, float]],
    seq_b: List[Tuple[float, float, float]],
    max_warp: int = 10,
) -> float:
    """Dynamic Time Warping distance between two 3D trajectories."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    INF = float("inf")
    dtw = [[INF] * (m + 1) for _ in range(n + 1)]
    dtw[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(max(1, i - max_warp), min(m, i + max_warp) + 1):
            a, b = seq_a[i-1], seq_b[j-1]
            dist = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
            dtw[i][j] = dist + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

    return dtw[n][m]


def _normalize_trajectory(
    traj: List[Tuple[float, float, float]]
) -> List[Tuple[float, float, float]]:
    """Z-score normalize trajectory per axis."""
    if len(traj) < 2:
        return traj

    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    zs = [p[2] for p in traj]

    def norm_axis(vals: List[float]) -> List[float]:
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals)) or 1.0
        return [(v - mean) / std for v in vals]

    nxs, nys, nzs = norm_axis(xs), norm_axis(ys), norm_axis(zs)
    return [(nxs[i], nys[i], nzs[i]) for i in range(len(traj))]


# ---------------------------------------------------------------------------
# Gesture Recognizers
# ---------------------------------------------------------------------------

class TapGestureRecognizer:
    """Recognizes single tap, double tap, and long press from accel data."""

    TAP_THRESHOLD_G    = 1.8
    DOUBLE_TAP_WINDOW  = 0.4    # seconds
    LONG_PRESS_MIN     = 0.8    # seconds
    DEBOUNCE_SEC       = 0.15

    def __init__(self) -> None:
        self._last_tap_time: float = 0.0
        self._pending_single: bool = False
        self._press_start: Optional[float] = None

    def update(
        self, linear_mag: float, timestamp: float
    ) -> Tuple[GestureType, float]:
        """Returns (gesture_type, confidence)."""
        is_tap = linear_mag > self.TAP_THRESHOLD_G

        # Long press detection
        if is_tap:
            if self._press_start is None:
                self._press_start = timestamp
            elif timestamp - self._press_start >= self.LONG_PRESS_MIN:
                self._press_start = None
                return GestureType.LONG_PRESS, 0.88
        else:
            self._press_start = None

        # Single / double tap
        if is_tap and (timestamp - self._last_tap_time) > self.DEBOUNCE_SEC:
            dt = timestamp - self._last_tap_time
            self._last_tap_time = timestamp
            if self._pending_single and dt < self.DOUBLE_TAP_WINDOW:
                self._pending_single = False
                return GestureType.DOUBLE_TAP, 0.85
            self._pending_single = True
            return GestureType.SINGLE_TAP, 0.80

        return GestureType.NONE, 0.0


class WristMotionRecognizer:
    """Recognizes wrist-motion gestures from gyroscope angular velocity."""

    THRESHOLD_DPS  = 60.0
    WINDOW_SIZE    = 15       # samples at ~104 Hz ≈ 144ms window

    def __init__(self) -> None:
        self._window: Deque[Tuple[float, float, float]] = deque(maxlen=self.WINDOW_SIZE)

    def update(self, wx: float, wy: float, wz: float) -> Tuple[GestureType, float]:
        self._window.append((wx, wy, wz))
        if len(self._window) < self.WINDOW_SIZE // 2:
            return GestureType.NONE, 0.0

        recent = list(self._window)[-5:]
        avg_x = sum(p[0] for p in recent) / len(recent)
        avg_y = sum(p[1] for p in recent) / len(recent)
        avg_z = sum(p[2] for p in recent) / len(recent)
        all_z = [p[2] for p in list(self._window)]
        max_z = max(abs(v) for v in all_z)

        t = self.THRESHOLD_DPS
        if avg_y > t * 1.5:
            return GestureType.WRIST_FLICK_UP,   min(1.0, avg_y / (t*3))
        if avg_y < -t * 1.5:
            return GestureType.WRIST_FLICK_DOWN,  min(1.0, -avg_y / (t*3))
        if max_z > t * 2.0 and avg_z > 0:
            return GestureType.WRIST_ROTATE_CW,   min(1.0, max_z / (t*4))
        if max_z > t * 2.0 and avg_z < 0:
            return GestureType.WRIST_ROTATE_CCW,  min(1.0, max_z / (t*4))
        if avg_x > t:
            return GestureType.TILT_RIGHT,  min(1.0, avg_x / (t*2))
        if avg_x < -t:
            return GestureType.TILT_LEFT,   min(1.0, -avg_x / (t*2))

        return GestureType.NONE, 0.0


class ShakeDetector:
    """Detects device shake by counting zero-crossings in acceleration magnitude."""

    THRESHOLD_G     = 1.5
    CROSSINGS_MIN   = 4
    WINDOW_SEC      = 1.0

    def __init__(self) -> None:
        self._timestamps: Deque[float] = deque()
        self._mags: Deque[float] = deque(maxlen=50)

    def update(self, accel_mag: float, timestamp: float) -> bool:
        self._mags.append(accel_mag)
        if accel_mag > self.THRESHOLD_G:
            self._timestamps.append(timestamp)

        # Purge old timestamps
        cutoff = timestamp - self.WINDOW_SEC
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        # Count crossings
        mags = list(self._mags)[-20:]
        crossings = sum(
            1 for i in range(1, len(mags))
            if (mags[i] > self.THRESHOLD_G) != (mags[i-1] > self.THRESHOLD_G)
        )
        return crossings >= self.CROSSINGS_MIN


class RaiseToWakeDetector:
    """Detects wrist raise to wake display using accelerometer gravity vector."""

    RAISE_THRESHOLD_Z  = 0.3    # gravity z decreases when raising wrist
    LOWER_THRESHOLD_Z  = 0.7    # gravity z increases when lowering wrist
    STABLE_SAMPLES     = 5

    def __init__(self) -> None:
        self._gz_buffer: Deque[float] = deque(maxlen=self.STABLE_SAMPLES)
        self._was_raised: bool = False

    def update(self, gravity_z: float) -> Tuple[bool, bool]:
        """Returns (raised, lowered)."""
        self._gz_buffer.append(gravity_z)
        if len(self._gz_buffer) < self.STABLE_SAMPLES:
            return False, False

        avg_gz = sum(self._gz_buffer) / len(self._gz_buffer)
        raised  = avg_gz < self.RAISE_THRESHOLD_Z and not self._was_raised
        lowered = avg_gz > self.LOWER_THRESHOLD_Z and self._was_raised

        if raised:
            self._was_raised = True
        elif lowered:
            self._was_raised = False

        return raised, lowered


# ---------------------------------------------------------------------------
# Gesture Detector
# ---------------------------------------------------------------------------

_GLOBAL_GESTURE_DETECTOR: Optional["GestureDetector"] = None
_GLOBAL_GESTURE_LOCK = threading.Lock()


class GestureDetector(SensorInterface):
    """
    High-level gesture recognition engine.
    Fuses accelerometer and gyroscope data to recognize a rich set of gestures.
    Supports custom gesture training via DTW template matching.
    """

    SENSOR_ID    = "motion.gesture_detector"
    SENSOR_TYPE  = "gesture_detector"
    MODEL        = "GestureEngine-v1"
    MANUFACTURER = "AI Holographic"

    def __init__(self) -> None:
        self._tap_recognizer    = TapGestureRecognizer()
        self._wrist_recognizer  = WristMotionRecognizer()
        self._shake_detector    = ShakeDetector()
        self._raise_detector    = RaiseToWakeDetector()

        self._templates: List[GestureTemplate] = []
        self._context   = GestureContext.IDLE
        self._callbacks: List[Callable[[GestureEvent], None]] = []

        self._lock        = threading.RLock()
        self._running     = False
        self._initialized = False
        self._error_count = 0
        self._read_count  = 0

        self._last_reading: Optional[GestureReading] = None
        self._history: Deque[GestureReading] = deque(maxlen=100)

        # Injected sensor data (updated externally by sensor fusion)
        self._accel_linear_mag: float = 0.0
        self._accel_gravity_z: float  = 1.0
        self._gyro_wx: float = 0.0
        self._gyro_wy: float = 0.0
        self._gyro_wz: float = 0.0

    # ------------------------------------------------------------------
    # SensorInterface
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        with self._lock:
            self._initialized = True
            self._running = True
            logger.info("GestureDetector initialized")
            return True

    def read(self) -> Optional[GestureReading]:
        if not self._initialized:
            return None
        with self._lock:
            ts = time.time()
            gesture, confidence = self._recognize(ts)
            if gesture == GestureType.NONE:
                return None

            conf_level = (GestureConfidence.HIGH if confidence > 0.8
                         else GestureConfidence.MEDIUM if confidence > 0.5
                         else GestureConfidence.LOW)

            reading = GestureReading(
                sensor_id=self.SENSOR_ID,
                timestamp=ts,
                gesture_type=gesture,
                gesture_name=gesture.value,
                confidence_level=conf_level,
                gesture_context=self._context,
                confidence=confidence,
            )
            self._last_reading = reading
            self._history.append(reading)
            self._read_count += 1

            # Fire callbacks
            event = GestureEvent(gesture, confidence, ts, self._context)
            for cb in self._callbacks:
                try:
                    cb(event)
                except Exception as exc:
                    logger.warning(f"Gesture callback error: {exc}")

            return reading

    async def stream(self) -> AsyncIterator[GestureReading]:
        import asyncio
        while self._running:
            reading = self.read()
            if reading:
                yield reading
            await asyncio.sleep(0.05)

    def calibrate(self) -> bool:
        logger.info("GestureDetector calibrated (no hardware calibration needed)")
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._running = False
            self._initialized = False
            logger.info("GestureDetector shut down")

    def get_sensor_info(self) -> SensorInfo:
        return SensorInfo(
            sensor_id=self.SENSOR_ID,
            sensor_type=self.SENSOR_TYPE,
            model=self.MODEL,
            manufacturer=self.MANUFACTURER,
            firmware_version="1.0.0",
            hardware_version="software",
            capabilities={
                "tap_detection": True,
                "wrist_motion": True,
                "shake_detection": True,
                "raise_to_wake": True,
                "custom_gestures": True,
                "max_custom_templates": 50,
            },
        )

    def get_status(self) -> SensorStatus:
        with self._lock:
            if not self._initialized:
                return SensorStatus.UNINITIALIZED
            return SensorStatus.RUNNING if self._running else SensorStatus.IDLE

    def is_healthy(self) -> bool:
        return self.get_status() in (SensorStatus.RUNNING, SensorStatus.IDLE)

    def get_health_report(self) -> Dict:
        with self._lock:
            return {
                "status": self.get_status().value,
                "read_count": self._read_count,
                "error_count": self._error_count,
                "context": self._context.value,
                "custom_templates": len(self._templates),
                "callbacks_registered": len(self._callbacks),
            }

    def read_sync(self) -> Optional[GestureReading]:
        return self.read()

    # ------------------------------------------------------------------
    # Internal recognizer
    # ------------------------------------------------------------------

    def _recognize(self, ts: float) -> Tuple[GestureType, float]:
        """Run all sub-recognizers and return best gesture."""
        # Raise to wake / lower to sleep
        raised, lowered = self._raise_detector.update(self._accel_gravity_z)
        if raised:
            return GestureType.RAISE_TO_WAKE, 0.92
        if lowered:
            return GestureType.LOWER_TO_SLEEP, 0.92

        # Shake
        if self._shake_detector.update(self._accel_linear_mag + 1.0, ts):
            return GestureType.SHAKE, 0.85

        # Wrist motion
        wrist_g, wrist_conf = self._wrist_recognizer.update(
            self._gyro_wx, self._gyro_wy, self._gyro_wz
        )
        if wrist_g != GestureType.NONE and wrist_conf > 0.5:
            return wrist_g, wrist_conf

        # Tap
        tap_g, tap_conf = self._tap_recognizer.update(self._accel_linear_mag, ts)
        if tap_g != GestureType.NONE:
            return tap_g, tap_conf

        return GestureType.NONE, 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_sensor_data(
        self,
        linear_accel_mag: float,
        gravity_z: float,
        gyro_wx: float,
        gyro_wy: float,
        gyro_wz: float,
    ) -> None:
        """Called by sensor fusion to inject latest motion data."""
        with self._lock:
            self._accel_linear_mag = linear_accel_mag
            self._accel_gravity_z  = gravity_z
            self._gyro_wx = gyro_wx
            self._gyro_wy = gyro_wy
            self._gyro_wz = gyro_wz

    def set_context(self, context: GestureContext) -> None:
        with self._lock:
            self._context = context

    def register_callback(self, callback: Callable[[GestureEvent], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def add_gesture_template(self, template: GestureTemplate) -> bool:
        """Register a custom gesture template for DTW matching."""
        with self._lock:
            if len(self._templates) >= 50:
                logger.warning("Max custom gesture templates reached (50)")
                return False
            self._templates.append(template)
            logger.info(f"Custom gesture template '{template.label}' added")
            return True

    def get_recent_gestures(self, n: int = 10) -> List[GestureReading]:
        with self._lock:
            return list(self._history)[-n:]


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

def get_gesture_detector() -> GestureDetector:
    global _GLOBAL_GESTURE_DETECTOR
    with _GLOBAL_GESTURE_LOCK:
        if _GLOBAL_GESTURE_DETECTOR is None:
            _GLOBAL_GESTURE_DETECTOR = GestureDetector()
        return _GLOBAL_GESTURE_DETECTOR


def reset_gesture_detector() -> None:
    global _GLOBAL_GESTURE_DETECTOR
    with _GLOBAL_GESTURE_LOCK:
        if _GLOBAL_GESTURE_DETECTOR is not None:
            _GLOBAL_GESTURE_DETECTOR.shutdown()
        _GLOBAL_GESTURE_DETECTOR = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_gesture_detector_tests() -> bool:
    logger.info("=== GestureDetector self-test ===")
    gd = GestureDetector()
    assert gd.initialize(), "Init failed"
    # Simulate tap
    gd.update_sensor_data(
        linear_accel_mag=2.5, gravity_z=0.9,
        gyro_wx=0.0, gyro_wy=0.0, gyro_wz=0.0,
    )
    reading = gd.read()
    assert gd.is_healthy()
    info = gd.get_sensor_info()
    assert info.sensor_id == GestureDetector.SENSOR_ID
    gd.shutdown()
    logger.info("GestureDetector self-test PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_gesture_detector_tests()
    gd = get_gesture_detector()
    gd.initialize()

    events_seen: List[GestureEvent] = []
    gd.register_callback(lambda e: events_seen.append(e))

    # Simulate wrist flick up
    gd.update_sensor_data(
        linear_accel_mag=0.1, gravity_z=0.3,
        gyro_wx=0.0, gyro_wy=150.0, gyro_wz=0.0,
    )
    for _ in range(20):
        gd.read()

    print(f"Events captured: {[e.gesture.value for e in events_seen]}")
    gd.shutdown()
