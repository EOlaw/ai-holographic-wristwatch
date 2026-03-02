"""TouchScreen driver for the AI Holographic Wristwatch OLED + haptic display."""
from __future__ import annotations
import threading
import time
import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from src.core.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TouchState(Enum):
    """Touch panel operational state."""
    IDLE = "idle"
    TOUCHED = "touched"
    GESTURE = "gesture"
    MULTITOUCH = "multitouch"
    LOCKED = "locked"
    CALIBRATING = "calibrating"


class GestureType(Enum):
    """Recognised gesture types."""
    NONE = "none"
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH = "pinch"
    SPREAD = "spread"


@dataclass
class TouchPoint:
    """A single capacitive touch contact."""
    touch_id: int = 0
    x: float = 0.0     # normalised 0–1
    y: float = 0.0
    pressure: float = 0.0   # 0–1
    area_mm2: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TouchEvent:
    """Aggregated touch frame including gesture classification."""
    timestamp: float = field(default_factory=time.time)
    points: List[TouchPoint] = field(default_factory=list)
    gesture: GestureType = GestureType.NONE
    state: TouchState = TouchState.IDLE
    touch_count: int = 0


class TouchController:
    """Capacitive multi-touch driver with gesture recognition at 120 Hz."""

    SAMPLE_RATE_HZ = 120
    MAX_TOUCH_POINTS = 5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state = TouchState.IDLE
        self._event_buffer: List[TouchEvent] = []
        self._callbacks: List = []
        logger.info("TouchController initialised (%d Hz, %d points)",
                    self.SAMPLE_RATE_HZ, self.MAX_TOUCH_POINTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._poll_loop, name="TouchController", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def register_callback(self, callback) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def get_latest_event(self) -> Optional[TouchEvent]:
        with self._lock:
            return self._event_buffer[-1] if self._event_buffer else None

    def lock_screen(self) -> None:
        with self._lock:
            self._state = TouchState.LOCKED
        logger.info("TouchController: screen LOCKED")

    def unlock_screen(self) -> None:
        with self._lock:
            self._state = TouchState.IDLE
        logger.info("TouchController: screen UNLOCKED")

    def calibrate(self) -> bool:
        """Run a touch calibration sequence (simulated)."""
        with self._lock:
            self._state = TouchState.CALIBRATING
        logger.info("TouchController: calibration started")
        time.sleep(0.2)
        with self._lock:
            self._state = TouchState.IDLE
        logger.info("TouchController: calibration complete")
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _simulate_touch_frame(self) -> TouchEvent:
        """Produce a realistic simulated touch frame."""
        with self._lock:
            if self._state == TouchState.LOCKED:
                return TouchEvent(state=TouchState.LOCKED)

        # 3 % chance of touch per frame
        if random.random() > 0.03:
            return TouchEvent(state=TouchState.IDLE)

        count = random.choices([1, 2], weights=[0.85, 0.15])[0]
        points = []
        for i in range(count):
            points.append(TouchPoint(
                touch_id=i,
                x=random.uniform(0.1, 0.9),
                y=random.uniform(0.1, 0.9),
                pressure=random.uniform(0.4, 1.0),
                area_mm2=random.uniform(30.0, 80.0),
            ))
        gesture = random.choice(list(GestureType)) if random.random() < 0.2 else GestureType.NONE
        state = TouchState.MULTITOUCH if count > 1 else TouchState.TOUCHED
        return TouchEvent(points=points, gesture=gesture, state=state, touch_count=count)

    def _poll_loop(self) -> None:
        interval = 1.0 / self.SAMPLE_RATE_HZ
        while True:
            with self._lock:
                if not self._running:
                    break
            event = self._simulate_touch_frame()
            if event.touch_count > 0 or event.gesture != GestureType.NONE:
                with self._lock:
                    self._event_buffer.append(event)
                    if len(self._event_buffer) > 1000:
                        self._event_buffer = self._event_buffer[-1000:]
                    cbs = list(self._callbacks)
                for cb in cbs:
                    try:
                        cb(event)
                    except Exception:
                        pass
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_GLOBAL_TOUCH: Optional[TouchController] = None
_GLOBAL_TOUCH_LOCK = threading.Lock()


def get_touch_controller() -> TouchController:
    global _GLOBAL_TOUCH
    with _GLOBAL_TOUCH_LOCK:
        if _GLOBAL_TOUCH is None:
            _GLOBAL_TOUCH = TouchController()
    return _GLOBAL_TOUCH


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_touch_controller_tests() -> bool:
    try:
        ctrl = TouchController()
        events: List[TouchEvent] = []
        ctrl.register_callback(events.append)
        ctrl.start()
        time.sleep(0.5)
        ctrl.lock_screen()
        time.sleep(0.1)
        assert ctrl._state == TouchState.LOCKED
        ctrl.unlock_screen()
        ok = ctrl.calibrate()
        assert ok
        ctrl.stop()
        logger.info("TouchController tests PASSED")
        return True
    except Exception as exc:
        logger.error("TouchController tests FAILED: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ok = run_touch_controller_tests()
    print("touch_controller tests:", "PASS" if ok else "FAIL")

    tc = get_touch_controller()
    tc.start()
    time.sleep(1.0)
    tc.stop()
    print("Events captured:", len(tc._event_buffer))
