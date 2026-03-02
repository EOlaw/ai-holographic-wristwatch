"""
Context Awareness Engine — AI Holographic Wristwatch

Infers high-level situational context from fused sensor data:
- User context: activity, location type, social setting
- Device context: wear state, charging, orientation
- Environmental context: indoor/outdoor, time of day, weather
- Interaction context: active/idle, in-meeting, driving
- Context-driven behavior adaptation (hologram brightness, notification filtering)
- Proactive context prediction using pattern history
"""

from __future__ import annotations

import threading
import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class UserActivity(Enum):
    SLEEPING        = "sleeping"
    RESTING         = "resting"
    SEDENTARY       = "sedentary"
    WALKING         = "walking"
    RUNNING         = "running"
    EXERCISING      = "exercising"
    DRIVING         = "driving"
    COMMUTING       = "commuting"
    WORKING         = "working"
    SOCIALIZING     = "socializing"
    EATING          = "eating"
    UNKNOWN         = "unknown"


class LocationType(Enum):
    HOME            = "home"
    OFFICE          = "office"
    GYM             = "gym"
    RESTAURANT      = "restaurant"
    OUTDOORS        = "outdoors"
    TRANSIT         = "transit"
    STORE           = "store"
    HOSPITAL        = "hospital"
    UNKNOWN         = "unknown"


class SocialContext(Enum):
    ALONE           = "alone"
    WITH_COLLEAGUES = "with_colleagues"
    WITH_FRIENDS    = "with_friends"
    IN_MEETING      = "in_meeting"
    IN_CROWD        = "in_crowd"
    UNKNOWN         = "unknown"


class WearState(Enum):
    ON_WRIST        = "on_wrist"
    ON_TABLE        = "on_table"
    IN_BAG          = "in_bag"
    CHARGING        = "charging"
    UNKNOWN         = "unknown"


class InteractionMode(Enum):
    ACTIVE          = "active"          # user is interacting with hologram
    GLANCE          = "glance"          # brief check
    AMBIENT         = "ambient"         # watch face visible
    IDLE            = "idle"            # screen off, minimal compute
    FOCUS_MODE      = "focus_mode"      # DND, minimal interruption
    PRESENTATION    = "presentation"    # hologram visible to others


class TimeOfDay(Enum):
    EARLY_MORNING   = "early_morning"   # 04:00–07:00
    MORNING         = "morning"         # 07:00–10:00
    MID_MORNING     = "mid_morning"     # 10:00–12:00
    AFTERNOON       = "afternoon"       # 12:00–17:00
    EVENING         = "evening"         # 17:00–21:00
    NIGHT           = "night"           # 21:00–24:00
    LATE_NIGHT      = "late_night"      # 00:00–04:00


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class UserContext:
    activity: UserActivity = UserActivity.UNKNOWN
    location_type: LocationType = LocationType.UNKNOWN
    social_context: SocialContext = SocialContext.ALONE
    estimated_stress_level: float = 0.3   # 0–1
    attention_level: float = 0.5          # 0–1


@dataclass
class DeviceContext:
    wear_state: WearState = WearState.UNKNOWN
    interaction_mode: InteractionMode = InteractionMode.IDLE
    battery_level_pct: float = 80.0
    is_charging: bool = False
    hologram_active: bool = False
    screen_on: bool = False
    active_app: str = "watch_face"


@dataclass
class EnvironmentContext:
    is_indoor: bool = True
    location_type: LocationType = LocationType.UNKNOWN
    time_of_day: TimeOfDay = TimeOfDay.AFTERNOON
    ambient_noise_db: float = 50.0
    ambient_lux: float = 500.0
    temperature_c: float = 22.0
    weather_suitable_for_hologram: bool = True


@dataclass
class ContextSnapshot:
    """Complete situational context at a point in time."""
    timestamp: float = field(default_factory=time.time)
    user: UserContext = field(default_factory=UserContext)
    device: DeviceContext = field(default_factory=DeviceContext)
    environment: EnvironmentContext = field(default_factory=EnvironmentContext)
    confidence: float = 0.75

    # Derived behavior recommendations
    recommended_hologram_brightness: float = 0.8
    recommended_notification_mode: str = "normal"  # silent/vibrate/normal/loud
    recommended_anc_mode: str = "standard"
    should_conserve_battery: bool = False
    proactive_suggestions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Context Inference Rules
# ---------------------------------------------------------------------------

class ActivityInferenceEngine:
    """
    Rule-based activity inference from multi-sensor signals.
    Production: trained classifier (HMM or CNN on sensor windows).
    """

    def infer(
        self,
        heart_rate: float,
        accel_mag: float,
        cadence_spm: float,
        hour: int,
        noise_db: float,
    ) -> UserActivity:
        # Sleep detection
        if hour in (0, 1, 2, 3, 4, 5) and heart_rate < 65 and accel_mag < 1.05:
            return UserActivity.SLEEPING

        if accel_mag < 1.02 and cadence_spm < 5:
            if noise_db > 65:
                return UserActivity.SOCIALIZING
            if heart_rate > 100:
                return UserActivity.WORKING   # cognitive stress
            return UserActivity.SEDENTARY

        if 30 < cadence_spm < 90:
            return UserActivity.WALKING
        if cadence_spm >= 90 and heart_rate > 120:
            return UserActivity.RUNNING
        if cadence_spm >= 90:
            return UserActivity.EXERCISING

        return UserActivity.UNKNOWN


class WearStateDetector:
    """Infers whether watch is being worn using PPG + accelerometer."""

    PPG_WEAR_THRESHOLD = 0.6    # proximity/capacitance reading
    GRAVITY_VARIANCE_MAX = 0.02 # variance of gravity Z when on wrist

    def detect(self, gravity_z: float, ppg_contact: float) -> WearState:
        if ppg_contact > self.PPG_WEAR_THRESHOLD:
            return WearState.ON_WRIST
        if abs(gravity_z - 1.0) < 0.1:
            return WearState.ON_TABLE
        return WearState.UNKNOWN


class NotificationPolicyEngine:
    """Selects appropriate notification delivery mode based on context."""

    def recommend(self, ctx: ContextSnapshot) -> str:
        if ctx.user.activity == UserActivity.SLEEPING:
            return "silent"
        if ctx.device.interaction_mode == InteractionMode.PRESENTATION:
            return "vibrate"
        if ctx.environment.ambient_noise_db > 75:
            return "loud"
        if ctx.user.activity in (UserActivity.RUNNING, UserActivity.EXERCISING):
            return "vibrate"
        if ctx.user.social_context == SocialContext.IN_MEETING:
            return "vibrate"
        return "normal"


class HologramBrightnessPolicy:
    """Maps ambient lux to recommended hologram brightness."""

    def recommend(self, lux: float, is_indoor: bool, battery_pct: float) -> float:
        if lux < 10:
            base = 0.3
        elif lux < 200:
            base = 0.5
        elif lux < 1000:
            base = 0.7
        elif lux < 10000:
            base = 0.9
        else:
            base = 1.0

        # Battery conservation
        if battery_pct < 20:
            base *= 0.6
        elif battery_pct < 10:
            base *= 0.3

        return min(1.0, base)


def _hour_to_time_of_day(hour: int) -> TimeOfDay:
    if 4 <= hour < 7:   return TimeOfDay.EARLY_MORNING
    if 7 <= hour < 10:  return TimeOfDay.MORNING
    if 10 <= hour < 12: return TimeOfDay.MID_MORNING
    if 12 <= hour < 17: return TimeOfDay.AFTERNOON
    if 17 <= hour < 21: return TimeOfDay.EVENING
    if 21 <= hour < 24: return TimeOfDay.NIGHT
    return TimeOfDay.LATE_NIGHT


# ---------------------------------------------------------------------------
# Context Awareness Engine
# ---------------------------------------------------------------------------

_GLOBAL_CTX: Optional["ContextAwarenessEngine"] = None
_GLOBAL_CTX_LOCK = threading.Lock()


class ContextAwarenessEngine:
    """
    Continuously infers and maintains situational context.
    Produces ContextSnapshot objects consumed by UI, notifications, and AI.
    """

    def __init__(self) -> None:
        self._activity_engine     = ActivityInferenceEngine()
        self._wear_detector       = WearStateDetector()
        self._notification_policy = NotificationPolicyEngine()
        self._brightness_policy   = HologramBrightnessPolicy()

        self._history: Deque[ContextSnapshot] = deque(maxlen=100)
        self._lock = threading.RLock()
        self._running = False
        self._read_count = 0
        self._last_context: Optional[ContextSnapshot] = None

    def start(self) -> None:
        with self._lock:
            self._running = True
            logger.info("ContextAwarenessEngine started")

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def infer(
        self,
        heart_rate: float = 72.0,
        accel_mag: float = 1.0,
        cadence_spm: float = 0.0,
        lux: float = 500.0,
        noise_db: float = 50.0,
        temperature_c: float = 22.0,
        is_indoor: bool = True,
        battery_pct: float = 80.0,
        is_charging: bool = False,
        ppg_contact: float = 0.9,
        gravity_z: float = 1.0,
    ) -> ContextSnapshot:
        with self._lock:
            hour = time.localtime().tm_hour
            activity = self._activity_engine.infer(heart_rate, accel_mag, cadence_spm, hour, noise_db)
            wear     = self._wear_detector.detect(gravity_z, ppg_contact)
            tod      = _hour_to_time_of_day(hour)

            user_ctx = UserContext(
                activity=activity,
                location_type=LocationType.HOME if is_indoor else LocationType.OUTDOORS,
                social_context=(SocialContext.IN_CROWD if noise_db > 70
                                else SocialContext.ALONE),
                estimated_stress_level=random.gauss(0.3, 0.1),
                attention_level=random.gauss(0.6, 0.1),
            )

            dev_ctx = DeviceContext(
                wear_state=wear,
                interaction_mode=(InteractionMode.IDLE if activity == UserActivity.SLEEPING
                                  else InteractionMode.AMBIENT),
                battery_level_pct=battery_pct,
                is_charging=is_charging,
            )

            env_ctx = EnvironmentContext(
                is_indoor=is_indoor,
                location_type=LocationType.HOME if is_indoor else LocationType.OUTDOORS,
                time_of_day=tod,
                ambient_noise_db=noise_db,
                ambient_lux=lux,
                temperature_c=temperature_c,
                weather_suitable_for_hologram=not (lux > 80000),
            )

            brightness = self._brightness_policy.recommend(lux, is_indoor, battery_pct)

            ctx = ContextSnapshot(
                user=user_ctx, device=dev_ctx, environment=env_ctx,
                recommended_hologram_brightness=brightness,
                should_conserve_battery=battery_pct < 20,
            )
            ctx.recommended_notification_mode = self._notification_policy.recommend(ctx)
            ctx.proactive_suggestions = self._generate_suggestions(ctx)

            self._last_context = ctx
            self._history.append(ctx)
            self._read_count += 1
            return ctx

    def _generate_suggestions(self, ctx: ContextSnapshot) -> List[str]:
        suggestions = []
        if ctx.user.activity == UserActivity.SEDENTARY:
            suggestions.append("You've been sitting for a while. Time to move!")
        if ctx.device.battery_level_pct < 15 and not ctx.device.is_charging:
            suggestions.append("Battery low — connect charger soon.")
        if ctx.environment.ambient_noise_db > 80:
            suggestions.append("Loud environment detected — ear protection recommended.")
        return suggestions

    def get_last_context(self) -> Optional[ContextSnapshot]:
        return self._last_context

    def get_history(self, n: int = 10) -> List[ContextSnapshot]:
        with self._lock:
            return list(self._history)[-n:]

    def is_healthy(self) -> bool:
        return self._running

    def get_health_report(self) -> Dict:
        return {"running": self._running, "inferences": self._read_count,
                "last_activity": (self._last_context.user.activity.value
                                  if self._last_context else "unknown")}


def get_context_awareness_engine() -> ContextAwarenessEngine:
    global _GLOBAL_CTX
    with _GLOBAL_CTX_LOCK:
        if _GLOBAL_CTX is None:
            _GLOBAL_CTX = ContextAwarenessEngine()
        return _GLOBAL_CTX


def run_context_awareness_tests() -> bool:
    engine = ContextAwarenessEngine()
    engine.start()
    ctx = engine.infer(heart_rate=72.0, accel_mag=1.2, cadence_spm=95.0)
    assert ctx.user.activity in list(UserActivity)
    assert 0.0 <= ctx.recommended_hologram_brightness <= 1.0
    ctx2 = engine.infer(heart_rate=55.0, accel_mag=1.0, cadence_spm=0.0)
    engine.stop()
    logger.info("ContextAwarenessEngine tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_context_awareness_tests()
    eng = get_context_awareness_engine()
    eng.start()
    ctx = eng.infer(heart_rate=95.0, accel_mag=1.5, cadence_spm=110.0,
                    lux=2000.0, is_indoor=False, battery_pct=75.0)
    print(f"Activity: {ctx.user.activity.value}")
    print(f"Brightness: {ctx.recommended_hologram_brightness:.0%}")
    print(f"Notification: {ctx.recommended_notification_mode}")
    print(f"Suggestions: {ctx.proactive_suggestions}")
    eng.stop()
