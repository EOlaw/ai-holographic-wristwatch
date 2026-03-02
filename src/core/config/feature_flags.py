"""
Feature Flag System for AI Holographic Wristwatch

Runtime-togglable feature flags with A/B testing percentage rollout, remote
override support, and full audit trail. Feature flags allow progressive
deployment, canary releases, and emergency kill-switches without redeployment.

Design principles:
- Every flag defaults to the SAFE / conservative state (disabled for unstable features)
- All flag evaluations are logged for audit
- A/B rollout uses deterministic hashing (user_id) so the same user always gets
  the same variant within a rollout window
- Flags can be overridden at runtime, but changes are tracked and reversible
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..constants import SystemConstants
from .base_config import BaseConfiguration, ConfigValidationResult


# ============================================================================
# Enumerations
# ============================================================================

class FlagState(Enum):
    """Explicit three-state flag: ON / OFF / ROLLOUT (percentage-based)."""
    ON = "on"
    OFF = "off"
    ROLLOUT = "rollout"      # Partial rollout by percentage


class FlagCategory(Enum):
    """Category for organizational grouping and bulk operations."""
    DISPLAY = "display"
    AI = "ai"
    SENSOR = "sensor"
    HEALTH = "health"
    COMMUNICATION = "communication"
    SECURITY = "security"
    POWER = "power"
    UI = "ui"
    EXPERIMENTAL = "experimental"
    KILL_SWITCH = "kill_switch"    # Emergency disablement flags


class OverrideSource(Enum):
    """Who/what applied the last override."""
    DEFAULT = "default"
    CONFIG_FILE = "config_file"
    REMOTE_SERVER = "remote_server"
    LOCAL_RUNTIME = "local_runtime"
    TEST_HARNESS = "test_harness"
    EMERGENCY = "emergency"


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class FeatureFlag:
    """
    A single feature flag with rollout, metadata, and override tracking.

    Fields:
        key:            Unique dot-separated identifier, e.g. "display.holographic.v2"
        default_state:  The state used when no override is applied
        rollout_pct:    0–100 inclusive; only relevant when state=ROLLOUT
        category:       Grouping for bulk enable/disable
        description:    Human-readable purpose
        owner:          Team or module that owns this flag
        expires_at:     Unix timestamp after which the flag should be cleaned up
        dependencies:   Other flag keys that MUST be ON for this flag to activate
        override_state: Current runtime override (None = use default_state)
        override_source: Where the current override came from
        override_timestamp: When the override was applied
        tags:           Free-form tags for search/filter
    """
    key: str
    default_state: FlagState = FlagState.OFF
    rollout_pct: float = 0.0
    category: FlagCategory = FlagCategory.EXPERIMENTAL
    description: str = ""
    owner: str = "platform"
    expires_at: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    override_state: Optional[FlagState] = None
    override_source: OverrideSource = OverrideSource.DEFAULT
    override_timestamp: Optional[float] = None
    tags: Set[str] = field(default_factory=set)

    @property
    def effective_state(self) -> FlagState:
        """Return the currently active state (override takes precedence)."""
        return self.override_state if self.override_state is not None else self.default_state

    @property
    def is_expired(self) -> bool:
        """True if the flag has an expiry and it has passed."""
        return self.expires_at is not None and time.time() > self.expires_at

    def is_enabled_for(self, user_id: Optional[str] = None) -> bool:
        """
        Evaluate whether this flag is enabled for the given user.

        For ROLLOUT state uses deterministic hashing of (key + user_id) to
        assign users consistently to the enabled or disabled bucket.
        """
        if self.is_expired:
            return False
        state = self.effective_state
        if state == FlagState.ON:
            return True
        if state == FlagState.OFF:
            return False
        # ROLLOUT — deterministic bucket assignment
        bucket_input = f"{self.key}:{user_id or 'anonymous'}"
        digest = hashlib.md5(bucket_input.encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return bucket < self.rollout_pct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "default_state": self.default_state.value,
            "rollout_pct": self.rollout_pct,
            "category": self.category.value,
            "description": self.description,
            "owner": self.owner,
            "expires_at": self.expires_at,
            "dependencies": self.dependencies,
            "override_state": self.override_state.value if self.override_state else None,
            "override_source": self.override_source.value,
            "override_timestamp": self.override_timestamp,
            "tags": list(self.tags),
        }


@dataclass
class FlagEvaluationEvent:
    """Audit record for a single flag evaluation."""
    key: str
    result: bool
    user_id: Optional[str]
    effective_state: str
    rollout_pct: float
    timestamp: float = field(default_factory=time.time)
    caller: str = ""


@dataclass
class FlagOverrideRecord:
    """Immutable audit record of a flag change."""
    key: str
    previous_state: Optional[str]
    new_state: str
    source: str
    timestamp: float = field(default_factory=time.time)
    reason: str = ""


# ============================================================================
# Feature Flag Registry
# ============================================================================

class FeatureFlagRegistry:
    """
    Thread-safe registry that stores all feature flags and evaluates them.

    Responsibilities:
    - Register / unregister flags
    - Evaluate flags (with optional user scoping for rollout)
    - Apply runtime overrides with audit trail
    - Trigger callbacks on flag changes
    - Bulk operations by category
    - Serialize / deserialize for remote sync
    """

    def __init__(self) -> None:
        self._flags: Dict[str, FeatureFlag] = {}
        self._lock = threading.RLock()
        self._audit_log: List[FlagOverrideRecord] = []
        self._eval_log: List[FlagEvaluationEvent] = []
        self._max_audit_entries: int = 10_000
        self._max_eval_entries: int = 50_000
        self._change_callbacks: List[Callable[[str, bool], None]] = []
        self._audit_enabled: bool = True
        self._eval_audit_enabled: bool = False  # Eval audit can be noisy

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, flag: FeatureFlag) -> None:
        """Register a flag. Existing flag with same key is replaced."""
        with self._lock:
            self._flags[flag.key] = flag

    def register_many(self, flags: List[FeatureFlag]) -> None:
        with self._lock:
            for flag in flags:
                self._flags[flag.key] = flag

    def unregister(self, key: str) -> bool:
        with self._lock:
            return self._flags.pop(key, None) is not None

    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        with self._lock:
            return self._flags.get(key)

    def list_flags(self, category: Optional[FlagCategory] = None) -> List[FeatureFlag]:
        with self._lock:
            flags = list(self._flags.values())
        if category:
            flags = [f for f in flags if f.category == category]
        return flags

    # ── Evaluation ───────────────────────────────────────────────────────────

    def is_enabled(self, key: str, user_id: Optional[str] = None,
                   caller: str = "") -> bool:
        """
        Evaluate a feature flag.

        Returns False for unknown flags (fail-safe).
        Respects dependency chain: if a dependency is disabled, this flag
        is also disabled regardless of its own state.
        """
        with self._lock:
            flag = self._flags.get(key)
        if flag is None:
            return False

        # Check dependencies first
        for dep_key in flag.dependencies:
            if not self.is_enabled(dep_key, user_id, caller):
                return False

        result = flag.is_enabled_for(user_id)

        if self._eval_audit_enabled:
            event = FlagEvaluationEvent(
                key=key,
                result=result,
                user_id=user_id,
                effective_state=flag.effective_state.value,
                rollout_pct=flag.rollout_pct,
                caller=caller,
            )
            with self._lock:
                self._eval_log.append(event)
                if len(self._eval_log) > self._max_eval_entries:
                    self._eval_log = self._eval_log[-self._max_eval_entries:]

        return result

    def is_disabled(self, key: str, user_id: Optional[str] = None) -> bool:
        return not self.is_enabled(key, user_id)

    # ── Overrides ────────────────────────────────────────────────────────────

    def override(self, key: str, state: FlagState,
                 source: OverrideSource = OverrideSource.LOCAL_RUNTIME,
                 reason: str = "") -> bool:
        """
        Apply a runtime override to a flag. Returns True if flag existed.
        """
        with self._lock:
            flag = self._flags.get(key)
            if flag is None:
                return False
            previous = flag.override_state.value if flag.override_state else None
            flag.override_state = state
            flag.override_source = source
            flag.override_timestamp = time.time()

            if self._audit_enabled:
                record = FlagOverrideRecord(
                    key=key,
                    previous_state=previous,
                    new_state=state.value,
                    source=source.value,
                    reason=reason,
                )
                self._audit_log.append(record)
                if len(self._audit_log) > self._max_audit_entries:
                    self._audit_log = self._audit_log[-self._max_audit_entries:]

        # Fire callbacks outside the lock
        enabled = self.is_enabled(key)
        for cb in self._change_callbacks:
            try:
                cb(key, enabled)
            except Exception:
                pass
        return True

    def clear_override(self, key: str, reason: str = "") -> bool:
        """Remove runtime override, reverting flag to its default state."""
        return self.override(key, None, OverrideSource.LOCAL_RUNTIME, reason)  # type: ignore[arg-type]

    def clear_all_overrides(self) -> int:
        """Remove all runtime overrides. Returns count cleared."""
        cleared = 0
        with self._lock:
            for flag in self._flags.values():
                if flag.override_state is not None:
                    flag.override_state = None
                    flag.override_source = OverrideSource.DEFAULT
                    flag.override_timestamp = None
                    cleared += 1
        return cleared

    # ── Bulk operations ───────────────────────────────────────────────────────

    def enable_category(self, category: FlagCategory,
                        source: OverrideSource = OverrideSource.LOCAL_RUNTIME) -> int:
        """Enable all flags in a category. Returns count affected."""
        count = 0
        for flag in self.list_flags(category):
            if self.override(flag.key, FlagState.ON, source):
                count += 1
        return count

    def disable_category(self, category: FlagCategory,
                         source: OverrideSource = OverrideSource.LOCAL_RUNTIME) -> int:
        """Disable all flags in a category. Returns count affected."""
        count = 0
        for flag in self.list_flags(category):
            if self.override(flag.key, FlagState.OFF, source):
                count += 1
        return count

    def emergency_kill(self, category: FlagCategory, reason: str) -> int:
        """Emergency kill-switch for an entire category."""
        return self.disable_category(category, OverrideSource.EMERGENCY)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def add_change_callback(self, callback: Callable[[str, bool], None]) -> None:
        """Register a callback invoked when any flag changes: cb(key, is_enabled)."""
        self._change_callbacks.append(callback)

    def remove_change_callback(self, callback: Callable[[str, bool], None]) -> None:
        self._change_callbacks = [c for c in self._change_callbacks if c != callback]

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {key: flag.to_dict() for key, flag in self._flags.items()}

    def apply_remote_overrides(self, remote_dict: Dict[str, Any]) -> int:
        """
        Apply a dict of {flag_key: state_string} from a remote config server.
        Returns number of flags updated.
        """
        count = 0
        for key, raw_state in remote_dict.items():
            try:
                state = FlagState(raw_state)
                if self.override(key, state, OverrideSource.REMOTE_SERVER):
                    count += 1
            except (ValueError, KeyError):
                pass
        return count

    def get_audit_log(self, limit: int = 100) -> List[FlagOverrideRecord]:
        with self._lock:
            return list(self._audit_log[-limit:])


# ============================================================================
# Predefined Platform Flags
# ============================================================================

def _build_default_flags() -> List[FeatureFlag]:
    """Return the canonical set of feature flags for the AI Holographic Wristwatch."""
    return [
        # ── Holographic Display ──────────────────────────────────────────────
        FeatureFlag(
            key="display.holographic.enabled",
            default_state=FlagState.ON,
            category=FlagCategory.DISPLAY,
            description="Master switch for holographic projection subsystem",
            owner="display_team",
        ),
        FeatureFlag(
            key="display.holographic.depth_occlusion",
            default_state=FlagState.ROLLOUT,
            rollout_pct=50.0,
            category=FlagCategory.DISPLAY,
            description="Real-time depth occlusion rendering — computationally expensive",
            owner="display_team",
            dependencies=["display.holographic.enabled"],
        ),
        FeatureFlag(
            key="display.holographic.eye_tracking_adaptation",
            default_state=FlagState.ON,
            category=FlagCategory.DISPLAY,
            description="Adapt hologram position/focus based on eye gaze",
            owner="display_team",
            dependencies=["display.holographic.enabled"],
        ),
        FeatureFlag(
            key="display.watchface.ambient_mode",
            default_state=FlagState.ON,
            category=FlagCategory.DISPLAY,
            description="Always-on watchface display at reduced brightness",
            owner="display_team",
        ),

        # ── AI System ────────────────────────────────────────────────────────
        FeatureFlag(
            key="ai.cloud_inference",
            default_state=FlagState.ON,
            category=FlagCategory.AI,
            description="Allow sending prompts to cloud AI for complex queries",
            owner="ai_team",
        ),
        FeatureFlag(
            key="ai.on_device_inference",
            default_state=FlagState.ON,
            category=FlagCategory.AI,
            description="Run small models directly on device for privacy",
            owner="ai_team",
        ),
        FeatureFlag(
            key="ai.streaming_responses",
            default_state=FlagState.ON,
            category=FlagCategory.AI,
            description="Stream AI responses token-by-token for lower latency",
            owner="ai_team",
        ),
        FeatureFlag(
            key="ai.personality.adaptation",
            default_state=FlagState.ON,
            category=FlagCategory.AI,
            description="Allow AI personality to adapt based on user feedback",
            owner="ai_team",
        ),
        FeatureFlag(
            key="ai.hallucination_detection",
            default_state=FlagState.ON,
            category=FlagCategory.AI,
            description="Cross-check AI responses against knowledge base",
            owner="ai_team",
        ),
        FeatureFlag(
            key="ai.multimodal",
            default_state=FlagState.ON,
            category=FlagCategory.AI,
            description="Enable combined voice + holographic + text responses",
            owner="ai_team",
        ),

        # ── Health & Sensors ─────────────────────────────────────────────────
        FeatureFlag(
            key="health.continuous_heart_rate",
            default_state=FlagState.ON,
            category=FlagCategory.HEALTH,
            description="Continuous HR monitoring (increases battery drain)",
            owner="health_team",
        ),
        FeatureFlag(
            key="health.ecg_monitoring",
            default_state=FlagState.ON,
            category=FlagCategory.HEALTH,
            description="ECG single-lead cardiac monitoring",
            owner="health_team",
        ),
        FeatureFlag(
            key="health.blood_pressure",
            default_state=FlagState.ROLLOUT,
            rollout_pct=25.0,
            category=FlagCategory.HEALTH,
            description="Cuffless blood pressure estimation — under validation",
            owner="health_team",
            tags={"beta", "clinical_validation"},
        ),
        FeatureFlag(
            key="health.sleep_staging",
            default_state=FlagState.ON,
            category=FlagCategory.HEALTH,
            description="Automatic sleep stage classification (REM/light/deep)",
            owner="health_team",
        ),
        FeatureFlag(
            key="health.stress_detection",
            default_state=FlagState.ON,
            category=FlagCategory.HEALTH,
            description="HRV + EDA-based stress detection",
            owner="health_team",
        ),
        FeatureFlag(
            key="health.fall_detection",
            default_state=FlagState.ON,
            category=FlagCategory.HEALTH,
            description="Automatic fall detection with emergency alert",
            owner="health_team",
        ),
        FeatureFlag(
            key="health.hydration_tracking",
            default_state=FlagState.ROLLOUT,
            rollout_pct=30.0,
            category=FlagCategory.HEALTH,
            description="Optical hydration estimation",
            owner="health_team",
            tags={"beta"},
        ),

        # ── Communication ────────────────────────────────────────────────────
        FeatureFlag(
            key="comm.bluetooth",
            default_state=FlagState.ON,
            category=FlagCategory.COMMUNICATION,
            description="Bluetooth LE communication stack",
            owner="comms_team",
        ),
        FeatureFlag(
            key="comm.wifi",
            default_state=FlagState.ON,
            category=FlagCategory.COMMUNICATION,
            description="Wi-Fi 6 stack",
            owner="comms_team",
        ),
        FeatureFlag(
            key="comm.nfc",
            default_state=FlagState.ON,
            category=FlagCategory.COMMUNICATION,
            description="NFC for payments and device pairing",
            owner="comms_team",
        ),
        FeatureFlag(
            key="comm.cloud_sync",
            default_state=FlagState.ON,
            category=FlagCategory.COMMUNICATION,
            description="Sync health and configuration data to cloud",
            owner="comms_team",
            dependencies=["comm.wifi"],
        ),
        FeatureFlag(
            key="comm.mesh_networking",
            default_state=FlagState.OFF,
            category=FlagCategory.COMMUNICATION,
            description="Experimental mesh networking between watches",
            owner="comms_team",
            tags={"experimental"},
        ),

        # ── Security ─────────────────────────────────────────────────────────
        FeatureFlag(
            key="security.biometric_auth",
            default_state=FlagState.ON,
            category=FlagCategory.SECURITY,
            description="Biometric (face/fingerprint) primary authentication",
            owner="security_team",
        ),
        FeatureFlag(
            key="security.secure_enclave",
            default_state=FlagState.ON,
            category=FlagCategory.SECURITY,
            description="Store cryptographic keys in hardware secure element",
            owner="security_team",
        ),
        FeatureFlag(
            key="security.certificate_pinning",
            default_state=FlagState.ON,
            category=FlagCategory.SECURITY,
            description="TLS certificate pinning for all cloud connections",
            owner="security_team",
        ),
        FeatureFlag(
            key="security.tamper_detection",
            default_state=FlagState.ON,
            category=FlagCategory.SECURITY,
            description="Physical tamper detection via accelerometer + secure boot",
            owner="security_team",
        ),

        # ── Power ────────────────────────────────────────────────────────────
        FeatureFlag(
            key="power.solar_harvesting",
            default_state=FlagState.ON,
            category=FlagCategory.POWER,
            description="Use ambient solar panel to trickle-charge battery",
            owner="power_team",
        ),
        FeatureFlag(
            key="power.kinetic_harvesting",
            default_state=FlagState.ON,
            category=FlagCategory.POWER,
            description="Use wrist motion to generate power via piezo",
            owner="power_team",
        ),
        FeatureFlag(
            key="power.auto_power_save",
            default_state=FlagState.ON,
            category=FlagCategory.POWER,
            description="Automatically enter power-save mode at low battery",
            owner="power_team",
        ),

        # ── UI / UX ──────────────────────────────────────────────────────────
        FeatureFlag(
            key="ui.haptic_feedback",
            default_state=FlagState.ON,
            category=FlagCategory.UI,
            description="Haptic feedback for touch interactions",
            owner="ui_team",
        ),
        FeatureFlag(
            key="ui.contextual_suggestions",
            default_state=FlagState.ON,
            category=FlagCategory.UI,
            description="Show proactive AI suggestions on watchface",
            owner="ui_team",
        ),
        FeatureFlag(
            key="ui.voice_wake_word",
            default_state=FlagState.ON,
            category=FlagCategory.UI,
            description="Wake word detection for hands-free activation",
            owner="ui_team",
        ),

        # ── Kill Switches ────────────────────────────────────────────────────
        FeatureFlag(
            key="kill_switch.disable_all_radios",
            default_state=FlagState.OFF,
            category=FlagCategory.KILL_SWITCH,
            description="Emergency: disable all wireless radios immediately",
            owner="security_team",
        ),
        FeatureFlag(
            key="kill_switch.disable_holographic",
            default_state=FlagState.OFF,
            category=FlagCategory.KILL_SWITCH,
            description="Emergency: disable holographic projector",
            owner="display_team",
        ),
        FeatureFlag(
            key="kill_switch.force_data_wipe",
            default_state=FlagState.OFF,
            category=FlagCategory.KILL_SWITCH,
            description="Emergency: trigger secure data wipe on next boot",
            owner="security_team",
        ),
    ]


# ============================================================================
# Feature Flags Configuration (integrates with BaseConfiguration)
# ============================================================================

@dataclass
class FeatureFlagsConfig(BaseConfiguration):
    """
    Configuration that controls the feature flag system itself
    (not the individual flags — those are in FeatureFlagRegistry).
    """
    enable_remote_overrides: bool = True
    remote_sync_interval_seconds: int = 300         # 5 minutes
    remote_flags_url: Optional[str] = None          # Populated from vault/env
    audit_flag_changes: bool = True
    audit_flag_evaluations: bool = False             # High-volume; off by default
    allow_test_harness_overrides: bool = False       # Only True in dev/test
    stale_flag_warn_days: int = 90                   # Warn if flag unchanged > N days
    max_rollout_pct: float = 100.0
    flags_override_dict: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        if self.remote_sync_interval_seconds < 10:
            result.add_issue("remote_sync_interval_seconds", "out_of_range",
                             "Sync interval must be >= 10 seconds")
        self._check_range(result, "max_rollout_pct",
                          self.max_rollout_pct, 0.0, 100.0)
        return result


# ============================================================================
# Global Feature Flag Manager
# ============================================================================

class FeatureFlagManager:
    """
    Singleton facade combining FeatureFlagsConfig + FeatureFlagRegistry.

    This is the primary entry point for all feature flag checks in the app.
    """

    def __init__(self, config: Optional[FeatureFlagsConfig] = None) -> None:
        self._config = config or FeatureFlagsConfig()
        self._registry = FeatureFlagRegistry()
        self._registry._audit_enabled = self._config.audit_flag_changes
        self._registry._eval_audit_enabled = self._config.audit_flag_evaluations
        # Register the canonical default flags
        self._registry.register_many(_build_default_flags())
        # Apply any static overrides from config
        if self._config.flags_override_dict:
            self._registry.apply_remote_overrides(self._config.flags_override_dict)

    # ── Core evaluation ───────────────────────────────────────────────────────

    def is_enabled(self, key: str, user_id: Optional[str] = None,
                   caller: str = "") -> bool:
        return self._registry.is_enabled(key, user_id, caller)

    def is_disabled(self, key: str, user_id: Optional[str] = None) -> bool:
        return not self.is_enabled(key, user_id)

    # ── Shortcuts for common flags ────────────────────────────────────────────

    @property
    def holographic_enabled(self) -> bool:
        return self.is_enabled("display.holographic.enabled")

    @property
    def cloud_ai_enabled(self) -> bool:
        return self.is_enabled("ai.cloud_inference")

    @property
    def on_device_ai_enabled(self) -> bool:
        return self.is_enabled("ai.on_device_inference")

    @property
    def biometric_auth_enabled(self) -> bool:
        return self.is_enabled("security.biometric_auth")

    @property
    def fall_detection_enabled(self) -> bool:
        return self.is_enabled("health.fall_detection")

    # ── Delegation ───────────────────────────────────────────────────────────

    def override(self, key: str, state: FlagState,
                 source: OverrideSource = OverrideSource.LOCAL_RUNTIME,
                 reason: str = "") -> bool:
        return self._registry.override(key, state, source, reason)

    def clear_override(self, key: str, reason: str = "") -> bool:
        return self._registry.clear_override(key, reason)

    def emergency_kill(self, category: FlagCategory, reason: str) -> int:
        return self._registry.emergency_kill(category, reason)

    def add_change_callback(self, callback: Callable[[str, bool], None]) -> None:
        self._registry.add_change_callback(callback)

    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        return self._registry.get_flag(key)

    def list_flags(self, category: Optional[FlagCategory] = None) -> List[FeatureFlag]:
        return self._registry.list_flags(category)

    def apply_remote_overrides(self, remote_dict: Dict[str, Any]) -> int:
        return self._registry.apply_remote_overrides(remote_dict)

    def get_audit_log(self, limit: int = 100) -> List[FlagOverrideRecord]:
        return self._registry.get_audit_log(limit)

    @property
    def registry(self) -> FeatureFlagRegistry:
        return self._registry

    @property
    def config(self) -> FeatureFlagsConfig:
        return self._config


# ============================================================================
# Global singleton
# ============================================================================

_feature_flag_manager: Optional[FeatureFlagManager] = None
_ff_lock = threading.Lock()


def get_feature_flags(config: Optional[FeatureFlagsConfig] = None) -> FeatureFlagManager:
    """Return the global FeatureFlagManager singleton, creating it if needed."""
    global _feature_flag_manager
    with _ff_lock:
        if _feature_flag_manager is None:
            _feature_flag_manager = FeatureFlagManager(config)
    return _feature_flag_manager


def reset_feature_flags() -> None:
    """Reset the global manager (for testing)."""
    global _feature_flag_manager
    with _ff_lock:
        _feature_flag_manager = None


# ============================================================================
# Tests
# ============================================================================

def run_feature_flag_tests() -> None:
    print("Testing feature flag system...")

    reset_feature_flags()
    mgr = get_feature_flags()

    # Default ON flag
    assert mgr.is_enabled("display.holographic.enabled") is True
    # Default OFF kill switch
    assert mgr.is_enabled("kill_switch.disable_all_radios") is False

    # Override to OFF
    mgr.override("display.holographic.enabled", FlagState.OFF, reason="test")
    assert mgr.is_enabled("display.holographic.enabled") is False

    # Clear override
    mgr.clear_override("display.holographic.enabled")
    assert mgr.is_enabled("display.holographic.enabled") is True

    # Rollout determinism
    flag = mgr.get_flag("health.blood_pressure")
    assert flag is not None
    r1 = flag.is_enabled_for("user_abc")
    r2 = flag.is_enabled_for("user_abc")
    assert r1 == r2, "Rollout must be deterministic for same user"

    # Unknown flag → False (fail-safe)
    assert mgr.is_enabled("nonexistent.flag") is False

    # Dependency check: depth_occlusion depends on holographic.enabled
    mgr.override("display.holographic.enabled", FlagState.OFF)
    assert mgr.is_enabled("display.holographic.depth_occlusion") is False
    mgr.clear_override("display.holographic.enabled")

    # Emergency kill
    killed = mgr.emergency_kill(FlagCategory.KILL_SWITCH, "drill")
    # Audit log populated
    log = mgr.get_audit_log()
    assert len(log) > 0

    # Remote override application
    applied = mgr.apply_remote_overrides({"comm.mesh_networking": "on"})
    assert applied == 1
    assert mgr.is_enabled("comm.mesh_networking") is True

    reset_feature_flags()
    print("  Feature flag tests passed.")


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "FlagState", "FlagCategory", "OverrideSource",
    "FeatureFlag", "FlagEvaluationEvent", "FlagOverrideRecord",
    "FeatureFlagRegistry",
    "FeatureFlagsConfig",
    "FeatureFlagManager",
    "get_feature_flags", "reset_feature_flags",
]


if __name__ == "__main__":
    print("AI Holographic Wristwatch — Feature Flag System")
    print("=" * 55)
    run_feature_flag_tests()

    mgr = get_feature_flags()
    print(f"\nRegistered flags: {len(mgr.list_flags())}")
    for cat in FlagCategory:
        flags = mgr.list_flags(cat)
        if flags:
            print(f"  [{cat.value}] {len(flags)} flags")
