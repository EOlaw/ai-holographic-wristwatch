"""
AI System Configuration for AI Holographic Wristwatch

Typed configuration classes for every AI subsystem: language models,
personality engine, conversation management, learning systems, and safety
filters. All values have production-safe defaults and full validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..constants import AIConstants
from .base_config import BaseConfiguration, ConfigValidationResult


# ============================================================================
# Enumerations
# ============================================================================

class ModelBackend(Enum):
    """Where the AI model inference runs."""
    ON_DEVICE = "on_device"
    EDGE_NODE = "edge_node"
    CLOUD = "cloud"
    HYBRID = "hybrid"          # on-device for privacy, cloud for complex queries


class PersonalityProfile(Enum):
    """Pre-built personality profile presets."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CONCISE = "concise"
    EMPATHETIC = "empathetic"
    TECHNICAL = "technical"
    CUSTOM = "custom"


# ============================================================================
# Sub-configurations
# ============================================================================

@dataclass
class ModelConfig(BaseConfiguration):
    """Configuration for a single language model."""
    model_id: str = "claude-sonnet-4-6"
    backend: str = ModelBackend.HYBRID.value
    max_context_tokens: int = AIConstants.MAX_CONTEXT_TOKENS
    max_output_tokens: int = AIConstants.MAX_OUTPUT_TOKENS
    temperature: float = AIConstants.DEFAULT_TEMPERATURE
    top_p: float = AIConstants.TOP_P_DEFAULT
    top_k: int = AIConstants.TOP_K_DEFAULT
    frequency_penalty: float = AIConstants.FREQUENCY_PENALTY
    presence_penalty: float = AIConstants.PRESENCE_PENALTY
    stop_sequences: List[str] = field(default_factory=list)
    timeout_ms: float = AIConstants.EDGE_INFERENCE_TIMEOUT_MS
    cache_responses: bool = True
    cache_ttl_seconds: int = 300

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_required(result, "model_id", self.model_id)
        self._check_range(result, "temperature", self.temperature, 0.0, 2.0)
        self._check_range(result, "top_p", self.top_p, 0.0, 1.0)
        self._check_range(result, "max_output_tokens", self.max_output_tokens, 1, 8192)
        if self.timeout_ms <= 0:
            result.add_issue("timeout_ms", "out_of_range", "timeout_ms must be > 0")
        return result


@dataclass
class PersonalityConfig(BaseConfiguration):
    """Configuration for the AI personality engine."""
    profile: str = PersonalityProfile.FRIENDLY.value
    name: str = "Aria"
    voice_gender: str = "neutral"
    formality_level: float = 0.4          # 0.0 casual – 1.0 formal
    empathy_level: float = 0.7            # 0.0 none – 1.0 maximum
    verbosity: float = 0.5                # 0.0 terse – 1.0 verbose
    humor_level: float = 0.2              # 0.0 none – 1.0 frequent
    adaptation_rate: float = AIConstants.PERSONALITY_ADAPTATION_RATE
    max_personality_drift: float = AIConstants.MAX_PERSONALITY_DRIFT
    emotional_decay_rate: float = AIConstants.EMOTIONAL_DECAY_RATE
    allowed_topics: List[str] = field(default_factory=list)
    restricted_topics: List[str] = field(default_factory=list)
    custom_traits: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_required(result, "name", self.name)
        self._check_range(result, "formality_level", self.formality_level, 0.0, 1.0)
        self._check_range(result, "empathy_level", self.empathy_level, 0.0, 1.0)
        self._check_range(result, "adaptation_rate", self.adaptation_rate, 0.0, 1.0)
        return result


@dataclass
class ConversationConfig(BaseConfiguration):
    """Configuration for the conversational AI pipeline."""
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "es", "fr", "de", "zh", "ja", "ar", "pt", "hi"]
    )
    max_history_turns: int = AIConstants.MAX_CONVERSATION_HISTORY
    session_timeout_seconds: int = AIConstants.CONVERSATION_TIMEOUT_SECONDS
    min_response_confidence: float = AIConstants.MIN_RESPONSE_CONFIDENCE
    clarification_threshold: float = AIConstants.CLARIFICATION_THRESHOLD
    max_clarification_attempts: int = AIConstants.MAX_CLARIFICATION_ATTEMPTS
    enable_barge_in: bool = True           # allow interrupting the AI mid-response
    enable_follow_up_suggestions: bool = True
    max_follow_up_suggestions: int = 3
    context_compression_threshold: int = int(AIConstants.MAX_CONTEXT_TOKENS * 0.8)
    enable_sentiment_detection: bool = True
    enable_intent_caching: bool = True

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_required(result, "default_language", self.default_language)
        self._check_range(result, "min_response_confidence",
                          self.min_response_confidence, 0.0, 1.0)
        self._check_range(result, "clarification_threshold",
                          self.clarification_threshold, 0.0, 1.0)
        if self.max_history_turns < 1:
            result.add_issue("max_history_turns", "out_of_range",
                             "max_history_turns must be >= 1")
        return result


@dataclass
class LearningConfig(BaseConfiguration):
    """Configuration for online learning and personalization."""
    enable_online_learning: bool = True
    enable_personalization: bool = True
    learning_rate: float = 0.001
    memory_decay_rate: float = 0.01
    episodic_memory_capacity: int = AIConstants.EPISODIC_MEMORY_CAPACITY
    semantic_memory_capacity: int = AIConstants.SEMANTIC_MEMORY_CAPACITY
    short_term_memory_ttl_seconds: int = AIConstants.SHORT_TERM_MEMORY_SECONDS
    long_term_retention_days: int = AIConstants.LONG_TERM_MEMORY_RETENTION_DAYS
    consolidation_interval_hours: int = AIConstants.MEMORY_CONSOLIDATION_INTERVAL_HOURS
    federated_learning_enabled: bool = False
    federated_upload_interval_hours: int = 24
    min_interactions_before_adaptation: int = 10

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_range(result, "learning_rate", self.learning_rate, 0.0, 1.0)
        self._check_range(result, "memory_decay_rate", self.memory_decay_rate, 0.0, 1.0)
        if self.episodic_memory_capacity < 100:
            result.add_issue("episodic_memory_capacity", "out_of_range",
                             "Must be at least 100 episodes")
        return result


@dataclass
class SafetyConfig(BaseConfiguration):
    """Configuration for AI safety filters and ethical guardrails."""
    enable_safety_filter: bool = True
    safety_score_threshold: float = AIConstants.SAFETY_SCORE_THRESHOLD
    enable_bias_detection: bool = True
    bias_detection_threshold: float = AIConstants.BIAS_DETECTION_THRESHOLD
    enable_hallucination_detection: bool = True
    hallucination_threshold: float = AIConstants.HALLUCINATION_DETECTION_CONFIDENCE
    max_retry_on_safety_violation: int = AIConstants.MAX_RETRY_ON_SAFETY_VIOLATION
    blocked_content_categories: List[str] = field(default_factory=lambda: [
        "violence", "self_harm", "illegal_activities", "explicit_adult",
        "misinformation", "hate_speech"
    ])
    medical_advice_disclaimer: bool = True
    emergency_escalation_enabled: bool = True
    audit_all_decisions: bool = True

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)
        self._check_range(result, "safety_score_threshold",
                          self.safety_score_threshold, 0.0, 1.0)
        self._check_range(result, "bias_detection_threshold",
                          self.bias_detection_threshold, 0.0, 1.0)
        if self.max_retry_on_safety_violation < 0:
            result.add_issue("max_retry_on_safety_violation", "out_of_range",
                             "Must be >= 0")
        return result


# ============================================================================
# Root AI Configuration
# ============================================================================

@dataclass
class AIConfig(BaseConfiguration):
    """
    Aggregate root for all AI system configuration.

    Instantiate this class to get a fully typed, validated AI configuration
    from a config dict (from YAML/JSON files or AppConfig).

    Usage:
        raw = app_config.get_section("ai")
        ai_cfg = AIConfig(raw)
        validation = ai_cfg.validate()
    """

    # Nested configuration sections
    model: ModelConfig = field(default_factory=ModelConfig)
    fallback_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            model_id="claude-haiku-4-5-20251001",
            max_output_tokens=256,
            timeout_ms=100.0,
        )
    )
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # Top-level AI flags
    enable_cloud_ai: bool = True
    enable_on_device_ai: bool = True
    prefer_on_device: bool = True          # Use on-device first for privacy
    enable_streaming_responses: bool = True
    enable_multimodal: bool = True          # Voice + holographic + text
    debug_mode: bool = False
    log_all_conversations: bool = False    # False in production for privacy

    def __post_init__(self):
        """Upgrade nested dicts to typed config objects after __init__."""
        if isinstance(self.model, dict):
            self.model = ModelConfig(self.model)
        if isinstance(self.personality, dict):
            self.personality = PersonalityConfig(self.personality)
        if isinstance(self.conversation, dict):
            self.conversation = ConversationConfig(self.conversation)
        if isinstance(self.learning, dict):
            self.learning = LearningConfig(self.learning)
        if isinstance(self.safety, dict):
            self.safety = SafetyConfig(self.safety)

    def validate(self) -> ConfigValidationResult:
        result = ConfigValidationResult(is_valid=True)

        # Validate each section
        for section_name in ("model", "personality", "conversation",
                              "learning", "safety"):
            section: BaseConfiguration = getattr(self, section_name)
            section_result = section.validate()
            for issue in section_result.issues:
                result.add_issue(
                    f"{section_name}.{issue.field_path}",
                    issue.issue_type,
                    issue.message,
                    issue.is_fatal
                )
            result.warnings.extend(section_result.warnings)

        # Logical consistency checks
        if not self.enable_cloud_ai and not self.enable_on_device_ai:
            result.add_issue("enable_*", "invalid",
                             "At least one of cloud or on-device AI must be enabled")

        return result


# ============================================================================
# Convenience factory
# ============================================================================

def create_ai_config(config_dict: Optional[Dict[str, Any]] = None) -> AIConfig:
    """
    Create a validated AIConfig from a raw config dictionary.

    Args:
        config_dict: Raw config dict (e.g., from AppConfig.get_section("ai")).
                     Uses all defaults if None.
    Returns:
        Validated AIConfig instance.
    Raises:
        InvalidConfigError: If validation fails with fatal issues.
    """
    from ..exceptions import InvalidConfigError
    config = AIConfig(config_dict or {})
    result = config.validate()
    if not result.is_valid:
        issues_str = "; ".join(f"{i.field_path}: {i.message}"
                                for i in result.fatal_issues)
        raise InvalidConfigError(f"AI configuration is invalid: {issues_str}",
                                  config_key="ai")
    return config


# ============================================================================
# Tests
# ============================================================================

def run_ai_config_tests() -> None:
    """Smoke test for AI configuration."""
    print("Testing AI configuration...")

    # Default config
    config = AIConfig()
    result = config.validate()
    assert result.is_valid, f"Default config should be valid: {result.issues}"

    # Custom config
    custom = AIConfig({
        "model": {"temperature": 0.9, "max_output_tokens": 1024},
        "personality": {"name": "Nova", "empathy_level": 0.9},
        "safety": {"safety_score_threshold": 0.95},
    })
    assert custom.model.temperature == 0.9
    assert custom.personality.name == "Nova"
    assert custom.safety.safety_score_threshold == 0.95

    # Invalid config detection
    bad = AIConfig({"model": {"temperature": 5.0}})   # out of range
    bad_result = bad.validate()
    assert not bad_result.is_valid

    print("  AI config tests passed.")


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "ModelBackend", "PersonalityProfile",
    "ModelConfig", "PersonalityConfig", "ConversationConfig",
    "LearningConfig", "SafetyConfig", "AIConfig",
    "create_ai_config",
]


if __name__ == "__main__":
    print("AI Holographic Wristwatch — AI Configuration Module")
    print("=" * 55)
    run_ai_config_tests()

    cfg = AIConfig()
    print(f"\nDefault AI config:")
    print(f"  Model:       {cfg.model.model_id}")
    print(f"  Temperature: {cfg.model.temperature}")
    print(f"  Max tokens:  {cfg.model.max_output_tokens}")
    print(f"  Personality: {cfg.personality.name} ({cfg.personality.profile})")
    print(f"  Safety:      threshold={cfg.safety.safety_score_threshold}")
