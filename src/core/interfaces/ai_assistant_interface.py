"""
AI Assistant Interface Contracts for AI Holographic Wristwatch System

Defines the abstract contracts that all AI assistant implementations must
satisfy. This includes the main conversational AI engine, personality system,
response generator, and safety filter. Programming against these interfaces
enables model swapping, A/B testing, and full mock-based unit testing of all
layers that depend on AI behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from ..constants import AIConstants, AIMode, AlertSeverity
from ..exceptions import AISystemError, ConversationError, SafetyViolationError


# ============================================================================
# Enumerations
# ============================================================================

class ConversationRole(Enum):
    """Speaker role in a conversation turn."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class ResponseModality(Enum):
    """Output channel for the AI response."""
    TEXT = "text"
    VOICE = "voice"
    HOLOGRAPHIC = "holographic"
    HAPTIC = "haptic"
    MULTIMODAL = "multimodal"


class EmotionalState(Enum):
    """AI personality emotional state."""
    NEUTRAL = "neutral"
    ENGAGED = "engaged"
    CONCERNED = "concerned"
    ENTHUSIASTIC = "enthusiastic"
    CALM = "calm"
    EMPATHETIC = "empathetic"


class ConversationStatus(Enum):
    """Current state of a conversation session."""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    RESPONDING = "responding"
    WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
    EXECUTING_TASK = "executing_task"
    ERROR = "error"


# ============================================================================
# Data Containers
# ============================================================================

@dataclass
class ConversationTurn:
    """A single exchange in a conversation."""
    turn_id: str
    conversation_id: str
    role: ConversationRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    audio_duration_seconds: Optional[float] = None


@dataclass
class ConversationContext:
    """Full context for one conversation session."""
    conversation_id: str
    user_id: str
    device_id: str
    history: List[ConversationTurn] = field(default_factory=list)
    mode: AIMode = AIMode.CONVERSATIONAL
    language: str = "en"
    timezone: str = "UTC"
    session_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_profile: Dict[str, Any] = field(default_factory=dict)
    environmental_context: Dict[str, Any] = field(default_factory=dict)
    active_tasks: List[str] = field(default_factory=list)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Append a turn, evicting oldest if history exceeds max length."""
        self.history.append(turn)
        if len(self.history) > AIConstants.MAX_CONVERSATION_HISTORY:
            self.history.pop(0)

    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Return the most recent n turns."""
        return self.history[-n:]


@dataclass
class UserInput:
    """Structured user input for the AI pipeline."""
    input_id: str
    raw_text: str
    language: str = "en"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    audio_data: Optional[bytes] = None
    audio_duration_seconds: Optional[float] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIResponse:
    """Structured output from the AI assistant."""
    response_id: str
    conversation_id: str
    text: str
    modality: ResponseModality = ResponseModality.TEXT
    confidence: float = 1.0
    safety_score: float = 1.0
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    emotional_tone: EmotionalState = EmotionalState.NEUTRAL
    follow_up_suggestions: List[str] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_safe(self) -> bool:
        """Return True if the safety score clears the minimum threshold."""
        return self.safety_score >= AIConstants.SAFETY_SCORE_THRESHOLD


@dataclass
class PersonalityState:
    """Snapshot of the AI personality engine state."""
    personality_id: str
    emotional_state: EmotionalState
    engagement_level: float         # 0.0–1.0
    empathy_score: float            # 0.0–1.0
    formality_level: float          # 0.0 (casual) – 1.0 (formal)
    user_rapport: float             # 0.0–1.0
    active_traits: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SafetyAssessment:
    """Result of running an AI response through the safety filter."""
    content: str
    is_safe: bool
    safety_score: float             # 0.0 = unsafe, 1.0 = fully safe
    violations: List[Dict[str, Any]] = field(default_factory=list)
    bias_detected: bool = False
    bias_score: float = 0.0
    recommended_action: str = "allow"   # "allow", "modify", "block"
    modified_content: Optional[str] = None


@dataclass
class IntentResult:
    """Result of intent recognition on user input."""
    intent_id: str
    intent_name: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    sub_intents: List[str] = field(default_factory=list)
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None


# ============================================================================
# AI Assistant Core Interface
# ============================================================================

class AIAssistantInterface(ABC):
    """
    Abstract contract for the main AI assistant engine.

    Covers the full request-response pipeline: input processing → context
    management → response generation → safety filtering → output.
    All I/O-bound operations are async to prevent blocking the event loop.
    """

    @abstractmethod
    async def process_input(self, user_input: UserInput,
                            context: ConversationContext) -> AIResponse:
        """
        Process user input and produce an AI response.

        Args:
            user_input: Structured input from the user.
            context:    Full conversation context including history.
        Returns:
            AIResponse with text, modality, safety score, and metadata.
        Raises:
            ConversationError: If processing fails.
            SafetyViolationError: If input is unsafe.
        """
        ...

    @abstractmethod
    async def stream_response(self, user_input: UserInput,
                              context: ConversationContext
                              ) -> AsyncIterator[str]:
        """
        Stream the AI response token-by-token for low-latency display.

        Args:
            user_input: Structured user input.
            context:    Full conversation context.
        Yields:
            Successive text tokens as they are generated.
        """
        ...

    @abstractmethod
    async def recognize_intent(self, user_input: UserInput) -> IntentResult:
        """
        Classify the user's intent from raw input.

        Args:
            user_input: Raw user input to classify.
        Returns:
            IntentResult with intent name, confidence, and extracted entities.
        """
        ...

    @abstractmethod
    async def update_context(self, context: ConversationContext,
                             turn: ConversationTurn) -> ConversationContext:
        """
        Incorporate a new conversation turn into the context.

        Args:
            context: Current conversation context.
            turn:    The new turn to integrate.
        Returns:
            Updated ConversationContext.
        """
        ...

    @abstractmethod
    def get_conversation_status(self, conversation_id: str) -> ConversationStatus:
        """Return current processing status of a conversation."""
        ...

    @abstractmethod
    async def reset_conversation(self, conversation_id: str) -> None:
        """Clear conversation history and reset context for a session."""
        ...

    @abstractmethod
    def get_confidence_score(self) -> float:
        """Return the AI engine's current overall confidence level (0.0–1.0)."""
        ...

    @abstractmethod
    async def explain_response(self, response: AIResponse) -> str:
        """
        Return a brief explanation of why the AI gave a particular response.
        Used for transparency / explainability features.

        Args:
            response: The AIResponse to explain.
        Returns:
            Human-readable explanation string.
        """
        ...


# ============================================================================
# Personality Engine Interface
# ============================================================================

class PersonalityEngineInterface(ABC):
    """Contract for the AI personality and emotional modeling system."""

    @abstractmethod
    async def get_personality_state(self) -> PersonalityState:
        """Return the current personality engine state snapshot."""
        ...

    @abstractmethod
    async def adapt_to_user(self, interaction_history: List[ConversationTurn]
                            ) -> None:
        """Update personality model based on recent interactions.

        Args:
            interaction_history: Recent turns from which to learn style.
        """
        ...

    @abstractmethod
    def apply_personality_to_response(self, raw_response: str,
                                      state: PersonalityState) -> str:
        """
        Style a raw AI response according to current personality state.

        Args:
            raw_response: Unformatted response text from the language model.
            state:        Current PersonalityState to apply.
        Returns:
            Personality-adjusted response text.
        """
        ...

    @abstractmethod
    def get_emotional_response_weight(self, context: ConversationContext,
                                      emotional_cues: Dict[str, float]) -> float:
        """
        Calculate how much emotional weight to add to the response.

        Args:
            context:       Current conversation context.
            emotional_cues: Detected emotional signals (e.g., stress score).
        Returns:
            Weight 0.0–1.0 for empathetic response modulation.
        """
        ...

    @abstractmethod
    async def reset_to_baseline(self) -> None:
        """Reset personality state to the configured baseline."""
        ...


# ============================================================================
# Safety Filter Interface
# ============================================================================

class SafetyFilterInterface(ABC):
    """Contract for AI content safety and ethical guardrails."""

    @abstractmethod
    async def assess_input(self, user_input: UserInput) -> SafetyAssessment:
        """Check user input for harmful or inappropriate content.

        Args:
            user_input: The raw user input to assess.
        Returns:
            SafetyAssessment with score and recommended action.
        """
        ...

    @abstractmethod
    async def assess_output(self, response: str,
                             context: ConversationContext) -> SafetyAssessment:
        """Check AI output before delivering it to the user.

        Args:
            response: The AI-generated response text.
            context:  Conversation context for contextual safety assessment.
        Returns:
            SafetyAssessment — block or allow.
        """
        ...

    @abstractmethod
    async def filter_response(self, response: str) -> str:
        """
        Modify a response to remove unsafe content while preserving meaning.

        Args:
            response: Response text that partially failed safety checks.
        Returns:
            Cleaned response text. Returns empty string if unrecoverable.
        """
        ...

    @abstractmethod
    def get_safety_threshold(self) -> float:
        """Return the current safety score threshold (0.0–1.0)."""
        ...

    @abstractmethod
    def set_safety_threshold(self, threshold: float) -> None:
        """Update the safety threshold. Higher = stricter.

        Args:
            threshold: New threshold in [0.0, 1.0].
        """
        ...


# ============================================================================
# Response Generator Interface
# ============================================================================

class ResponseGeneratorInterface(ABC):
    """Contract for the language model response generation backend."""

    @abstractmethod
    async def generate(self, prompt: str, context: ConversationContext,
                       max_tokens: int = AIConstants.MAX_OUTPUT_TOKENS) -> str:
        """Generate a response from the language model.

        Args:
            prompt:     Formatted prompt string.
            context:    Full conversation context.
            max_tokens: Maximum tokens in the response.
        Returns:
            Generated response text.
        """
        ...

    @abstractmethod
    async def generate_stream(self, prompt: str,
                              context: ConversationContext) -> AsyncIterator[str]:
        """Stream tokens from the language model.

        Yields:
            Successive text tokens.
        """
        ...

    @abstractmethod
    def format_prompt(self, context: ConversationContext,
                      user_input: UserInput) -> str:
        """Build a formatted prompt from context and user input.

        Args:
            context:    Conversation context with history.
            user_input: Current user input.
        Returns:
            Complete prompt string ready for the language model.
        """
        ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the underlying language model."""
        ...

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Return the token count for the given text.

        Args:
            text: Text to tokenize.
        Returns:
            Token count.
        """
        ...


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "ConversationRole", "ResponseModality", "EmotionalState",
    "ConversationStatus",
    "ConversationTurn", "ConversationContext", "UserInput", "AIResponse",
    "PersonalityState", "SafetyAssessment", "IntentResult",
    "AIAssistantInterface", "PersonalityEngineInterface",
    "SafetyFilterInterface", "ResponseGeneratorInterface",
]
