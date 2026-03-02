# src/core/interfaces/__init__.py
"""
Core Interface Contracts Package for AI Holographic Wristwatch System

This package exports every abstract interface, protocol, and data container
that defines the contracts between the system's layers. Import from this
package to get clean, layer-agnostic types without pulling in hardware or
AI-specific dependencies.

Usage:
    from src.core.interfaces import SensorInterface, HolographicDisplayInterface
    from src.core.interfaces import AIAssistantInterface, KnowledgeBaseInterface
    from src.core.interfaces import TaskExecutorInterface
"""

# ── Sensor interfaces ────────────────────────────────────────────────────────
from .sensor_interface import (
    SensorType,
    SensorReadingQuality,
    SensorInfo,
    SensorReading,
    CalibrationResult,
    FusedSensorData,
    SensorHealthReport,
    SensorInterface,
    SensorFusionInterface,
    SensorRegistryInterface,
    SensorEventHandlerInterface,
    CalibrationStrategyInterface,
)

# ── Holographic display interfaces ───────────────────────────────────────────
from .display_interface import (
    DisplayMode,
    RenderQuality,
    InteractionType,
    ColorSpace,
    DisplayStatus,
    HologramData,
    RenderResult,
    CalibrationResult as DisplayCalibrationResult,
    UIElement,
    InteractionEvent,
    ProjectionSpec,
    HolographicDisplayInterface,
    InteractionTrackingInterface,
    DisplayCalibrationInterface,
    WatchFaceDisplayInterface,
)

# ── AI assistant interfaces ───────────────────────────────────────────────────
from .ai_assistant_interface import (
    ConversationRole,
    ResponseModality,
    EmotionalState,
    ConversationStatus,
    ConversationTurn,
    ConversationContext,
    UserInput,
    AIResponse,
    PersonalityState,
    SafetyAssessment,
    IntentResult,
    AIAssistantInterface,
    PersonalityEngineInterface,
    SafetyFilterInterface,
    ResponseGeneratorInterface,
)

# ── Knowledge system interfaces ───────────────────────────────────────────────
from .knowledge_interface import (
    KnowledgeType,
    KnowledgeConfidence,
    RetrievalStrategy,
    Fact,
    QueryContext,
    KnowledgeResult,
    KnowledgeStats,
    KnowledgeUpdateResult,
    KnowledgeGraphNode,
    KnowledgeGraphEdge,
    KnowledgeBaseInterface,
    PersonalKnowledgeInterface,
    RealTimeInformationInterface,
    KnowledgeGraphInterface,
)

# ── Task automation interfaces ────────────────────────────────────────────────
from .task_interface import (
    TaskStatus,
    TaskPriority,
    TriggerType,
    TaskCategory,
    TaskTrigger,
    Task,
    TaskResult,
    WorkflowStep,
    Workflow,
    WorkflowResult,
    ScheduledTask,
    TaskExecutorInterface,
    TaskSchedulerInterface,
    WorkflowOrchestratorInterface,
    SmartHomeControlInterface,
)

__version__ = "1.0.0"
__all__ = [
    # Sensor
    "SensorType", "SensorReadingQuality", "SensorInfo", "SensorReading",
    "CalibrationResult", "FusedSensorData", "SensorHealthReport",
    "SensorInterface", "SensorFusionInterface", "SensorRegistryInterface",
    "SensorEventHandlerInterface", "CalibrationStrategyInterface",

    # Display
    "DisplayMode", "RenderQuality", "InteractionType", "ColorSpace",
    "DisplayStatus", "HologramData", "RenderResult", "DisplayCalibrationResult",
    "UIElement", "InteractionEvent", "ProjectionSpec",
    "HolographicDisplayInterface", "InteractionTrackingInterface",
    "DisplayCalibrationInterface", "WatchFaceDisplayInterface",

    # AI Assistant
    "ConversationRole", "ResponseModality", "EmotionalState", "ConversationStatus",
    "ConversationTurn", "ConversationContext", "UserInput", "AIResponse",
    "PersonalityState", "SafetyAssessment", "IntentResult",
    "AIAssistantInterface", "PersonalityEngineInterface",
    "SafetyFilterInterface", "ResponseGeneratorInterface",

    # Knowledge
    "KnowledgeType", "KnowledgeConfidence", "RetrievalStrategy",
    "Fact", "QueryContext", "KnowledgeResult", "KnowledgeStats",
    "KnowledgeUpdateResult", "KnowledgeGraphNode", "KnowledgeGraphEdge",
    "KnowledgeBaseInterface", "PersonalKnowledgeInterface",
    "RealTimeInformationInterface", "KnowledgeGraphInterface",

    # Task
    "TaskStatus", "TaskPriority", "TriggerType", "TaskCategory",
    "TaskTrigger", "Task", "TaskResult",
    "WorkflowStep", "Workflow", "WorkflowResult", "ScheduledTask",
    "TaskExecutorInterface", "TaskSchedulerInterface",
    "WorkflowOrchestratorInterface", "SmartHomeControlInterface",
]
