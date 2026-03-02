"""
AI System Package — AI Holographic Wristwatch

The AI system provides intelligence, personality, and adaptive learning:

- conversational_ai/   : Dialogue management, intent recognition, LLM integration,
                         memory systems, and multimodal input processing
- health_monitoring/   : Anomaly detection, health insights, predictive health,
                         and real-time vital signs monitoring
- knowledge_systems/   : Knowledge base, reasoning engines, and real-time information
- learning_systems/    : User modeling, adaptive learning, and reinforcement learning
- personality_engine/  : Personality model, emotional intelligence, and style adaptation

Architecture:
    The AI system is powered by Claude (claude-sonnet-4-6) via the Anthropic SDK.
    All subsystems are lazy-initialized singletons.
    Context and memory flow: SensorFusionEngine → ContextAwareness → AI System.
    The personality engine shapes all output before delivery to the user.
"""

__version__ = "1.0.0"

__all__ = [
    "conversational_ai",
    "health_monitoring",
    "knowledge_systems",
    "learning_systems",
    "personality_engine",
]
