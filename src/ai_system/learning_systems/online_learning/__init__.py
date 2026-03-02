"""Online Learning subpackage — AI Holographic Wristwatch."""
from src.ai_system.learning_systems.online_learning.incremental_learning import IncrementalLearner, LearningResult, get_incremental_learner
from src.ai_system.learning_systems.online_learning.knowledge_expansion import KnowledgeExpander, KnowledgeItem, get_knowledge_expander
from src.ai_system.learning_systems.online_learning.pattern_discovery import PatternDiscovery, Pattern, get_pattern_discovery
from src.ai_system.learning_systems.online_learning.user_feedback_integration import UserFeedbackIntegrator, Feedback, get_user_feedback_integrator
__all__ = [
    "IncrementalLearner", "LearningResult", "get_incremental_learner",
    "KnowledgeExpander", "KnowledgeItem", "get_knowledge_expander",
    "PatternDiscovery", "Pattern", "get_pattern_discovery",
    "UserFeedbackIntegrator", "Feedback", "get_user_feedback_integrator",
]
