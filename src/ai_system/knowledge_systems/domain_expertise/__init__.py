"""Domain Expertise subpackage — AI Holographic Wristwatch."""
from src.ai_system.knowledge_systems.domain_expertise.health_knowledge import HealthKnowledgeBase, KnowledgeResult, get_health_knowledge_base
from src.ai_system.knowledge_systems.domain_expertise.technical_knowledge import TechnicalKnowledgeBase, get_technical_knowledge_base
from src.ai_system.knowledge_systems.domain_expertise.general_education import GeneralEducationBase, get_general_education_base
from src.ai_system.knowledge_systems.domain_expertise.entertainment_knowledge import EntertainmentKnowledge, get_entertainment_knowledge
from src.ai_system.knowledge_systems.domain_expertise.practical_skills import PracticalSkillsGuide, get_practical_skills_guide
__all__ = ["HealthKnowledgeBase", "KnowledgeResult", "get_health_knowledge_base", "TechnicalKnowledgeBase", "get_technical_knowledge_base", "GeneralEducationBase", "get_general_education_base", "EntertainmentKnowledge", "get_entertainment_knowledge", "PracticalSkillsGuide", "get_practical_skills_guide"]
