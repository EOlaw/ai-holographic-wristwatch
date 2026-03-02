"""Wellness Coaching subpackage — AI Holographic Wristwatch."""
from src.ai_system.health_monitoring.wellness_coaching.fitness_guidance import FitnessGuide, WorkoutPlan, get_fitness_guide
from src.ai_system.health_monitoring.wellness_coaching.nutrition_advice import NutritionAdvisor, NutritionSummary, get_nutrition_advisor
from src.ai_system.health_monitoring.wellness_coaching.sleep_optimization import SleepOptimizer, SleepEntry, get_sleep_optimizer
from src.ai_system.health_monitoring.wellness_coaching.stress_management import StressManager, StressEvent, get_stress_manager
__all__ = ["FitnessGuide", "WorkoutPlan", "get_fitness_guide", "NutritionAdvisor", "NutritionSummary", "get_nutrition_advisor", "SleepOptimizer", "SleepEntry", "get_sleep_optimizer", "StressManager", "StressEvent", "get_stress_manager"]
