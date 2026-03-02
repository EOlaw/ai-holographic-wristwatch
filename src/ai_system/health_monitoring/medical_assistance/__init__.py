"""Medical Assistance subpackage — AI Holographic Wristwatch."""
from src.ai_system.health_monitoring.medical_assistance.medication_reminders import MedicationReminderSystem, MedicationEntry, ScheduleType, get_medication_reminder_system
from src.ai_system.health_monitoring.medical_assistance.symptom_analysis import SymptomAnalyzer, SymptomEntry, get_symptom_analyzer
from src.ai_system.health_monitoring.medical_assistance.emergency_contacts import EmergencyContactManager, Contact, get_emergency_contact_manager
from src.ai_system.health_monitoring.medical_assistance.healthcare_scheduling import HealthcareScheduler, Appointment, get_healthcare_scheduler
__all__ = ["MedicationReminderSystem", "MedicationEntry", "ScheduleType", "get_medication_reminder_system", "SymptomAnalyzer", "SymptomEntry", "get_symptom_analyzer", "EmergencyContactManager", "Contact", "get_emergency_contact_manager", "HealthcareScheduler", "Appointment", "get_healthcare_scheduler"]
