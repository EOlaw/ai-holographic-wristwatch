"""Health Knowledge module — AI Holographic Wristwatch."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KnowledgeResult:
    """Result returned from a knowledge base query."""
    query: str
    answer: str
    confidence: float
    sources: List[str]
    related_topics: List[str]


@dataclass
class NormalRange:
    """Normal physiological range for a health metric."""
    metric: str
    min_value: float
    max_value: float
    unit: str
    notes: str = ""


@dataclass
class DrugInfo:
    """Basic pharmaceutical information for a medication."""
    name: str
    generic_name: str
    uses: List[str]
    common_side_effects: List[str]
    warnings: List[str]


class HealthKnowledgeBase:
    """Provides health-related knowledge including normal ranges, drug info, and health facts."""

    NORMAL_RANGES: Dict[str, NormalRange] = {
        "heart_rate": NormalRange(
            metric="heart_rate", min_value=60.0, max_value=100.0, unit="bpm",
            notes="Resting heart rate for adults. Athletes may have 40-60 bpm."
        ),
        "spo2": NormalRange(
            metric="spo2", min_value=95.0, max_value=100.0, unit="%",
            notes="Blood oxygen saturation. Below 90% is considered a medical emergency."
        ),
        "blood_pressure_systolic": NormalRange(
            metric="blood_pressure_systolic", min_value=90.0, max_value=120.0, unit="mmHg",
            notes="Top number. 120-129 is elevated; 130+ is hypertension."
        ),
        "blood_pressure_diastolic": NormalRange(
            metric="blood_pressure_diastolic", min_value=60.0, max_value=80.0, unit="mmHg",
            notes="Bottom number. 80-89 is stage 1 hypertension; 90+ is stage 2."
        ),
        "body_temp": NormalRange(
            metric="body_temp", min_value=36.1, max_value=37.2, unit="°C",
            notes="Oral temperature. Fever is generally considered 38°C or higher."
        ),
        "steps_daily": NormalRange(
            metric="steps_daily", min_value=7000.0, max_value=10000.0, unit="steps",
            notes="10,000 steps/day is a widely cited goal; 7,000+ linked to reduced mortality."
        ),
        "sleep_hours": NormalRange(
            metric="sleep_hours", min_value=7.0, max_value=9.0, unit="hours",
            notes="Recommended for adults 18-64. Teens need 8-10 hours; older adults 7-8."
        ),
        "bmi": NormalRange(
            metric="bmi", min_value=18.5, max_value=24.9, unit="kg/m²",
            notes="Body Mass Index. 25-29.9 is overweight; 30+ is obese; <18.5 is underweight."
        ),
        "respiratory_rate": NormalRange(
            metric="respiratory_rate", min_value=12.0, max_value=20.0, unit="breaths/min",
            notes="Normal resting rate for adults. >25 may indicate respiratory distress."
        ),
    }

    DRUG_DATABASE: Dict[str, DrugInfo] = {
        "aspirin": DrugInfo(
            name="Aspirin",
            generic_name="acetylsalicylic acid",
            uses=["pain relief", "fever reduction", "anti-inflammatory", "blood thinning", "heart attack prevention"],
            common_side_effects=["stomach upset", "nausea", "heartburn", "gastrointestinal bleeding"],
            warnings=["Do not give to children under 12 (Reye's syndrome risk)", "Avoid with blood thinners",
                      "Can cause stomach ulcers with long-term use", "Avoid before surgery"]
        ),
        "ibuprofen": DrugInfo(
            name="Ibuprofen",
            generic_name="ibuprofen",
            uses=["pain relief", "fever reduction", "anti-inflammatory", "arthritis", "menstrual cramps"],
            common_side_effects=["stomach upset", "nausea", "dizziness", "headache", "fluid retention"],
            warnings=["Take with food or milk", "Avoid with kidney disease", "Increases risk of heart attack/stroke",
                      "Avoid in third trimester of pregnancy"]
        ),
        "acetaminophen": DrugInfo(
            name="Acetaminophen",
            generic_name="paracetamol",
            uses=["pain relief", "fever reduction"],
            common_side_effects=["rare at normal doses", "nausea", "stomach pain"],
            warnings=["Liver damage risk with excessive use or alcohol", "Do not exceed 4g/day",
                      "Check other medications for hidden acetaminophen content"]
        ),
        "metformin": DrugInfo(
            name="Metformin",
            generic_name="metformin hydrochloride",
            uses=["type 2 diabetes management", "insulin resistance", "PCOS"],
            common_side_effects=["nausea", "diarrhea", "stomach upset", "metallic taste"],
            warnings=["Take with food", "Avoid with severe kidney disease", "Hold before contrast dye procedures",
                      "Risk of lactic acidosis in rare cases"]
        ),
        "lisinopril": DrugInfo(
            name="Lisinopril",
            generic_name="lisinopril",
            uses=["hypertension", "heart failure", "post-heart attack recovery", "diabetic kidney protection"],
            common_side_effects=["dry cough", "dizziness", "headache", "fatigue"],
            warnings=["Do not use in pregnancy", "Monitor potassium levels", "Can cause angioedema",
                      "Avoid with potassium supplements unless directed"]
        ),
        "atorvastatin": DrugInfo(
            name="Atorvastatin",
            generic_name="atorvastatin calcium",
            uses=["high cholesterol", "cardiovascular disease prevention", "triglyceride reduction"],
            common_side_effects=["muscle pain", "joint pain", "diarrhea", "nausea"],
            warnings=["Monitor liver function", "Avoid grapefruit juice", "Report muscle pain immediately",
                      "Avoid in pregnancy and breastfeeding"]
        ),
        "omeprazole": DrugInfo(
            name="Omeprazole",
            generic_name="omeprazole",
            uses=["acid reflux (GERD)", "peptic ulcers", "H. pylori infection", "Zollinger-Ellison syndrome"],
            common_side_effects=["headache", "nausea", "diarrhea", "abdominal pain"],
            warnings=["Long-term use may reduce magnesium and B12 absorption", "Increased fracture risk with prolonged use",
                      "May mask symptoms of gastric cancer"]
        ),
        "amoxicillin": DrugInfo(
            name="Amoxicillin",
            generic_name="amoxicillin",
            uses=["bacterial infections", "ear infections", "strep throat", "urinary tract infections", "pneumonia"],
            common_side_effects=["diarrhea", "nausea", "rash", "headache"],
            warnings=["Penicillin allergy risk", "Complete full course of treatment",
                      "Can cause antibiotic-associated diarrhea", "May reduce effectiveness of oral contraceptives"]
        ),
        "cetirizine": DrugInfo(
            name="Cetirizine",
            generic_name="cetirizine hydrochloride",
            uses=["allergies", "hay fever", "hives", "allergic rhinitis"],
            common_side_effects=["drowsiness", "dry mouth", "fatigue", "headache"],
            warnings=["Avoid alcohol", "Use caution when driving", "Lower dose needed with kidney disease"]
        ),
        "sertraline": DrugInfo(
            name="Sertraline",
            generic_name="sertraline hydrochloride",
            uses=["depression", "anxiety disorders", "OCD", "PTSD", "panic disorder"],
            common_side_effects=["nausea", "diarrhea", "insomnia", "drowsiness", "sexual dysfunction"],
            warnings=["Increased suicidal thoughts in young adults (monitor closely)", "Do not stop abruptly",
                      "Avoid with MAOIs", "Serotonin syndrome risk with other serotonergic drugs"]
        ),
        "metoprolol": DrugInfo(
            name="Metoprolol",
            generic_name="metoprolol succinate/tartrate",
            uses=["hypertension", "heart failure", "angina", "heart rate control", "migraine prevention"],
            common_side_effects=["fatigue", "dizziness", "slow heart rate", "cold hands/feet"],
            warnings=["Do not stop abruptly (can trigger heart attack)", "Masks hypoglycemia symptoms in diabetics",
                      "Avoid with certain heart rhythm disorders"]
        ),
    }

    HEALTH_FACTS: Dict[str, str] = {
        "what is heart rate": (
            "Heart rate is the number of times your heart beats per minute. Normal resting heart rate for adults "
            "is 60-100 bpm. A lower rate generally indicates more efficient cardiovascular fitness."
        ),
        "what is blood pressure": (
            "Blood pressure measures the force of blood against artery walls. It is expressed as systolic/diastolic "
            "(e.g., 120/80 mmHg). Systolic is the pressure when the heart beats; diastolic is the pressure between beats."
        ),
        "what is bmi": (
            "Body Mass Index (BMI) is a weight-to-height ratio (kg/m²). Normal: 18.5-24.9; Overweight: 25-29.9; "
            "Obese: 30+; Underweight: <18.5. It is a screening tool, not a direct measure of body fat or health."
        ),
        "what is spo2": (
            "SpO2 (oxygen saturation) measures the percentage of hemoglobin carrying oxygen in your blood. "
            "Normal levels are 95-100%. Below 90% is considered a medical emergency requiring immediate attention."
        ),
        "how much water should i drink": (
            "The general recommendation is about 8 cups (2 liters) of water per day, but this varies based on "
            "activity level, climate, and body size. A useful guide is to drink enough so urine is pale yellow."
        ),
        "what causes high blood pressure": (
            "High blood pressure (hypertension) can be caused by genetics, age, obesity, lack of physical activity, "
            "high sodium diet, excessive alcohol, smoking, stress, and certain medications or conditions like kidney disease."
        ),
        "how to lower blood pressure": (
            "Lifestyle changes include reducing sodium intake, exercising regularly, maintaining a healthy weight, "
            "limiting alcohol, quitting smoking, reducing stress, and eating a balanced diet rich in fruits and vegetables. "
            "Medications may also be prescribed when lifestyle changes are insufficient."
        ),
        "what is diabetes": (
            "Diabetes is a chronic condition where the body cannot properly regulate blood sugar (glucose). "
            "Type 1 is an autoimmune condition; the pancreas produces little or no insulin. "
            "Type 2 occurs when cells become insulin resistant. Both require management to prevent complications."
        ),
        "what is cholesterol": (
            "Cholesterol is a waxy substance in the blood. HDL ('good') cholesterol removes other cholesterol from "
            "the bloodstream. LDL ('bad') cholesterol builds up in arteries. High LDL increases cardiovascular risk. "
            "Total cholesterol under 200 mg/dL is desirable."
        ),
        "how much sleep do i need": (
            "Adults (18-64) need 7-9 hours per night. Teenagers need 8-10 hours. Children need 9-11 hours. "
            "Sleep quality matters as much as quantity — consistent sleep timing and avoiding screen light before bed improve sleep."
        ),
        "what is inflammation": (
            "Inflammation is the body's natural protective response to injury or infection. Acute inflammation is short-term "
            "and beneficial. Chronic inflammation, however, is linked to diseases like heart disease, diabetes, "
            "arthritis, and cancer. Anti-inflammatory diets and lifestyle habits can help reduce chronic inflammation."
        ),
        "how to boost immune system": (
            "Key strategies include eating a nutritious diet rich in vitamins C, D, and zinc; getting regular exercise; "
            "sleeping 7-9 hours; managing stress; not smoking; limiting alcohol; and maintaining a healthy weight. "
            "Vaccines also play a critical role in immune defense."
        ),
        "what is a fever": (
            "A fever is a body temperature above 38°C (100.4°F). It is typically caused by infection and is part of "
            "the body's immune response. Most fevers resolve on their own. Seek medical care for fevers above 39.5°C, "
            "lasting more than 3 days, or accompanied by severe symptoms."
        ),
        "what is dehydration": (
            "Dehydration occurs when the body loses more fluids than it takes in. Symptoms include thirst, dark urine, "
            "dizziness, fatigue, dry mouth, and reduced urine output. Severe dehydration is a medical emergency. "
            "Rehydration with water or electrolyte solutions is the treatment."
        ),
        "what is a healthy diet": (
            "A healthy diet emphasizes whole fruits, vegetables, whole grains, lean proteins, and healthy fats. "
            "Limit processed foods, added sugars, saturated fats, and excess sodium. The Mediterranean diet is "
            "consistently ranked as one of the healthiest dietary patterns globally."
        ),
        "how many calories should i eat": (
            "Caloric needs vary by age, sex, activity level, and goals. General estimates: sedentary adult women ~1,600-2,000 kcal; "
            "sedentary adult men ~2,000-2,400 kcal. Active individuals need more. A registered dietitian can provide personalized guidance."
        ),
        "what is stress and its effects": (
            "Stress is the body's response to challenging situations. Short-term stress can be beneficial, but chronic stress "
            "raises cortisol levels, which can lead to high blood pressure, heart disease, obesity, depression, anxiety, "
            "digestive issues, and impaired immune function."
        ),
        "what are signs of a heart attack": (
            "Common signs include chest pain or pressure, pain radiating to the arm/shoulder/jaw/back, shortness of breath, "
            "nausea, lightheadedness, and cold sweats. Women may experience more subtle symptoms. "
            "Call emergency services immediately if a heart attack is suspected."
        ),
        "what is mental health": (
            "Mental health encompasses emotional, psychological, and social wellbeing. It affects how we think, feel, and act. "
            "Good mental health is not just the absence of mental illness but the presence of positive wellbeing. "
            "Common disorders include anxiety, depression, bipolar disorder, and schizophrenia."
        ),
        "how to improve mental health": (
            "Strategies include regular physical exercise, maintaining social connections, getting adequate sleep, "
            "practicing mindfulness or meditation, limiting alcohol and drugs, seeking professional therapy when needed, "
            "setting realistic goals, and engaging in meaningful activities."
        ),
        "what is exercise and its benefits": (
            "Exercise is physical activity that improves health and fitness. Benefits include reduced risk of heart disease, "
            "type 2 diabetes, obesity, and some cancers; improved mood and mental health; stronger bones and muscles; "
            "better sleep; and increased longevity. WHO recommends 150-300 minutes of moderate activity per week."
        ),
        "what is hypertension": (
            "Hypertension (high blood pressure) is a systolic reading ≥130 mmHg or diastolic ≥80 mmHg. "
            "It is known as the 'silent killer' because it often has no symptoms but significantly increases the risk "
            "of heart attack, stroke, and kidney disease. Lifestyle changes and medications can effectively manage it."
        ),
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._query_count: int = 0
        logger.info("HealthKnowledgeBase initialized")

    def query(self, question: str) -> KnowledgeResult:
        """Search for a health answer based on the question using keyword matching."""
        with self._lock:
            self._query_count += 1

        question_lower = question.lower().strip()

        # Try direct match first
        if question_lower in self.HEALTH_FACTS:
            answer = self.HEALTH_FACTS[question_lower]
            return KnowledgeResult(
                query=question,
                answer=answer,
                confidence=0.95,
                sources=["WHO Guidelines", "NIH Health Information", "Mayo Clinic"],
                related_topics=self._get_related_health_topics(question_lower),
            )

        # Keyword-based fuzzy search
        best_match: Optional[str] = None
        best_score: float = 0.0
        q_words = set(question_lower.split())

        for key, answer in self.HEALTH_FACTS.items():
            key_words = set(key.split())
            overlap = len(q_words & key_words)
            score = overlap / max(len(q_words), len(key_words), 1)
            if score > best_score:
                best_score = score
                best_match = key

        if best_match and best_score >= 0.3:
            return KnowledgeResult(
                query=question,
                answer=self.HEALTH_FACTS[best_match],
                confidence=min(0.9, best_score * 1.2),
                sources=["WHO Guidelines", "NIH Health Information"],
                related_topics=self._get_related_health_topics(best_match),
            )

        # Check drug database
        for drug_name, drug_info in self.DRUG_DATABASE.items():
            if drug_name in question_lower or drug_info.name.lower() in question_lower:
                answer = (
                    f"{drug_info.name} ({drug_info.generic_name}): Used for {', '.join(drug_info.uses[:3])}. "
                    f"Common side effects: {', '.join(drug_info.common_side_effects[:3])}."
                )
                return KnowledgeResult(
                    query=question,
                    answer=answer,
                    confidence=0.85,
                    sources=["FDA Drug Database", "RxList"],
                    related_topics=["medications", "drug interactions", "pharmacology"],
                )

        return KnowledgeResult(
            query=question,
            answer=(
                "I don't have specific information on that health topic. "
                "Please consult a qualified healthcare professional for medical advice."
            ),
            confidence=0.1,
            sources=[],
            related_topics=["consult a doctor", "medical professional"],
        )

    def _get_related_health_topics(self, topic: str) -> List[str]:
        """Return related health topics for a given topic key."""
        topic_map: Dict[str, List[str]] = {
            "heart rate": ["blood pressure", "cardiovascular health", "exercise", "arrhythmia"],
            "blood pressure": ["heart rate", "hypertension", "sodium intake", "cardiovascular risk"],
            "bmi": ["weight management", "obesity", "nutrition", "body composition"],
            "spo2": ["respiratory health", "pulse oximetry", "oxygen therapy", "lung health"],
            "water": ["hydration", "electrolytes", "kidney health"],
            "diabetes": ["blood sugar", "insulin", "metformin", "diet"],
            "cholesterol": ["cardiovascular risk", "statins", "diet", "heart disease"],
            "sleep": ["sleep hygiene", "circadian rhythm", "insomnia", "mental health"],
            "stress": ["mental health", "cortisol", "meditation", "exercise"],
            "heart attack": ["cardiovascular disease", "chest pain", "CPR", "emergency care"],
            "fever": ["infection", "body temperature", "acetaminophen", "ibuprofen"],
        }
        for key, related in topic_map.items():
            if key in topic:
                return related
        return ["general health", "wellness", "preventive care"]

    def get_normal_range(self, metric: str) -> Optional[NormalRange]:
        """Return normal physiological range for the given metric."""
        return self.NORMAL_RANGES.get(metric.lower().replace(" ", "_"))

    def is_value_normal(self, metric: str, value: float) -> Tuple[bool, str]:
        """Check if a value is within the normal range for a given metric."""
        key = metric.lower().replace(" ", "_")
        normal = self.NORMAL_RANGES.get(key)
        if normal is None:
            return False, f"Unknown metric: {metric}"

        if normal.min_value <= value <= normal.max_value:
            return True, (
                f"{metric} value of {value} {normal.unit} is within normal range "
                f"({normal.min_value}-{normal.max_value} {normal.unit})."
            )

        if value < normal.min_value:
            diff = normal.min_value - value
            return False, (
                f"{metric} value of {value} {normal.unit} is LOW (normal: "
                f"{normal.min_value}-{normal.max_value} {normal.unit}). "
                f"It is {diff:.1f} {normal.unit} below the minimum."
            )

        diff = value - normal.max_value
        return False, (
            f"{metric} value of {value} {normal.unit} is HIGH (normal: "
            f"{normal.min_value}-{normal.max_value} {normal.unit}). "
            f"It is {diff:.1f} {normal.unit} above the maximum."
        )

    def explain_condition(self, condition_name: str) -> str:
        """Return a description of a medical condition."""
        condition_lower = condition_name.lower()
        conditions: Dict[str, str] = {
            "hypertension": (
                "Hypertension (high blood pressure) is a chronic condition where the force of blood against arterial walls "
                "is persistently elevated (≥130/80 mmHg). Risk factors include age, genetics, obesity, high sodium intake, "
                "inactivity, and smoking. It increases risk of heart attack, stroke, and kidney disease. "
                "Management includes lifestyle changes and antihypertensive medications."
            ),
            "diabetes": (
                "Diabetes mellitus is a group of metabolic diseases characterized by high blood sugar. Type 1 results from "
                "autoimmune destruction of insulin-producing beta cells. Type 2, the most common form, results from insulin "
                "resistance and relative insulin deficiency. Complications include cardiovascular disease, neuropathy, "
                "nephropathy, and retinopathy. Management involves blood sugar monitoring, diet, exercise, and medication."
            ),
            "asthma": (
                "Asthma is a chronic inflammatory disease of the airways causing recurring episodes of wheezing, breathlessness, "
                "chest tightness, and coughing. It is triggered by allergens, exercise, cold air, or stress. "
                "Treatment includes inhaled corticosteroids for prevention and short-acting bronchodilators for acute relief."
            ),
            "depression": (
                "Depression is a common mental health disorder characterized by persistent sadness, loss of interest, "
                "fatigue, sleep and appetite changes, and feelings of worthlessness. It is caused by a combination of "
                "genetic, biological, psychological, and social factors. Treatment includes therapy, antidepressants, "
                "lifestyle changes, and support networks."
            ),
            "anxiety": (
                "Anxiety disorders involve excessive fear or worry that interferes with daily activities. Types include "
                "generalized anxiety disorder (GAD), social anxiety, panic disorder, and phobias. "
                "Treatment includes cognitive behavioral therapy (CBT), medication, and stress management techniques."
            ),
            "arthritis": (
                "Arthritis refers to joint inflammation. Osteoarthritis involves cartilage breakdown from wear and tear. "
                "Rheumatoid arthritis is an autoimmune disease attacking joint linings. Symptoms include pain, swelling, "
                "stiffness, and reduced range of motion. Management involves pain relief, physiotherapy, and in some cases, surgery."
            ),
            "obesity": (
                "Obesity is a complex chronic disease defined by excess body fat (BMI ≥30 kg/m²). It significantly increases "
                "risk of type 2 diabetes, cardiovascular disease, sleep apnea, and certain cancers. "
                "Causes include genetic predisposition, poor diet, physical inactivity, and metabolic factors. "
                "Treatment involves diet, exercise, behavioral therapy, and sometimes medication or surgery."
            ),
        }
        for key, explanation in conditions.items():
            if key in condition_lower or condition_lower in key:
                return explanation
        return (
            f"No detailed information available for '{condition_name}'. "
            "Please consult a medical professional for condition-specific guidance."
        )

    def get_drug_info(self, medication: str) -> Optional[DrugInfo]:
        """Return drug information for a given medication name."""
        medication_lower = medication.lower().strip()
        # Direct lookup
        if medication_lower in self.DRUG_DATABASE:
            return self.DRUG_DATABASE[medication_lower]
        # Partial match
        for key, drug in self.DRUG_DATABASE.items():
            if medication_lower in key or medication_lower in drug.generic_name.lower():
                return drug
            if medication_lower in drug.name.lower():
                return drug
        return None

    def get_stats(self) -> Dict:
        """Return operational statistics for this knowledge base."""
        with self._lock:
            return {
                "query_count": self._query_count,
                "total_health_facts": len(self.HEALTH_FACTS),
                "total_drugs": len(self.DRUG_DATABASE),
                "total_normal_ranges": len(self.NORMAL_RANGES),
            }


_health_knowledge_base_instance: Optional[HealthKnowledgeBase] = None
_health_kb_lock = threading.Lock()


def get_health_knowledge_base() -> HealthKnowledgeBase:
    """Return the singleton HealthKnowledgeBase instance."""
    global _health_knowledge_base_instance
    if _health_knowledge_base_instance is None:
        with _health_kb_lock:
            if _health_knowledge_base_instance is None:
                _health_knowledge_base_instance = HealthKnowledgeBase()
    return _health_knowledge_base_instance
