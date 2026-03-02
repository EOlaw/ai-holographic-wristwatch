"""Symptom Analysis — AI Holographic Wristwatch
Logs reported symptoms, identifies patterns, and provides urgency recommendations."""

from __future__ import annotations

import uuid
import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class SymptomSeverity(Enum):
    MILD = 1
    MODERATE = 5
    SEVERE = 8
    CRITICAL = 10


class UrgencyLevel(Enum):
    ROUTINE = "routine"
    MONITOR = "monitor"
    SEE_DOCTOR_SOON = "see_doctor_soon"
    SEEK_URGENT_CARE = "seek_urgent_care"
    CALL_EMERGENCY = "call_emergency"


@dataclass
class SymptomEntry:
    name: str
    severity: int  # 1-10
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    body_part: str = ""
    notes: str = ""
    resolved: bool = False


@dataclass
class SymptomPattern:
    symptoms: List[str]
    frequency: str
    duration_hours: float
    urgency: UrgencyLevel
    pattern_description: str


# ---------------------------------------------------------------------------
# Symptom urgency map — 25+ common symptoms
# ---------------------------------------------------------------------------
SYMPTOM_URGENCY_MAP: Dict[str, UrgencyLevel] = {
    # Emergency
    "chest pain": UrgencyLevel.CALL_EMERGENCY,
    "chest tightness": UrgencyLevel.CALL_EMERGENCY,
    "difficulty breathing": UrgencyLevel.CALL_EMERGENCY,
    "shortness of breath": UrgencyLevel.CALL_EMERGENCY,
    "sudden severe headache": UrgencyLevel.CALL_EMERGENCY,
    "loss of consciousness": UrgencyLevel.CALL_EMERGENCY,
    "stroke symptoms": UrgencyLevel.CALL_EMERGENCY,
    "severe allergic reaction": UrgencyLevel.CALL_EMERGENCY,
    "uncontrolled bleeding": UrgencyLevel.CALL_EMERGENCY,
    "seizure": UrgencyLevel.CALL_EMERGENCY,
    # Urgent care
    "high fever": UrgencyLevel.SEEK_URGENT_CARE,
    "severe abdominal pain": UrgencyLevel.SEEK_URGENT_CARE,
    "broken bone": UrgencyLevel.SEEK_URGENT_CARE,
    "deep laceration": UrgencyLevel.SEEK_URGENT_CARE,
    "confusion": UrgencyLevel.SEEK_URGENT_CARE,
    "severe dizziness": UrgencyLevel.SEEK_URGENT_CARE,
    "vision changes": UrgencyLevel.SEEK_URGENT_CARE,
    # See doctor soon
    "persistent cough": UrgencyLevel.SEE_DOCTOR_SOON,
    "fever": UrgencyLevel.SEE_DOCTOR_SOON,
    "ear pain": UrgencyLevel.SEE_DOCTOR_SOON,
    "joint pain": UrgencyLevel.SEE_DOCTOR_SOON,
    "swollen lymph nodes": UrgencyLevel.SEE_DOCTOR_SOON,
    "rash": UrgencyLevel.SEE_DOCTOR_SOON,
    "urinary pain": UrgencyLevel.SEE_DOCTOR_SOON,
    "back pain": UrgencyLevel.SEE_DOCTOR_SOON,
    # Monitor
    "mild nausea": UrgencyLevel.MONITOR,
    "nausea": UrgencyLevel.MONITOR,
    "fatigue": UrgencyLevel.MONITOR,
    "mild headache": UrgencyLevel.MONITOR,
    "sore throat": UrgencyLevel.MONITOR,
    "runny nose": UrgencyLevel.MONITOR,
    "mild dizziness": UrgencyLevel.MONITOR,
    "muscle ache": UrgencyLevel.MONITOR,
    "bloating": UrgencyLevel.MONITOR,
    "mild anxiety": UrgencyLevel.MONITOR,
    # Routine
    "headache": UrgencyLevel.ROUTINE,
    "sneezing": UrgencyLevel.ROUTINE,
    "mild fatigue": UrgencyLevel.ROUTINE,
    "dry eyes": UrgencyLevel.ROUTINE,
    "hiccups": UrgencyLevel.ROUTINE,
    "mild indigestion": UrgencyLevel.ROUTINE,
    "tension headache": UrgencyLevel.ROUTINE,
}


class SymptomAnalyzer:
    """Records symptoms, detects patterns, and recommends appropriate care levels."""

    # Severity score thresholds for urgency escalation
    _SEVERITY_URGENCY_THRESHOLDS: Dict[UrgencyLevel, int] = {
        UrgencyLevel.CALL_EMERGENCY: 9,
        UrgencyLevel.SEEK_URGENT_CARE: 7,
        UrgencyLevel.SEE_DOCTOR_SOON: 5,
        UrgencyLevel.MONITOR: 3,
        UrgencyLevel.ROUTINE: 0,
    }

    def __init__(self) -> None:
        self._symptoms: List[SymptomEntry] = []
        self._lock = threading.RLock()
        self._log_count: int = 0
        logger.info("SymptomAnalyzer initialized")

    def log_symptom(
        self,
        name: str,
        severity: int,
        body_part: str = "",
        notes: str = "",
    ) -> SymptomEntry:
        """Log a new symptom entry. Severity must be 1–10."""
        severity = max(1, min(10, severity))
        entry = SymptomEntry(
            name=name.lower().strip(),
            severity=severity,
            body_part=body_part,
            notes=notes,
        )
        with self._lock:
            self._symptoms.append(entry)
            self._log_count += 1
        logger.info(
            "Symptom logged: %s (severity=%d, body_part=%s) [%s]",
            entry.name, entry.severity, entry.body_part or "unspecified", entry.id,
        )
        return entry

    def resolve_symptom(self, symptom_id: str) -> bool:
        """Mark a symptom as resolved."""
        with self._lock:
            for entry in self._symptoms:
                if entry.id == symptom_id:
                    entry.resolved = True
                    logger.info("Symptom resolved: %s (%s)", entry.name, symptom_id)
                    return True
        logger.warning("resolve_symptom: id not found: %s", symptom_id)
        return False

    def analyze_pattern(self, hours: int = 24) -> SymptomPattern:
        """Analyze symptoms reported in the last `hours` hours and return a pattern summary."""
        cutoff = time.time() - hours * 3600.0
        with self._lock:
            recent = [s for s in self._symptoms if s.timestamp >= cutoff and not s.resolved]

        if not recent:
            return SymptomPattern(
                symptoms=[],
                frequency="none",
                duration_hours=0.0,
                urgency=UrgencyLevel.ROUTINE,
                pattern_description="No active symptoms reported.",
            )

        symptom_names = [s.name for s in recent]
        unique_names = list(dict.fromkeys(symptom_names))  # preserve order, deduplicate

        # Determine overall urgency — take the highest urgency found
        highest_urgency = UrgencyLevel.ROUTINE
        urgency_order = [
            UrgencyLevel.CALL_EMERGENCY,
            UrgencyLevel.SEEK_URGENT_CARE,
            UrgencyLevel.SEE_DOCTOR_SOON,
            UrgencyLevel.MONITOR,
            UrgencyLevel.ROUTINE,
        ]
        for symptom in recent:
            name_urgency = SYMPTOM_URGENCY_MAP.get(symptom.name, UrgencyLevel.ROUTINE)
            severity_urgency = self._urgency_from_severity(symptom.severity)
            # Take the more urgent of the two
            candidate = self._max_urgency(name_urgency, severity_urgency, urgency_order)
            highest_urgency = self._max_urgency(highest_urgency, candidate, urgency_order)

        # Compute frequency description
        count = len(recent)
        if count == 1:
            frequency = "isolated"
        elif count <= 3:
            frequency = "occasional"
        elif count <= 7:
            frequency = "frequent"
        else:
            frequency = "persistent"

        # Duration: span from oldest to newest timestamp
        timestamps = sorted(s.timestamp for s in recent)
        duration_hours = (timestamps[-1] - timestamps[0]) / 3600.0 if len(timestamps) > 1 else 0.0

        # Build description
        avg_severity = sum(s.severity for s in recent) / len(recent)
        pattern_description = (
            f"{count} symptom(s) reported over {duration_hours:.1f}h: "
            f"{', '.join(unique_names[:5])}{'...' if len(unique_names) > 5 else ''}. "
            f"Average severity: {avg_severity:.1f}/10. Urgency: {highest_urgency.value}."
        )

        return SymptomPattern(
            symptoms=unique_names,
            frequency=frequency,
            duration_hours=duration_hours,
            urgency=highest_urgency,
            pattern_description=pattern_description,
        )

    def get_recommendations(self, pattern: SymptomPattern) -> List[str]:
        """Generate actionable recommendations based on the symptom pattern."""
        recommendations: List[str] = []

        if pattern.urgency == UrgencyLevel.CALL_EMERGENCY:
            recommendations += [
                "Call emergency services (911) immediately.",
                "Do not drive yourself — wait for emergency responders.",
                "Notify your emergency contacts now.",
                "Stay calm and follow dispatcher instructions.",
            ]
        elif pattern.urgency == UrgencyLevel.SEEK_URGENT_CARE:
            recommendations += [
                "Seek urgent care or visit an emergency room within 1–2 hours.",
                "Notify a family member or caregiver.",
                "Avoid strenuous activity.",
                "Track symptom changes closely.",
            ]
        elif pattern.urgency == UrgencyLevel.SEE_DOCTOR_SOON:
            recommendations += [
                "Schedule an appointment with your doctor within 24–48 hours.",
                "Rest and stay hydrated.",
                "Monitor symptoms for any worsening.",
                "Avoid self-medicating without guidance.",
            ]
        elif pattern.urgency == UrgencyLevel.MONITOR:
            recommendations += [
                "Monitor symptoms over the next few hours.",
                "Stay hydrated and get adequate rest.",
                "Log any changes in severity or new symptoms.",
                "Contact your doctor if symptoms worsen or persist beyond 48 hours.",
            ]
        else:  # ROUTINE
            recommendations += [
                "Symptoms appear routine. Continue normal activities.",
                "Ensure adequate rest and hydration.",
                "Log symptoms again if they persist beyond 2 days.",
            ]

        # Add specific advice for certain symptoms
        if "headache" in pattern.symptoms or "mild headache" in pattern.symptoms:
            recommendations.append("Consider reducing screen time and staying hydrated for headache relief.")
        if "fever" in pattern.symptoms or "high fever" in pattern.symptoms:
            recommendations.append("Monitor temperature every 2 hours; fever >103°F (39.4°C) warrants urgent care.")
        if "nausea" in pattern.symptoms or "mild nausea" in pattern.symptoms:
            recommendations.append("Eat small, bland meals. Avoid fatty or spicy foods.")
        if "fatigue" in pattern.symptoms or "mild fatigue" in pattern.symptoms:
            recommendations.append("Ensure 7–9 hours of sleep. Consider iron and B12 levels if persistent.")

        return recommendations

    def should_seek_medical_attention(self, symptoms: List[SymptomEntry]) -> bool:
        """Return True if any symptom warrants at least SEE_DOCTOR_SOON urgency."""
        urgency_order = [
            UrgencyLevel.CALL_EMERGENCY,
            UrgencyLevel.SEEK_URGENT_CARE,
            UrgencyLevel.SEE_DOCTOR_SOON,
            UrgencyLevel.MONITOR,
            UrgencyLevel.ROUTINE,
        ]
        medical_threshold_index = urgency_order.index(UrgencyLevel.SEE_DOCTOR_SOON)
        for symptom in symptoms:
            name_urgency = SYMPTOM_URGENCY_MAP.get(symptom.name, UrgencyLevel.ROUTINE)
            severity_urgency = self._urgency_from_severity(symptom.severity)
            effective_urgency = self._max_urgency(name_urgency, severity_urgency, urgency_order)
            if urgency_order.index(effective_urgency) <= medical_threshold_index:
                return True
        return False

    def get_active_symptoms(self) -> List[SymptomEntry]:
        """Return all unresolved symptom entries."""
        with self._lock:
            return [s for s in self._symptoms if not s.resolved]

    def get_stats(self) -> Dict:
        """Return statistics about logged symptoms."""
        with self._lock:
            active = [s for s in self._symptoms if not s.resolved]
            resolved = [s for s in self._symptoms if s.resolved]
            avg_severity = (
                sum(s.severity for s in active) / len(active) if active else 0.0
            )
            urgency_dist: Dict[str, int] = {}
            for s in active:
                u = SYMPTOM_URGENCY_MAP.get(s.name, UrgencyLevel.ROUTINE).value
                urgency_dist[u] = urgency_dist.get(u, 0) + 1
            return {
                "total_logged": len(self._symptoms),
                "active_symptoms": len(active),
                "resolved_symptoms": len(resolved),
                "average_severity": round(avg_severity, 2),
                "urgency_distribution": urgency_dist,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _urgency_from_severity(self, severity: int) -> UrgencyLevel:
        """Map a numeric severity (1–10) to an urgency level."""
        if severity >= 9:
            return UrgencyLevel.CALL_EMERGENCY
        elif severity >= 7:
            return UrgencyLevel.SEEK_URGENT_CARE
        elif severity >= 5:
            return UrgencyLevel.SEE_DOCTOR_SOON
        elif severity >= 3:
            return UrgencyLevel.MONITOR
        else:
            return UrgencyLevel.ROUTINE

    @staticmethod
    def _max_urgency(
        a: UrgencyLevel,
        b: UrgencyLevel,
        order: List[UrgencyLevel],
    ) -> UrgencyLevel:
        """Return whichever urgency level is more serious (lower index = more urgent)."""
        return a if order.index(a) <= order.index(b) else b


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_symptom_analyzer: Optional[SymptomAnalyzer] = None
_sa_lock = threading.Lock()


def get_symptom_analyzer() -> SymptomAnalyzer:
    """Return the process-wide SymptomAnalyzer singleton."""
    global _symptom_analyzer
    with _sa_lock:
        if _symptom_analyzer is None:
            _symptom_analyzer = SymptomAnalyzer()
    return _symptom_analyzer


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyzer = get_symptom_analyzer()

    print("=== Symptom Analyzer Demo ===\n")

    # Log some symptoms
    s1 = analyzer.log_symptom("headache", 4, body_part="head", notes="Mild tension headache after work")
    s2 = analyzer.log_symptom("fatigue", 5, notes="Feeling tired all day")
    s3 = analyzer.log_symptom("nausea", 3, body_part="stomach")
    s4 = analyzer.log_symptom("chest pain", 8, body_part="chest", notes="Pressure on left side")

    print(f"Active symptoms: {len(analyzer.get_active_symptoms())}")

    pattern = analyzer.analyze_pattern(hours=24)
    print(f"\nPattern: {pattern.pattern_description}")
    print(f"Urgency: {pattern.urgency.value}")
    print(f"Symptoms: {pattern.symptoms}")

    print(f"\nShould seek medical attention: {analyzer.should_seek_medical_attention(analyzer.get_active_symptoms())}")

    print("\nRecommendations:")
    for rec in analyzer.get_recommendations(pattern):
        print(f"  - {rec}")

    # Resolve a symptom
    analyzer.resolve_symptom(s3.id)
    print(f"\nAfter resolving nausea — active symptoms: {len(analyzer.get_active_symptoms())}")

    print("\n--- Stats ---")
    for k, v in analyzer.get_stats().items():
        print(f"  {k}: {v}")
