"""Mental Health Support — AI Holographic Wristwatch
Tracks mood, assesses wellbeing, and provides mental health resources and coping strategies."""

from __future__ import annotations

import uuid
import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class WellbeingDimension(Enum):
    MOOD = "mood"
    ENERGY = "energy"
    ANXIETY = "anxiety"
    FOCUS = "focus"
    SOCIAL = "social"
    SLEEP = "sleep"


@dataclass
class WellbeingScore:
    overall: float          # 0–100
    mood_score: float       # 0–100
    stress_score: float     # 0–100 (higher = more stress)
    sleep_score: float      # 0–100
    energy_score: float     # 0–100
    trend: str              # "improving", "declining", "stable"
    timestamp: float = field(default_factory=time.time)


@dataclass
class MoodEntry:
    mood: int       # 1–10
    energy: int     # 1–10
    anxiety: int    # 1–10
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    notes: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class Resource:
    title: str
    type: str          # "hotline", "app", "exercise", "article", "therapy"
    description: str
    url: str = ""
    contact: str = ""


class MentalHealthSupport:
    """Tracks mood history, assesses mental wellbeing, and provides evidence-based resources."""

    # -----------------------------------------------------------------------
    # Static resource library
    # -----------------------------------------------------------------------
    _RESOURCES: Dict[str, List[Resource]] = {
        "anxiety": [
            Resource(
                title="National Anxiety Hotline",
                type="hotline",
                description="24/7 support for anxiety and panic attacks.",
                contact="1-800-950-NAMI (6264)",
            ),
            Resource(
                title="Calm App",
                type="app",
                description="Guided meditation and breathing exercises for anxiety relief.",
                url="https://www.calm.com",
            ),
            Resource(
                title="4-7-8 Breathing Exercise",
                type="exercise",
                description="Inhale for 4 seconds, hold for 7, exhale for 8. Repeat 4 cycles.",
            ),
            Resource(
                title="Progressive Muscle Relaxation",
                type="exercise",
                description="Systematically tense and release muscle groups to reduce physical anxiety tension.",
            ),
            Resource(
                title="Cognitive Behavioral Therapy (CBT) Worksheets",
                type="article",
                description="Free CBT worksheets to challenge anxious thought patterns.",
                url="https://www.therapistaid.com/therapy-worksheets/cbt",
            ),
        ],
        "depression": [
            Resource(
                title="National Suicide Prevention Lifeline",
                type="hotline",
                description="24/7 crisis support. Call or text 988.",
                contact="988",
            ),
            Resource(
                title="Crisis Text Line",
                type="hotline",
                description="Text HOME to 741741 to reach a trained crisis counselor.",
                contact="741741",
            ),
            Resource(
                title="Behavioral Activation",
                type="exercise",
                description="Schedule small, pleasurable activities to improve mood and motivation.",
            ),
            Resource(
                title="Headspace App",
                type="app",
                description="Guided meditations specifically designed for depression and low mood.",
                url="https://www.headspace.com",
            ),
            Resource(
                title="NAMI (National Alliance on Mental Illness)",
                type="article",
                description="Information, support, and treatment resources for depression.",
                url="https://www.nami.org",
            ),
        ],
        "stress": [
            Resource(
                title="Box Breathing Technique",
                type="exercise",
                description="Inhale 4s → Hold 4s → Exhale 4s → Hold 4s. Repeat to reduce cortisol.",
            ),
            Resource(
                title="SAMHSA Helpline",
                type="hotline",
                description="Free, confidential mental health and substance use referrals.",
                contact="1-800-662-HELP (4357)",
            ),
            Resource(
                title="Stress Journal",
                type="exercise",
                description="Write 3 stressors and 1 coping strategy for each. 5-minute daily exercise.",
            ),
            Resource(
                title="Time Management: Pomodoro Technique",
                type="exercise",
                description="Work in 25-minute focused bursts with 5-minute breaks to reduce overwhelm.",
            ),
            Resource(
                title="Mindful Walking",
                type="exercise",
                description="A 10-minute walk while focusing on your breath and surroundings to reset stress response.",
            ),
        ],
        "burnout": [
            Resource(
                title="Burnout Recovery Guide",
                type="article",
                description="Evidence-based strategies to recover from workplace and caregiver burnout.",
                url="https://www.psychologytoday.com/burnout",
            ),
            Resource(
                title="Set Boundaries Workshop",
                type="exercise",
                description="Identify energy drains and practice saying 'no' to non-essential tasks.",
            ),
            Resource(
                title="Digital Detox Plan",
                type="exercise",
                description="Scheduled screen-free periods (1h morning, 2h evening) to restore mental energy.",
            ),
            Resource(
                title="Employee Assistance Program (EAP)",
                type="therapy",
                description="Ask your employer about free, confidential counseling sessions via EAP.",
            ),
        ],
        "general": [
            Resource(
                title="MindShift CBT App",
                type="app",
                description="Evidence-based tools to manage anxiety and mood.",
                url="https://www.anxietycanada.com/resources/mindshift-cbt/",
            ),
            Resource(
                title="Psychology Today Therapist Finder",
                type="therapy",
                description="Find licensed therapists near you by specialty and insurance.",
                url="https://www.psychologytoday.com/us/therapists",
            ),
            Resource(
                title="Daily Gratitude Journaling",
                type="exercise",
                description="Write 3 things you are grateful for each morning to improve baseline mood.",
            ),
        ],
    }

    # -----------------------------------------------------------------------
    # Coping strategies by stress level (1–10)
    # -----------------------------------------------------------------------
    _COPING_STRATEGIES: Dict[int, str] = {
        1: "Your stress level is minimal. Maintain your routine and continue activities that bring you joy.",
        2: "Light stress detected. Take a 5-minute break, stretch, and drink a glass of water.",
        3: "Mild stress. Try a brief mindfulness check-in: close your eyes, take 3 deep breaths, and name 5 things you can see.",
        4: "Moderate-low stress. Go for a 10-minute walk outdoors or do a quick body scan meditation.",
        5: "Moderate stress. Practice box breathing (4-4-4-4) for 5 minutes. Consider a short break from screens.",
        6: "Noticeable stress. Journal your thoughts for 5 minutes — externalizing worries reduces their power.",
        7: "Elevated stress. Use the 4-7-8 breathing technique. Reach out to a friend or family member for support.",
        8: "High stress. Step away from your current task. Progressive muscle relaxation (10 min) can significantly lower cortisol.",
        9: "Very high stress. Prioritize rest tonight. Consider calling a support line or speaking to a counselor.",
        10: "Critical stress. Seek immediate support from a trusted person, therapist, or crisis line (988).",
    }

    def __init__(self) -> None:
        self._mood_history: Deque[MoodEntry] = deque(maxlen=30)
        self._lock = threading.RLock()
        self._entry_count: int = 0
        logger.info("MentalHealthSupport initialized")

    def track_mood_entry(
        self,
        mood: int,
        energy: int,
        anxiety: int,
        notes: str = "",
    ) -> MoodEntry:
        """Record a mood check-in. All values should be 1–10."""
        mood = max(1, min(10, mood))
        energy = max(1, min(10, energy))
        anxiety = max(1, min(10, anxiety))
        entry = MoodEntry(mood=mood, energy=energy, anxiety=anxiety, notes=notes)
        with self._lock:
            self._mood_history.append(entry)
            self._entry_count += 1
        logger.info(
            "Mood entry: mood=%d, energy=%d, anxiety=%d [%s]",
            mood, energy, anxiety, entry.id,
        )
        return entry

    def assess_mental_wellbeing(self) -> WellbeingScore:
        """Compute a WellbeingScore from recent mood history."""
        with self._lock:
            history = list(self._mood_history)

        if not history:
            return WellbeingScore(
                overall=50.0,
                mood_score=50.0,
                stress_score=50.0,
                sleep_score=50.0,
                energy_score=50.0,
                trend="stable",
            )

        recent = history[-7:] if len(history) >= 7 else history
        avg_mood = sum(e.mood for e in recent) / len(recent)
        avg_energy = sum(e.energy for e in recent) / len(recent)
        avg_anxiety = sum(e.anxiety for e in recent) / len(recent)

        # Normalize to 0–100 scale
        mood_score = (avg_mood / 10.0) * 100.0
        energy_score = (avg_energy / 10.0) * 100.0
        # Stress is inversely related to mood and directly to anxiety
        stress_score = ((avg_anxiety / 10.0) * 0.7 + (1 - avg_mood / 10.0) * 0.3) * 100.0
        # Sleep score estimated from energy (proxy) when no direct sleep data
        sleep_score = (avg_energy / 10.0) * 100.0

        overall = (mood_score * 0.35 + energy_score * 0.25 + (100 - stress_score) * 0.25 + sleep_score * 0.15)

        # Trend: compare first half vs second half of recent window
        trend = self.get_mood_trend(days=7)

        logger.debug(
            "Wellbeing assessment: overall=%.1f, mood=%.1f, stress=%.1f, energy=%.1f",
            overall, mood_score, stress_score, energy_score,
        )
        return WellbeingScore(
            overall=round(overall, 1),
            mood_score=round(mood_score, 1),
            stress_score=round(stress_score, 1),
            sleep_score=round(sleep_score, 1),
            energy_score=round(energy_score, 1),
            trend=trend,
        )

    def get_support_resources(self, concern_type: str) -> List[Resource]:
        """Return a list of resources for the given concern type."""
        key = concern_type.lower().strip()
        resources = self._RESOURCES.get(key, [])
        if not resources:
            # Fall back to general resources
            resources = self._RESOURCES.get("general", [])
            logger.info("No resources found for '%s', returning general resources.", concern_type)
        else:
            logger.info("Returning %d resources for concern: %s", len(resources), concern_type)
        return resources

    def generate_coping_strategy(self, stress_level: int, context: str = "") -> str:
        """Generate a coping strategy appropriate for the given stress level (1–10)."""
        stress_level = max(1, min(10, stress_level))
        strategy = self._COPING_STRATEGIES.get(stress_level, self._COPING_STRATEGIES[5])
        if context:
            strategy = f"[Context: {context}] {strategy}"
        logger.info("Coping strategy generated for stress level %d", stress_level)
        return strategy

    def get_mood_trend(self, days: int = 7) -> str:
        """
        Return 'improving', 'declining', or 'stable' based on mood history.
        Compares the average mood of the first half vs second half of the window.
        """
        cutoff = time.time() - days * 86400.0
        with self._lock:
            recent = [e for e in self._mood_history if e.timestamp >= cutoff]

        if len(recent) < 2:
            return "stable"

        half = len(recent) // 2
        first_half = recent[:half]
        second_half = recent[half:]

        avg_first = sum(e.mood for e in first_half) / len(first_half)
        avg_second = sum(e.mood for e in second_half) / len(second_half)

        delta = avg_second - avg_first
        if delta >= 0.5:
            return "improving"
        elif delta <= -0.5:
            return "declining"
        else:
            return "stable"

    def get_stats(self) -> Dict:
        """Return statistics about mental health tracking."""
        with self._lock:
            history = list(self._mood_history)
        if not history:
            return {
                "total_entries": 0,
                "entries_in_buffer": 0,
                "average_mood": None,
                "average_energy": None,
                "average_anxiety": None,
                "trend": "stable",
            }
        avg_mood = sum(e.mood for e in history) / len(history)
        avg_energy = sum(e.energy for e in history) / len(history)
        avg_anxiety = sum(e.anxiety for e in history) / len(history)
        return {
            "total_entries": self._entry_count,
            "entries_in_buffer": len(history),
            "average_mood": round(avg_mood, 2),
            "average_energy": round(avg_energy, 2),
            "average_anxiety": round(avg_anxiety, 2),
            "trend": self.get_mood_trend(),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_mental_health_support: Optional[MentalHealthSupport] = None
_mhs_lock = threading.Lock()


def get_mental_health_support() -> MentalHealthSupport:
    """Return the process-wide MentalHealthSupport singleton."""
    global _mental_health_support
    with _mhs_lock:
        if _mental_health_support is None:
            _mental_health_support = MentalHealthSupport()
    return _mental_health_support


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    support = get_mental_health_support()

    print("=== Mental Health Support Demo ===\n")

    # Simulate a week of mood entries
    import random
    random.seed(42)
    week_data = [
        (5, 4, 6),  # Mon
        (6, 5, 5),  # Tue
        (4, 3, 7),  # Wed
        (5, 4, 6),  # Thu
        (7, 6, 4),  # Fri
        (8, 7, 3),  # Sat
        (7, 7, 3),  # Sun
    ]
    for mood, energy, anxiety in week_data:
        support.track_mood_entry(mood, energy, anxiety)

    wellbeing = support.assess_mental_wellbeing()
    print(f"Overall wellbeing score: {wellbeing.overall}/100")
    print(f"  Mood score:   {wellbeing.mood_score}")
    print(f"  Stress score: {wellbeing.stress_score}")
    print(f"  Energy score: {wellbeing.energy_score}")
    print(f"  Sleep score:  {wellbeing.sleep_score}")
    print(f"  Trend:        {wellbeing.trend}")

    print("\nCoping strategy for stress level 7:")
    print(f"  {support.generate_coping_strategy(7, context='Work deadline pressure')}")

    print("\nAnxiety resources:")
    for res in support.get_support_resources("anxiety")[:3]:
        print(f"  [{res.type}] {res.title}")
        if res.contact:
            print(f"    Contact: {res.contact}")
        if res.url:
            print(f"    URL: {res.url}")

    print("\nMood trend (7 days):", support.get_mood_trend(days=7))

    print("\n--- Stats ---")
    for k, v in support.get_stats().items():
        print(f"  {k}: {v}")
