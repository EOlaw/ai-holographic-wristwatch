"""Mindfulness Guidance — AI Holographic Wristwatch
Provides guided breathing exercises, meditation scripts, and mindfulness sessions."""

from __future__ import annotations

import uuid
import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class SessionType(Enum):
    BREATHING = "breathing"
    MEDITATION = "meditation"
    BODY_SCAN = "body_scan"
    VISUALIZATION = "visualization"
    QUICK_CALM = "quick_calm"
    SLEEP_PREP = "sleep_prep"


@dataclass
class BreathingExercise:
    name: str
    technique: str
    inhale_secs: int
    hold_secs: int
    exhale_secs: int
    cycles: int
    description: str


@dataclass
class MindfulnessSession:
    session_type: SessionType
    start_time: float
    duration_mins: int
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    completed: bool = False
    notes: str = ""
    end_time: Optional[float] = None


class MindfulnessGuide:
    """Guides users through mindfulness practices: breathing, meditation, body scans, and more."""

    # -----------------------------------------------------------------------
    # Breathing exercise library
    # -----------------------------------------------------------------------
    _BREATHING_EXERCISES: Dict[str, BreathingExercise] = {
        "box_breathing": BreathingExercise(
            name="Box Breathing",
            technique="4-4-4-4",
            inhale_secs=4,
            hold_secs=4,
            exhale_secs=4,
            cycles=6,
            description=(
                "Box breathing (4-4-4-4) is used by Navy SEALs and first responders. "
                "Inhale for 4 seconds, hold for 4, exhale for 4, hold empty for 4. "
                "Repeat 6 cycles. Activates the parasympathetic nervous system and reduces cortisol."
            ),
        ),
        "478_breathing": BreathingExercise(
            name="4-7-8 Breathing",
            technique="4-7-8",
            inhale_secs=4,
            hold_secs=7,
            exhale_secs=8,
            cycles=4,
            description=(
                "Developed by Dr. Andrew Weil, the 4-7-8 technique is a powerful anxiolytic. "
                "Inhale through your nose for 4 seconds, hold for 7, exhale completely through "
                "your mouth for 8 seconds. Complete 4 cycles. Best for acute anxiety and falling asleep."
            ),
        ),
        "physiological_sigh": BreathingExercise(
            name="Physiological Sigh",
            technique="double-inhale-long-exhale",
            inhale_secs=2,
            hold_secs=1,
            exhale_secs=6,
            cycles=3,
            description=(
                "The physiological sigh: take a normal inhale through the nose, then a second "
                "shorter sniff to fully inflate the lungs, then exhale slowly and completely. "
                "This rapidly deflates alveoli and is the fastest known way to reduce stress. "
                "3 cycles produce near-immediate calm."
            ),
        ),
        "diaphragmatic": BreathingExercise(
            name="Diaphragmatic Breathing",
            technique="belly-breathing",
            inhale_secs=5,
            hold_secs=2,
            exhale_secs=5,
            cycles=8,
            description=(
                "Diaphragmatic (belly) breathing: place one hand on your belly, one on your chest. "
                "Breathe in slowly through the nose for 5 seconds, letting only the belly rise. "
                "Hold for 2 seconds, then exhale through pursed lips for 5 seconds. "
                "8 cycles lower blood pressure and promote deep relaxation."
            ),
        ),
    }

    # -----------------------------------------------------------------------
    # Meditation script library
    # -----------------------------------------------------------------------
    _MEDITATION_SCRIPTS: Dict[str, str] = {
        "calm": (
            "Find a comfortable position, either sitting upright or lying down. Gently close your eyes. "
            "Begin by taking three deep, slow breaths — inhaling peace and exhaling any tension. "
            "Notice the weight of your body against the surface beneath you. "
            "With each exhale, feel yourself becoming heavier, more relaxed. "
            "Bring your attention to your breath. There is nothing to do right now except be present. "
            "If thoughts arise, acknowledge them without judgment, then gently return to your breath. "
            "You are safe. You are calm. This moment is enough. "
            "Breathe in... and breathe out... "
            "Rest here in stillness for as long as you wish."
        ),
        "focus": (
            "Sit comfortably with your back straight and hands resting on your thighs. Close your eyes. "
            "Take three deep breaths to clear mental fog — exhaling completely each time. "
            "Bring your awareness to a single point of focus: the sensation of your breath entering and leaving your nostrils. "
            "When your mind wanders — and it will — gently redirect to that single point. "
            "Each time you notice distraction and return your focus, you are strengthening your attention muscle. "
            "There is no failure here — only practice. "
            "Continue anchoring your awareness to this one point. "
            "Feel clarity emerging with each breath. You are focused, present, and capable."
        ),
        "sleep": (
            "Lie down comfortably and close your eyes. Let your body sink completely into the mattress. "
            "Begin a slow body scan: starting at your feet, consciously release any tension you find. "
            "Feel your calves relax... your thighs... your hips... your abdomen... "
            "Let your chest expand and fall naturally with each breath. "
            "Your shoulders drop away from your ears... your jaw unclenches... your forehead smooths. "
            "You are perfectly safe. The day is complete. Nothing requires your attention right now. "
            "With each exhale, drift a little deeper toward sleep. "
            "Let go of thoughts... let go of plans... let go of worries. "
            "There is only this breath... this moment... and rest."
        ),
        "gratitude": (
            "Settle into a comfortable position and gently close your eyes. "
            "Take three centering breaths, releasing tension with each exhale. "
            "Now bring to mind one person in your life who has shown you kindness. "
            "Visualize their face, their smile. Feel genuine appreciation for their presence in your life. "
            "Now think of one experience from today, however small, that brought you comfort or joy. "
            "A warm cup of coffee. A ray of sunlight. A kind word. "
            "Let gratitude fill your chest like a warm light expanding outward. "
            "Finally, acknowledge one quality in yourself that you appreciate. "
            "Rest in this feeling of abundance. Your life contains beauty. You are enough."
        ),
    }

    def __init__(self) -> None:
        self._sessions: Dict[str, MindfulnessSession] = {}
        self._lock = threading.RLock()
        self._completed_count: int = 0
        logger.info("MindfulnessGuide initialized")

    def start_session(self, session_type: SessionType, duration_mins: int) -> MindfulnessSession:
        """Begin a new mindfulness session."""
        duration_mins = max(1, duration_mins)
        session = MindfulnessSession(
            session_type=session_type,
            start_time=time.time(),
            duration_mins=duration_mins,
        )
        with self._lock:
            self._sessions[session.id] = session
        logger.info(
            "Mindfulness session started: %s, %d min [%s]",
            session_type.value, duration_mins, session.id,
        )
        return session

    def end_session(self, session_id: str) -> Optional[MindfulnessSession]:
        """Mark a session as completed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning("end_session: id not found: %s", session_id)
                return None
            session.completed = True
            session.end_time = time.time()
            self._completed_count += 1
        actual_mins = (session.end_time - session.start_time) / 60.0
        logger.info(
            "Mindfulness session ended: %s (planned=%dmin, actual=%.1fmin) [%s]",
            session.session_type.value, session.duration_mins, actual_mins, session_id,
        )
        return session

    def get_breathing_exercise(self, stress_level: int) -> BreathingExercise:
        """
        Return the most appropriate breathing exercise for the given stress level (1–10).
        Higher stress levels map to more potent calming techniques.
        """
        stress_level = max(1, min(10, stress_level))
        if stress_level >= 8:
            # Fastest stress relief for acute/high stress
            exercise = self._BREATHING_EXERCISES["physiological_sigh"]
        elif stress_level >= 5:
            # 4-7-8 for moderate-high stress and anxiety
            exercise = self._BREATHING_EXERCISES["478_breathing"]
        elif stress_level >= 3:
            # Box breathing for moderate focus + calm
            exercise = self._BREATHING_EXERCISES["box_breathing"]
        else:
            # Diaphragmatic for baseline wellness and light stress
            exercise = self._BREATHING_EXERCISES["diaphragmatic"]
        logger.debug("Breathing exercise selected for stress_level=%d: %s", stress_level, exercise.name)
        return exercise

    def guided_meditation_script(self, theme: str, duration_mins: int) -> str:
        """
        Return a guided meditation script for the given theme.
        Themes: 'calm', 'focus', 'sleep', 'gratitude'.
        duration_mins is used to add a contextual preamble.
        """
        theme_key = theme.lower().strip()
        base_script = self._MEDITATION_SCRIPTS.get(theme_key)
        if base_script is None:
            # Default to calm if theme not found
            theme_key = "calm"
            base_script = self._MEDITATION_SCRIPTS["calm"]
            logger.info("Theme '%s' not found, using 'calm'.", theme)

        preamble = (
            f"Welcome to your {duration_mins}-minute {theme_key} meditation. "
            "Find a quiet place where you will not be disturbed. "
        )
        closing = (
            "\n\nWhen you are ready, gently wiggle your fingers and toes. "
            "Take a deeper breath, and slowly open your eyes. "
            f"Your {duration_mins}-minute {theme_key} meditation is complete. Well done."
        )
        full_script = preamble + base_script + closing
        logger.info("Meditation script generated: theme=%s, duration=%dmin", theme_key, duration_mins)
        return full_script

    def get_quick_calm_steps(self) -> List[str]:
        """Return 5 immediate steps for rapid stress reduction."""
        return [
            "1. PAUSE — Stop what you're doing. Close your eyes if safe to do so.",
            "2. BREATHE — Take one physiological sigh: a double inhale through the nose, then a long, slow exhale through the mouth.",
            "3. GROUND — Name 5 things you can see, 4 you can touch, 3 you can hear.",
            "4. RELEASE — Consciously unclench your jaw, drop your shoulders, and unball your hands.",
            "5. PROCEED — Re-engage with your task from a calmer state. You have what it takes.",
        ]

    def get_active_sessions(self) -> List[MindfulnessSession]:
        """Return all sessions that have been started but not completed."""
        with self._lock:
            return [s for s in self._sessions.values() if not s.completed]

    def get_stats(self) -> Dict:
        """Return statistics about mindfulness sessions."""
        with self._lock:
            all_sessions = list(self._sessions.values())

        completed = [s for s in all_sessions if s.completed]
        active = [s for s in all_sessions if not s.completed]

        type_counts: Dict[str, int] = {}
        total_minutes = 0
        for s in completed:
            type_counts[s.session_type.value] = type_counts.get(s.session_type.value, 0) + 1
            if s.end_time:
                total_minutes += (s.end_time - s.start_time) / 60.0

        avg_minutes = total_minutes / len(completed) if completed else 0.0

        return {
            "total_sessions": len(all_sessions),
            "completed_sessions": len(completed),
            "active_sessions": len(active),
            "session_type_breakdown": type_counts,
            "total_mindfulness_minutes": round(total_minutes, 1),
            "average_session_minutes": round(avg_minutes, 1),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_mindfulness_guide: Optional[MindfulnessGuide] = None
_mg_lock = threading.Lock()


def get_mindfulness_guide() -> MindfulnessGuide:
    """Return the process-wide MindfulnessGuide singleton."""
    global _mindfulness_guide
    with _mg_lock:
        if _mindfulness_guide is None:
            _mindfulness_guide = MindfulnessGuide()
    return _mindfulness_guide


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    guide = get_mindfulness_guide()

    print("=== Mindfulness Guide Demo ===\n")

    # Quick calm
    print("Quick Calm Steps:")
    for step in guide.get_quick_calm_steps():
        print(f"  {step}")

    # Breathing exercise for different stress levels
    print("\nBreathing exercise (stress level 9 — high):")
    ex = guide.get_breathing_exercise(stress_level=9)
    print(f"  {ex.name} ({ex.technique})")
    print(f"  Inhale {ex.inhale_secs}s | Hold {ex.hold_secs}s | Exhale {ex.exhale_secs}s | {ex.cycles} cycles")
    print(f"  {ex.description[:100]}...")

    print("\nBreathing exercise (stress level 3 — light):")
    ex2 = guide.get_breathing_exercise(stress_level=3)
    print(f"  {ex2.name} ({ex2.technique})")

    # Start and end a session
    session = guide.start_session(SessionType.BREATHING, duration_mins=5)
    print(f"\nStarted session: [{session.id}] {session.session_type.value}")
    time.sleep(0.05)  # Simulate brief session
    completed = guide.end_session(session.id)
    print(f"Ended session: completed={completed.completed}")

    # Meditation script (truncated)
    print("\nSleep meditation script (first 200 chars):")
    script = guide.guided_meditation_script("sleep", duration_mins=10)
    print(f"  {script[:200]}...")

    # Session for body scan
    bs = guide.start_session(SessionType.BODY_SCAN, duration_mins=15)
    active = guide.get_active_sessions()
    print(f"\nActive sessions: {len(active)}")

    print("\n--- Stats ---")
    for k, v in guide.get_stats().items():
        print(f"  {k}: {v}")
