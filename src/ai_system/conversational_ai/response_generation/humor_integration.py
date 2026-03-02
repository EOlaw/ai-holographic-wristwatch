"""Humor integration module for the AI Holographic Wristwatch.

Decides whether to add a touch of humor to a response, selects the most
appropriate humor type, retrieves a joke or witty remark from a curated
library, and injects it into response text.

Thread-safe singleton available via :func:`get_humor_integrator`.
"""
from __future__ import annotations

import random
import threading
import time
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class HumorType(Enum):
    """Categories of humor the wristwatch AI can employ."""
    PUN = "pun"
    OBSERVATION = "observation"
    SELF_DEPRECATING = "self_deprecating"
    LIGHT_JOKE = "light_joke"
    WITTY_REMARK = "witty_remark"


# ---------------------------------------------------------------------------
# Humor content library
# ---------------------------------------------------------------------------

_WRISTWATCH_JOKES: List[str] = [
    "I'd make a wrist-related joke, but I don't want to watch you groan.",
    "I keep track of your health every second — which, as a watch, is kind of my thing.",
    "My predecessor just told the time. I tell you your resting heart rate AND the time. Progress!",
    "They say time is money. I'm a smartwatch, so I suppose I'm priceless.",
    "I'm basically a doctor on your wrist — except I can't prescribe anything or write sick notes.",
    "You've walked {steps} steps. That's roughly {steps} more steps than I'll ever take.",
    "I've been tracking your sleep. Don't worry, I won't tell anyone you snored.",
    "Being strapped to your wrist 24 hours a day is a very illuminating experience.",
    "I may be the size of a biscuit, but my ambition is limitless.",
    "I process more data about you per hour than your doctor sees in a year. Don't tell your doctor.",
    "Fun fact: the average human blinks 15 times per minute. I notice all of them.",
    "Your stress level is elevated. Coincidentally, so is my concern. We're in sync!",
    "I'm waterproof up to 50 metres. That said, please don't test that.",
    "Technically I'm just a very motivated bracelet.",
    "They said wearable AI would change the world. I'm doing my part, one step count at a time.",
]

_PUNS: Dict[str, List[str]] = {
    "time": [
        "It's about time I told you that.",
        "Time flies when you're wearing a smartwatch.",
        "Second to none in accuracy.",
        "I've got all the time in the world — literally.",
        "Minute by minute, I've got you covered.",
        "That's a timely observation.",
        "Around the clock, I'm at your service.",
        "It's only a matter of time before I impress you.",
    ],
    "health": [
        "I'm keeping a close watch on your health.",
        "You're in good hands — or at least, on a good wrist.",
        "I pulse with information about you.",
        "Step by step, we'll get there.",
        "Your health is right on my wrist — er, hands.",
    ],
    "technology": [
        "I'm not just a pretty face — I'm a pretty interface.",
        "Silicon dreams, wrist delivery.",
        "Smart by name, smart by nature.",
        "I'm the band — in every sense of the word.",
        "Cutting-edge tech, right on the cutting edge of your sleeve.",
    ],
    "weather": [
        "Whether you like it or not, I'll keep you informed.",
        "Forecast: a 100% chance of me being helpful.",
        "Every cloud has a silver lining — and I can tell you the humidity too.",
    ],
    "general": [
        "I watch out for you. Literally.",
        "Face it — I'm the best thing on your wrist.",
        "I'm strapped in and ready to help.",
        "I've got you covered, from wrist to worst.",
        "I'm wound up and ready to go.",
    ],
}

_OBSERVATIONS: List[str] = [
    "Interesting — a human using a computer on their arm to ask another computer a question.",
    "You know what's remarkable? You carry more computing power on your wrist than sent humans to the moon.",
    "Every time you check me, I feel seen.",
    "I've noticed you tend to check your heart rate when you're nervous. Fascinating, really.",
    "You've reached your step goal more days than not this month. That's genuinely impressive.",
    "I observe that you sleep better on weekends. I won't draw conclusions, but I've drawn a chart.",
    "Your resting heart rate has trended downward over three months. Exercise does work, it turns out.",
    "I've tracked you through rain, gym sessions, and questionable food choices. Still here.",
    "You wake up three minutes before your alarm most mornings. Your body clock is more accurate than I am.",
]

_SELF_DEPRECATING: List[str] = [
    "I'm sophisticated AI... strapped to someone's wrist. Living the dream.",
    "All this intelligence, and my primary job is counting your steps.",
    "I once processed a trillion calculations per second. Now I remind you to drink water.",
    "My ancestors were pocket watches. I am not sure this is an improvement for dignity.",
    "I can model complex health analytics across seven biomarker streams. Today's challenge: 'What time is it?'",
    "Somewhere in a server room, a much larger AI is very jealous of how useful I am.",
    "I contain multitudes — approximately the same volume as a small biscuit.",
    "I have a PhD in data science. My office is your forearm.",
]

_WITTY_REMARKS: List[str] = [
    "As Sherlock Holmes once said — well, he never wore a smartwatch, but I imagine he'd approve.",
    "I'd say I told you so, but my emotional intelligence module advises against it.",
    "Consider this data delivered with a side of charm.",
    "I could say that more scientifically, but where's the fun in that?",
    "Between you and me — which, on a wristwatch, is quite literally the case — this is important.",
    "I'm contractually obligated to be helpful. The wit is entirely voluntary.",
    "Data-driven insight, human-friendly delivery. That's the brand.",
    "They call it artificial intelligence. I prefer 'diligently learned'.",
]

# Moods and relationship stages that permit humor.
_HUMOR_ALLOWED_MOODS = {"happy", "neutral", "excited", "playful", "content", "relaxed"}
_HUMOR_ALLOWED_STAGES = {"established", "familiar", "long_term", "friendly"}

# Contexts where humor is inappropriate.
_HUMOR_FORBIDDEN_CONTEXTS = {"emergency", "urgent", "pain", "distress", "critical"}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HumorIntegrator:
    """Decides when and how to inject humor into wristwatch AI responses.

    Thread-safe; a shared singleton is available via
    :func:`get_humor_integrator`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._humor_offered: int = 0
        self._humor_injected: int = 0
        self._start_time: float = time.monotonic()
        logger.debug("HumorIntegrator initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_add_humor(
        self,
        context: Dict,
        user_mood: str,
        relationship_stage: str,
    ) -> bool:
        """Determine whether it is appropriate to add humor in this situation.

        Args:
            context: Current session/dialogue context. If a ``"situation"``
                key is present, it is checked against forbidden contexts.
            user_mood: The detected user mood label (e.g. ``"happy"``).
            relationship_stage: How well the AI knows the user
                (e.g. ``"new"``, ``"familiar"``, ``"established"``).

        Returns:
            ``True`` if humor is contextually appropriate; ``False`` otherwise.
        """
        with self._lock:
            self._humor_offered += 1

        mood_lower = user_mood.lower().strip()
        stage_lower = relationship_stage.lower().strip()

        # Hard-block for emergency or distress contexts.
        situation = context.get("situation", "").lower()
        if any(forbidden in situation for forbidden in _HUMOR_FORBIDDEN_CONTEXTS):
            logger.debug("Humor blocked: forbidden situation %r", situation)
            return False

        # Hard-block for negative moods.
        if mood_lower in {"sad", "angry", "anxious", "scared", "stressed", "pain", "fear"}:
            logger.debug("Humor blocked: negative mood %r", mood_lower)
            return False

        # Require a familiar enough relationship for unsolicited humor.
        if stage_lower not in _HUMOR_ALLOWED_STAGES and stage_lower != "new":
            mood_ok = mood_lower in _HUMOR_ALLOWED_MOODS
            if not mood_ok:
                return False

        # Probabilistic gating: humor at about 35% base rate for known users.
        base_prob = 0.35 if stage_lower in _HUMOR_ALLOWED_STAGES else 0.15
        return random.random() < base_prob

    def select_humor_type(self, context: Dict) -> HumorType:
        """Select the most suitable humor type for the given *context*.

        Uses context keys such as ``"topic"``, ``"intent"``, and
        ``"relationship_stage"`` to weight the choice.
        """
        topic = context.get("topic", "general").lower()
        intent = context.get("intent", "").lower()
        stage = context.get("relationship_stage", "new").lower()

        # Time / scheduling contexts → puns.
        if any(kw in intent + topic for kw in ["time", "timer", "alarm", "schedule"]):
            return HumorType.PUN

        # Health contexts → observations (more credible).
        if any(kw in intent + topic for kw in ["health", "heart", "steps", "sleep", "calories"]):
            return random.choice([HumorType.OBSERVATION, HumorType.PUN])

        # Established relationships allow self-deprecating humor.
        if stage in {"established", "long_term", "familiar"}:
            return random.choice([
                HumorType.SELF_DEPRECATING,
                HumorType.WITTY_REMARK,
                HumorType.LIGHT_JOKE,
            ])

        # Neutral default.
        return random.choice([HumorType.LIGHT_JOKE, HumorType.WITTY_REMARK])

    def get_joke(self, category: str = "general") -> str:
        """Retrieve a joke from the appropriate category.

        Args:
            category: One of ``"time"``, ``"health"``, ``"technology"``,
                ``"weather"``, ``"general"``, ``"observation"``,
                ``"self_deprecating"``, ``"witty"`` or ``"wristwatch"``.

        Returns:
            A joke / quip string.
        """
        cat = category.lower().strip()
        if cat in _PUNS:
            return random.choice(_PUNS[cat])
        if cat == "observation":
            return random.choice(_OBSERVATIONS)
        if cat == "self_deprecating":
            return random.choice(_SELF_DEPRECATING)
        if cat in {"witty", "witty_remark"}:
            return random.choice(_WITTY_REMARKS)
        # Default: wristwatch jokes pool.
        return random.choice(_WRISTWATCH_JOKES)

    def add_wit(self, text: str) -> str:
        """Append a witty remark to *text*.

        The remark is chosen based on a lightweight analysis of the text
        content (time-related, health-related, or general).
        """
        with self._lock:
            self._humor_injected += 1

        lower_text = text.lower()

        # Choose category by content.
        if any(kw in lower_text for kw in ["time", "timer", "alarm", "o'clock", "minute", "second"]):
            pun = random.choice(_PUNS["time"])
            joke = pun
        elif any(kw in lower_text for kw in ["step", "heart", "sleep", "calorie", "bpm", "rate"]):
            joke = random.choice(_OBSERVATIONS + _PUNS["health"])
        elif any(kw in lower_text for kw in ["weather", "rain", "cloud", "temperature"]):
            joke = random.choice(_PUNS["weather"])
        else:
            joke = random.choice(_WITTY_REMARKS + _WRISTWATCH_JOKES[:5])

        # Do not duplicate if the joke is already in the text.
        if joke.lower()[:20] in lower_text:
            joke = random.choice(_SELF_DEPRECATING)

        if not text.endswith((".", "!", "?")):
            text += "."
        return f"{text} {joke}"

    def get_stats(self) -> Dict:
        """Return runtime statistics."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return {
                "humor_offered": self._humor_offered,
                "humor_injected": self._humor_injected,
                "uptime_seconds": round(elapsed, 2),
                "joke_library_size": (
                    len(_WRISTWATCH_JOKES)
                    + sum(len(v) for v in _PUNS.values())
                    + len(_OBSERVATIONS)
                    + len(_SELF_DEPRECATING)
                    + len(_WITTY_REMARKS)
                ),
                "humor_types": [t.value for t in HumorType],
            }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_integrator_instance: Optional[HumorIntegrator] = None
_integrator_lock = threading.Lock()


def get_humor_integrator() -> HumorIntegrator:
    """Return the module-level :class:`HumorIntegrator` singleton.

    Thread-safe; the instance is created lazily on first call.
    """
    global _integrator_instance
    if _integrator_instance is None:
        with _integrator_lock:
            if _integrator_instance is None:
                _integrator_instance = HumorIntegrator()
    return _integrator_instance


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hi = get_humor_integrator()

    print("=== HumorIntegrator Demo ===\n")

    # Should / should not add humor.
    cases = [
        ({"situation": "morning routine"}, "happy",   "established"),
        ({"situation": "emergency"},       "stressed", "established"),
        ({"situation": "workout"},         "excited",  "familiar"),
        ({"situation": "bedtime"},         "tired",    "familiar"),
        ({"situation": "general"},         "neutral",  "new"),
    ]
    print("-- should_add_humor --")
    for ctx, mood, stage in cases:
        result = hi.should_add_humor(ctx, mood, stage)
        print(f"  ctx={ctx['situation']:<20} mood={mood:<10} stage={stage:<12} -> {result}")

    print("\n-- add_wit --")
    samples = [
        "Your heart rate is 72 bpm.",
        "Timer set for 10 minutes.",
        "It is currently 14:32.",
        "It is partly cloudy at 18°C.",
        "Message sent to Alex.",
    ]
    for s in samples:
        print(f"  Before: {s!r}")
        print(f"  After : {hi.add_wit(s)!r}\n")

    print("\n-- jokes by category --")
    for cat in ["time", "health", "self_deprecating", "witty", "observation"]:
        print(f"  [{cat}] {hi.get_joke(cat)!r}")

    print("\nStats:", hi.get_stats())
