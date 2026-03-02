"""
Sentiment Analysis — AI Holographic Wristwatch

Real-time sentiment and emotion detection from text input:
- Valence (positive/negative/neutral) classification
- Arousal level (calm/excited)
- Emotion intensity scoring
- Contextual sentiment tracking across conversation turns
- Used to adapt AI personality and response tone
"""

from __future__ import annotations

import re
import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class Sentiment(Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE      = "positive"
    NEUTRAL       = "neutral"
    NEGATIVE      = "negative"
    VERY_NEGATIVE = "very_negative"


class EmotionLabel(Enum):
    JOY       = "joy"
    SADNESS   = "sadness"
    ANGER     = "anger"
    FEAR      = "fear"
    SURPRISE  = "surprise"
    DISGUST   = "disgust"
    NEUTRAL   = "neutral"
    ANXIETY   = "anxiety"
    EXCITEMENT = "excitement"


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a text input."""
    text: str
    sentiment: Sentiment = Sentiment.NEUTRAL
    valence: float = 0.0            # -1 (very negative) to +1 (very positive)
    arousal: float = 0.5            # 0 (calm) to 1 (excited)
    dominant_emotion: EmotionLabel = EmotionLabel.NEUTRAL
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    intensity: float = 0.5          # 0–1 overall emotion intensity
    subjectivity: float = 0.5       # 0 (objective) to 1 (subjective)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Lexicon-based sentiment scorer
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "great": 0.8, "love": 0.9, "excellent": 0.85, "happy": 0.8,
    "good": 0.6, "nice": 0.6, "wonderful": 0.9, "amazing": 0.85,
    "awesome": 0.85, "fantastic": 0.9, "perfect": 0.9, "thanks": 0.5,
    "helpful": 0.65, "beautiful": 0.75, "enjoy": 0.7, "pleased": 0.65,
    "terrific": 0.8, "splendid": 0.8, "delightful": 0.8, "superb": 0.85,
}

NEGATIVE_WORDS = {
    "bad": -0.6, "hate": -0.9, "terrible": -0.85, "awful": -0.8,
    "horrible": -0.85, "poor": -0.55, "sad": -0.65, "angry": -0.75,
    "frustrated": -0.65, "disappointed": -0.65, "annoying": -0.6,
    "worried": -0.55, "scared": -0.65, "anxious": -0.6, "hurt": -0.7,
    "pain": -0.65, "ugly": -0.65, "disgusting": -0.8, "wrong": -0.5,
}

INTENSIFIERS = {"very": 1.5, "really": 1.4, "extremely": 1.7, "so": 1.3,
                "absolutely": 1.6, "totally": 1.4, "quite": 1.2}

NEGATORS = {"not", "no", "never", "don't", "doesn't", "didn't", "won't",
            "can't", "isn't", "aren't", "wasn't", "weren't"}


class LexiconSentimentScorer:
    """VADER-inspired lexicon-based sentiment scorer."""

    def score(self, text: str) -> float:
        """Returns valence score in [-1, 1]."""
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        total = 0.0
        n = len(words)
        for i, word in enumerate(words):
            score = (POSITIVE_WORDS.get(word, 0.0) + NEGATIVE_WORDS.get(word, 0.0))
            if score == 0.0:
                continue

            # Check intensifier before
            if i > 0 and words[i-1] in INTENSIFIERS:
                score *= INTENSIFIERS[words[i-1]]

            # Check negator
            neg_window = words[max(0, i-3):i]
            if any(w in NEGATORS for w in neg_window):
                score = -score * 0.8

            # Exclamation emphasis
            if text.rstrip().endswith("!"):
                score *= 1.2

            total += score

        return max(-1.0, min(1.0, total / max(1, n) * 5))


class EmotionClassifier:
    """Rule-based emotion classification from lexical features."""

    EMOTION_KEYWORDS: Dict[EmotionLabel, List[str]] = {
        EmotionLabel.JOY:       ["happy", "joy", "love", "excited", "great", "wonderful"],
        EmotionLabel.SADNESS:   ["sad", "cry", "depressed", "unhappy", "miss", "lonely"],
        EmotionLabel.ANGER:     ["angry", "mad", "furious", "hate", "frustrated", "annoyed"],
        EmotionLabel.FEAR:      ["scared", "afraid", "terrified", "worried", "anxious"],
        EmotionLabel.SURPRISE:  ["wow", "amazing", "unexpected", "shocked", "surprised"],
        EmotionLabel.DISGUST:   ["disgusting", "gross", "awful", "horrible", "nasty"],
        EmotionLabel.ANXIETY:   ["nervous", "anxious", "stress", "worried", "overwhelmed"],
        EmotionLabel.EXCITEMENT:["excited", "thrilled", "can't wait", "awesome", "fantastic"],
    }

    def classify(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1.0 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion.value] = min(1.0, score / 3.0)
        if not scores:
            scores[EmotionLabel.NEUTRAL.value] = 1.0
        return scores


def _valence_to_sentiment(valence: float) -> Sentiment:
    if valence > 0.5:   return Sentiment.VERY_POSITIVE
    if valence > 0.15:  return Sentiment.POSITIVE
    if valence < -0.5:  return Sentiment.VERY_NEGATIVE
    if valence < -0.15: return Sentiment.NEGATIVE
    return Sentiment.NEUTRAL


_GLOBAL_SA: Optional["SentimentAnalyzer"] = None
_GLOBAL_SA_LOCK = threading.Lock()


class SentimentAnalyzer:
    """
    Text sentiment analyzer combining lexicon scoring and emotion classification.
    Maintains conversation-level sentiment trajectory for trend analysis.
    """

    def __init__(self) -> None:
        self._scorer     = LexiconSentimentScorer()
        self._classifier = EmotionClassifier()
        self._history: Deque[SentimentResult] = deque(maxlen=50)
        self._lock = threading.RLock()
        self._analysis_count = 0

    def analyze(self, text: str) -> SentimentResult:
        with self._lock:
            valence  = self._scorer.score(text)
            emotions = self._classifier.classify(text)
            dominant = max(emotions, key=emotions.get)
            intensity = max(emotions.values()) if emotions else 0.0
            arousal  = min(1.0, intensity + abs(valence) * 0.5)
            subj     = min(1.0, (len(POSITIVE_WORDS) + len(NEGATIVE_WORDS)) / 1000 +
                         sum(v for v in emotions.values()) * 0.3)

            result = SentimentResult(
                text=text,
                sentiment=_valence_to_sentiment(valence),
                valence=valence,
                arousal=arousal,
                dominant_emotion=EmotionLabel(dominant) if dominant != "neutral" else EmotionLabel.NEUTRAL,
                emotion_scores=emotions,
                intensity=intensity,
                subjectivity=min(1.0, subj),
            )
            self._history.append(result)
            self._analysis_count += 1
            return result

    def get_conversation_sentiment(self) -> float:
        """Rolling average valence over recent turns."""
        with self._lock:
            if not self._history:
                return 0.0
            recent = list(self._history)[-10:]
            return sum(r.valence for r in recent) / len(recent)

    def get_sentiment_trend(self) -> str:
        """Detect if conversation mood is improving or declining."""
        with self._lock:
            if len(self._history) < 4:
                return "stable"
            vals = [r.valence for r in list(self._history)[-8:]]
            slope = (vals[-1] - vals[0]) / len(vals)
            if slope > 0.05:  return "improving"
            if slope < -0.05: return "declining"
            return "stable"

    def get_stats(self) -> Dict:
        return {"analyzed": self._analysis_count,
                "current_trend": self.get_sentiment_trend()}


def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _GLOBAL_SA
    with _GLOBAL_SA_LOCK:
        if _GLOBAL_SA is None:
            _GLOBAL_SA = SentimentAnalyzer()
        return _GLOBAL_SA


def run_sentiment_analysis_tests() -> bool:
    sa = SentimentAnalyzer()
    r1 = sa.analyze("I love this watch, it's absolutely amazing!")
    assert r1.sentiment in (Sentiment.POSITIVE, Sentiment.VERY_POSITIVE), f"Got {r1.sentiment}"
    assert r1.valence > 0.0

    r2 = sa.analyze("This is terrible and I hate it")
    assert r2.sentiment in (Sentiment.NEGATIVE, Sentiment.VERY_NEGATIVE), f"Got {r2.sentiment}"
    assert r2.valence < 0.0

    r3 = sa.analyze("What time is it?")
    assert r3.sentiment == Sentiment.NEUTRAL

    logger.info("SentimentAnalyzer tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_sentiment_analysis_tests()
    sa = get_sentiment_analyzer()
    samples = [
        "I'm so happy today!",
        "I'm worried about my health results",
        "Can you set a reminder?",
        "This is frustrating, the hologram keeps glitching",
    ]
    for s in samples:
        r = sa.analyze(s)
        print(f"  '{s[:40]}...' → {r.sentiment.value} ({r.valence:+.2f}) [{r.dominant_emotion.value}]")
    print(f"  Conversation trend: {sa.get_sentiment_trend()}")
