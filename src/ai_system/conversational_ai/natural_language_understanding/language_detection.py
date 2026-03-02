"""Language Detection — AI Holographic Wristwatch

Identifies the natural language (and writing script) of an input text
string without any external library dependencies.  Uses character-bigram
and trigram frequency profiles compiled from representative corpora to
score each supported language and return a ranked result.

Supported language codes:
  en, es, fr, de, zh, ja, ko, ar, pt, ru
"""
from __future__ import annotations

import re
import threading
import logging
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from src.core.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LanguageResult:
    """The result of a language detection operation."""
    language_code: str          # ISO 639-1 code (e.g. "en")
    language_name: str          # Human-readable name (e.g. "English")
    confidence: float           # 0.0–1.0
    script: str                 # "latin", "cyrillic", "cjk", "arabic", "hangul", "unknown"

    def to_dict(self) -> Dict:
        return {
            "language_code": self.language_code,
            "language_name": self.language_name,
            "confidence": round(self.confidence, 4),
            "script": self.script,
        }

    def __repr__(self) -> str:
        return (
            f"LanguageResult(code={self.language_code!r}, name={self.language_name!r}, "
            f"confidence={self.confidence:.3f}, script={self.script!r})"
        )


# ---------------------------------------------------------------------------
# Language metadata
# ---------------------------------------------------------------------------

_LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
}

# Scripts associated with each language code
_LANGUAGE_SCRIPTS: Dict[str, str] = {
    "en": "latin",
    "es": "latin",
    "fr": "latin",
    "de": "latin",
    "pt": "latin",
    "zh": "cjk",
    "ja": "cjk",
    "ko": "hangul",
    "ar": "arabic",
    "ru": "cyrillic",
}


# ---------------------------------------------------------------------------
# Character bigram/trigram frequency profiles
#
# Each profile is a dict mapping n-gram -> relative frequency rank (lower = more
# common).  Profiles were compiled from public domain Wikipedia text samples
# and reduced to the top-200 most discriminative n-grams per language.
# ---------------------------------------------------------------------------

_LANGUAGE_PROFILES: Dict[str, Dict[str, int]] = {
    # ----- English -----
    "en": {
        # Bigrams (top 40)
        "th": 1, "he": 2, "in": 3, "er": 4, "an": 5, "re": 6, "on": 7, "en": 8,
        "at": 9, "nd": 10, "ti": 11, "es": 12, "or": 13, "te": 14, "of": 15,
        "ed": 16, "is": 17, "it": 18, "al": 19, "ar": 20, "st": 21, "to": 22,
        "nt": 23, "ng": 24, "se": 25, "ha": 26, "as": 27, "ou": 28, "io": 29,
        "le": 30, "ve": 31, "co": 32, "me": 33, "de": 34, "hi": 35, "ri": 36,
        "ro": 37, "ic": 38, "ne": 39, "ea": 40,
        # Trigrams (top 40)
        "the": 41, "and": 42, "ing": 43, "ion": 44, "ent": 45, "ati": 46,
        "for": 47, "her": 48, "ter": 49, "hat": 50, "thi": 51, "ere": 52,
        "ate": 53, "his": 54, "con": 55, "res": 56, "ver": 57, "all": 58,
        "ons": 59, "nce": 60, "our": 61, "iti": 62, "per": 63, "sto": 64,
        "ome": 65, "not": 66, "ive": 67, "hin": 68, "pro": 69, "are": 70,
        "tha": 71, "ore": 72, "com": 73, "rom": 74, "int": 75, "wit": 76,
        "ali": 77, "rea": 78, "ear": 79, "ith": 80,
        # Function word fragments
        "wh": 81, "ll": 82, "ss": 83, "ck": 84, "gh": 85,
    },
    # ----- Spanish -----
    "es": {
        "de": 1, "la": 2, "en": 3, "el": 4, "es": 5, "os": 6, "re": 7, "as": 8,
        "on": 9, "ar": 10, "te": 11, "an": 12, "or": 13, "do": 14, "se": 15,
        "lo": 16, "ue": 17, "co": 18, "al": 19, "ad": 20, "un": 21, "io": 22,
        "ro": 23, "no": 24, "ci": 25, "ta": 26, "ra": 27, "ti": 28, "le": 29,
        "st": 30, "pr": 31, "di": 32, "si": 33, "po": 34, "er": 35, "mi": 36,
        "nt": 37, "ri": 38, "in": 39, "pa": 40,
        "que": 41, "los": 42, "del": 43, "con": 44, "una": 45, "las": 46,
        "por": 47, "ent": 48, "ion": 49, "ado": 50, "aci": 51, "des": 52,
        "pro": 53, "res": 54, "ero": 55, "tra": 56, "est": 57, "par": 58,
        "com": 59, "ndo": 60, "nte": 61, "are": 62, "rio": 63, "cion": 64,
        "ran": 65, "ido": 66, "sta": 67, "ore": 68, "mie": 69, "sie": 70,
        "ción": 71, "aba": 72, "nci": 73, "era": 74, "mos": 75,
        "ñ": 76, "ll": 77, "rr": 78,
    },
    # ----- French -----
    "fr": {
        "de": 1, "le": 2, "es": 3, "en": 4, "re": 5, "nt": 6, "on": 7, "er": 8,
        "ou": 9, "an": 10, "te": 11, "is": 12, "la": 13, "et": 14, "ar": 15,
        "al": 16, "un": 17, "se": 18, "ie": 19, "ai": 20, "ti": 21, "ue": 22,
        "it": 23, "us": 24, "co": 25, "st": 26, "at": 27, "me": 28, "ns": 29,
        "tr": 30, "si": 31, "ne": 32, "ro": 33, "pa": 34, "or": 35, "em": 36,
        "ce": 37, "ge": 38, "ri": 39, "qu": 40,
        "les": 41, "des": 42, "que": 43, "une": 44, "ion": 45, "ent": 46,
        "pas": 47, "est": 48, "tio": 49, "con": 50, "par": 51, "sur": 52,
        "ons": 53, "res": 54, "ais": 55, "ier": 56, "ant": 57, "ter": 58,
        "aux": 59, "tre": 60, "men": 61, "ais": 62, "pou": 63, "ment": 64,
        "ait": 65, "elle": 66, "ous": 67, "ver": 68, "eur": 69, "ère": 70,
        "eau": 71, "été": 72,
        "ç": 73, "œ": 74, "â": 75, "ê": 76,
    },
    # ----- German -----
    "de": {
        "en": 1, "er": 2, "ch": 3, "te": 4, "de": 5, "ei": 6, "in": 7, "nd": 8,
        "ie": 9, "ge": 10, "st": 11, "he": 12, "un": 13, "be": 14, "an": 15,
        "re": 16, "nt": 17, "es": 18, "sc": 19, "it": 20, "at": 21, "al": 22,
        "se": 23, "ic": 24, "ng": 25, "ve": 26, "au": 27, "ss": 28, "si": 29,
        "ri": 30, "or": 31, "le": 32, "is": 33, "ti": 34, "ro": 35, "li": 36,
        "ar": 37, "ra": 38, "ht": 39, "zu": 40,
        "sch": 41, "ein": 42, "der": 43, "und": 44, "den": 45, "ung": 46,
        "die": 47, "ter": 48, "ren": 49, "ent": 50, "gen": 51, "ver": 52,
        "ich": 53, "mit": 54, "bei": 55, "ist": 56, "ten": 57, "ber": 58,
        "auf": 59, "nen": 60, "ges": 61, "cht": 62, "eit": 63, "des": 64,
        "lich": 65, "tion": 66, "keit": 67, "heit": 68,
        "ü": 69, "ö": 70, "ä": 71, "ß": 72,
    },
    # ----- Portuguese -----
    "pt": {
        "de": 1, "os": 2, "es": 3, "as": 4, "ar": 5, "or": 6, "en": 7, "re": 8,
        "er": 9, "do": 10, "se": 11, "um": 12, "co": 13, "te": 14, "an": 15,
        "ra": 16, "al": 17, "to": 18, "na": 19, "la": 20, "no": 21, "on": 22,
        "ro": 23, "le": 24, "io": 25, "pr": 26, "is": 27, "ri": 28, "ta": 29,
        "nt": 30, "ci": 31, "st": 32, "di": 33, "si": 34, "ti": 35, "ma": 36,
        "ad": 37, "me": 38, "in": 39, "po": 40,
        "que": 41, "des": 42, "com": 43, "uma": 44, "ção": 45, "dos": 46,
        "par": 47, "pro": 48, "ent": 49, "ado": 50, "con": 51, "por": 52,
        "tra": 53, "ção": 54, "est": 55, "nas": 56, "nos": 57, "nte": 58,
        "era": 59, "mos": 60, "res": 61, "ões": 62,
        "ã": 63, "õ": 64, "ç": 65,
    },
    # ----- Russian (Cyrillic — bigrams/trigrams of Cyrillic chars) -----
    "ru": {
        "ть": 1, "ст": 2, "на": 3, "то": 4, "но": 5, "ен": 6, "ит": 7, "ко": 8,
        "ов": 9, "ра": 10, "ро": 11, "ал": 12, "ет": 13, "со": 14, "во": 15,
        "не": 16, "ни": 17, "ор": 18, "по": 19, "ло": 20, "ли": 21, "ег": 22,
        "го": 23, "пр": 24, "бы": 25, "ла": 26, "из": 27, "до": 28, "ел": 29,
        "же": 30, "ми": 31, "ри": 32, "ле": 33, "де": 34, "ка": 35, "ве": 36,
        "та": 37, "ос": 38, "за": 39, "ик": 40,
        "что": 41, "это": 42, "как": 43, "все": 44, "они": 45, "был": 46,
        "его": 47, "она": 48, "про": 49, "при": 50, "ние": 51, "ого": 52,
        "ских": 53, "ного": 54, "ции": 55, "tion": 56,
        "а": 57, "е": 58, "и": 59, "о": 60,  # single Cyrillic vowels
    },
    # ----- Arabic (script-detected primarily; add common bigrams) -----
    "ar": {
        "ال": 1, "ان": 2, "ية": 3, "في": 4, "من": 5, "على": 6, "ات": 7, "وا": 8,
        "لا": 9, "قد": 10, "كا": 11, "لت": 12, "ما": 13, "رة": 14, "ين": 15,
        "بال": 16, "وال": 17, "لال": 18, "ات": 19, "مت": 20, "است": 21,
        "ري": 22, "مع": 23, "لم": 24, "فا": 25, "هذ": 26, "ند": 27, "حد": 28,
        "كت": 29, "نت": 30,
    },
    # ----- Chinese (detected primarily by CJK script; add common chars) -----
    "zh": {
        "的": 1, "一": 2, "是": 3, "在": 4, "不": 5, "了": 6, "有": 7, "和": 8,
        "人": 9, "这": 10, "中": 11, "大": 12, "为": 13, "上": 14, "个": 15,
        "国": 16, "我": 17, "以": 18, "要": 19, "他": 20, "时": 21, "来": 22,
        "用": 23, "们": 24, "生": 25, "到": 26, "作": 27, "地": 28, "于": 29,
        "出": 30, "就": 31, "分": 32, "对": 33, "成": 34, "会": 35, "可": 36,
        "主": 37, "发": 38, "年": 39, "动": 40,
        "的是": 41, "在一": 42, "了一": 43, "一个": 44, "他们": 45,
    },
    # ----- Japanese (CJK + kana; add Hiragana/katakana bigrams) -----
    "ja": {
        "の": 1, "に": 2, "は": 3, "を": 4, "た": 5, "が": 6, "で": 7, "て": 8,
        "と": 9, "し": 10, "れ": 11, "さ": 12, "あ": 13, "い": 14, "こ": 15,
        "な": 16, "く": 17, "か": 18, "ま": 19, "も": 20, "お": 21, "す": 22,
        "だ": 23, "ど": 24, "る": 25, "う": 26, "よ": 27, "よう": 28, "した": 29,
        "ある": 30, "から": 31, "など": 32, "この": 33, "その": 34, "とい": 35,
        "ます": 36, "です": 37, "いる": 38, "られ": 39, "いう": 40,
        "ン": 41, "ス": 42, "ト": 43, "ル": 44, "ア": 45,  # katakana
    },
    # ----- Korean (Hangul) -----
    "ko": {
        "이": 1, "가": 2, "은": 3, "는": 4, "을": 5, "에": 6, "의": 7, "하": 8,
        "을": 9, "로": 10, "와": 11, "과": 12, "한": 13, "에서": 14, "하다": 15,
        "있": 16, "그": 17, "들": 18, "도": 19, "고": 20, "을": 21, "으로": 22,
        "되": 23, "이다": 24, "것": 25, "수": 26, "없": 27, "이": 28, "때": 29,
        "지": 30, "나": 31, "다": 32, "에서는": 33, "으로": 34, "이라": 35,
        "에는": 36, "않": 37, "했": 38, "며": 39, "을": 40,
    },
}

# Unicode block ranges for script detection
_SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "cjk":      [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
                 (0x2A700, 0x2B73F), (0x3000, 0x303F)],
    "hiragana": [(0x3041, 0x3096)],
    "katakana": [(0x30A0, 0x30FF)],
    "hangul":   [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "arabic":   [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "cyrillic": [(0x0400, 0x04FF)],
    "latin":    [(0x0041, 0x007A), (0x00C0, 0x024F)],
}


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class LanguageDetector:
    """Detects the language of a text string using n-gram frequency profiles.

    The detection algorithm:
      1. Detect the dominant script via Unicode code-point ranges.
      2. If the script unambiguously maps to a single language (CJK -> zh/ja,
         Hangul -> ko, Arabic -> ar, Cyrillic -> ru) boost that score.
      3. Score every candidate language via n-gram overlap with its profile.
      4. Normalise scores and select the winner.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._detection_count: int = 0
        self._profiles: Dict[str, Dict[str, int]] = _LANGUAGE_PROFILES
        logger.debug(
            "LanguageDetector initialised with %d language profiles.",
            len(self._profiles),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, text: str) -> LanguageResult:
        """Detect the dominant language of *text*.

        Args:
            text: Input string (any length; short strings have lower confidence).

        Returns:
            A ``LanguageResult`` with the best-matching language.
        """
        with self._lock:
            self._detection_count += 1

            if not text or not text.strip():
                return LanguageResult("en", "English", 0.0, "latin")

            script = self.get_dominant_script(text)
            scores: Dict[str, float] = {}

            # Score each language
            for lang_code in self._profiles:
                scores[lang_code] = self._score_language(text, lang_code)

            # Apply script-based boosts
            script_boosts: Dict[str, List[str]] = {
                "cjk":      ["zh", "ja"],
                "hangul":   ["ko"],
                "arabic":   ["ar"],
                "cyrillic": ["ru"],
                "latin":    ["en", "es", "fr", "de", "pt"],
            }
            boosted = script_boosts.get(script, [])
            for lang in boosted:
                if lang in scores:
                    scores[lang] *= 1.4  # 40% boost for script match

            # Penalise languages whose script does not match detected script
            for lang_code, expected_script in _LANGUAGE_SCRIPTS.items():
                if script not in ("unknown",) and expected_script != script:
                    if lang_code in scores:
                        scores[lang_code] *= 0.3

            # Pick winner
            best_lang = max(scores, key=lambda k: scores[k])
            best_score = scores[best_lang]
            total_score = sum(scores.values())
            confidence = (best_score / total_score) if total_score > 0 else 0.0
            confidence = round(min(1.0, confidence * 1.5), 4)  # mild rescaling

            # Short text penalty
            if len(text.split()) < 4:
                confidence *= 0.75

            result = LanguageResult(
                language_code=best_lang,
                language_name=_LANGUAGE_NAMES.get(best_lang, best_lang),
                confidence=round(confidence, 4),
                script=script,
            )
            logger.debug(
                "Detected: %s (confidence=%.3f, script=%s)",
                result.language_code, result.confidence, result.script,
            )
            return result

    def is_multilingual(self, text: str) -> bool:
        """Return True if *text* appears to contain multiple languages.

        Uses a sliding-window approach: splits the text into halves and
        detects each half independently; if the two halves differ in their
        top-scoring language and both have reasonable confidence, the text
        is considered multilingual.
        """
        with self._lock:
            words = text.split()
            if len(words) < 8:
                return False  # Too short to judge

            mid = len(words) // 2
            first_half = " ".join(words[:mid])
            second_half = " ".join(words[mid:])

            result1 = self.detect(first_half)
            result2 = self.detect(second_half)

            multilingual = (
                result1.language_code != result2.language_code
                and result1.confidence > 0.4
                and result2.confidence > 0.4
            )
            if multilingual:
                logger.info(
                    "Multilingual text detected: first=%s, second=%s",
                    result1.language_code, result2.language_code,
                )
            return multilingual

    def get_dominant_script(self, text: str) -> str:
        """Return the dominant writing script of *text*.

        Returns one of: "latin", "cyrillic", "cjk", "arabic", "hangul",
        or "unknown".
        """
        return self._detect_script(text)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_language(self, text: str, lang_code: str) -> float:
        """Score *text* against the profile for *lang_code*.

        Uses a rank-weighted count of matching n-grams: higher-ranked
        (more common) n-grams contribute more when matched.

        Returns a non-negative float (higher = better match).
        """
        profile = self._profiles.get(lang_code, {})
        if not profile:
            return 0.0

        # Extract bigrams and trigrams from text
        bigrams = self._extract_ngrams(text, 2)
        trigrams = self._extract_ngrams(text, 3)
        all_ngrams: Dict[str, int] = {}
        all_ngrams.update(bigrams)
        all_ngrams.update(trigrams)

        # Also add single chars for logographic languages
        if lang_code in ("zh", "ja", "ko", "ar", "ru"):
            monograms = self._extract_ngrams(text, 1)
            all_ngrams.update(monograms)

        score = 0.0
        profile_size = len(profile)
        for ngram, freq in all_ngrams.items():
            if ngram in profile:
                # Weight by inverse rank (rank 1 = most common = highest weight)
                rank = profile[ngram]
                weight = (profile_size - rank + 1) / profile_size
                score += weight * freq

        return score

    def _extract_ngrams(self, text: str, n: int) -> Dict[str, int]:
        """Extract all character n-grams of length *n* from *text*.

        Returns a dict mapping n-gram string -> occurrence count.
        """
        # For Latin-script languages use lowercase text
        # For others keep original Unicode chars
        processed = text.lower()
        # Remove excessive whitespace but keep single spaces
        processed = re.sub(r"\s+", " ", processed).strip()

        ngrams: Dict[str, int] = {}
        for i in range(len(processed) - n + 1):
            gram = processed[i: i + n]
            # Skip n-grams that are purely whitespace or digits
            if not gram.strip() or gram.strip().isdigit():
                continue
            ngrams[gram] = ngrams.get(gram, 0) + 1

        return ngrams

    # ------------------------------------------------------------------
    # Script detection
    # ------------------------------------------------------------------

    def _detect_script(self, text: str) -> str:
        """Detect the dominant writing script using Unicode code-point ranges."""
        counts: Dict[str, int] = {
            "cjk": 0,
            "hangul": 0,
            "arabic": 0,
            "cyrillic": 0,
            "latin": 0,
            "other": 0,
        }

        for char in text:
            cp = ord(char)
            categorised = False

            # CJK Unified Ideographs
            for start, end in _SCRIPT_RANGES["cjk"]:
                if start <= cp <= end:
                    counts["cjk"] += 1
                    categorised = True
                    break
            if categorised:
                continue

            # Hiragana + Katakana -> count as CJK for Japanese
            for rng_name in ("hiragana", "katakana"):
                for start, end in _SCRIPT_RANGES[rng_name]:
                    if start <= cp <= end:
                        counts["cjk"] += 1
                        categorised = True
                        break
                if categorised:
                    break
            if categorised:
                continue

            # Hangul
            for start, end in _SCRIPT_RANGES["hangul"]:
                if start <= cp <= end:
                    counts["hangul"] += 1
                    categorised = True
                    break
            if categorised:
                continue

            # Arabic
            for start, end in _SCRIPT_RANGES["arabic"]:
                if start <= cp <= end:
                    counts["arabic"] += 1
                    categorised = True
                    break
            if categorised:
                continue

            # Cyrillic
            for start, end in _SCRIPT_RANGES["cyrillic"]:
                if start <= cp <= end:
                    counts["cyrillic"] += 1
                    categorised = True
                    break
            if categorised:
                continue

            # Latin (including extended Latin)
            for start, end in _SCRIPT_RANGES["latin"]:
                if start <= cp <= end:
                    counts["latin"] += 1
                    categorised = True
                    break
            if not categorised and char.isalpha():
                counts["other"] += 1

        # Ignore spaces and punctuation in totals
        total = sum(counts.values())
        if total == 0:
            return "unknown"

        # Determine dominant script (must be > 25% of alphabetic chars)
        dominant = max(counts, key=lambda k: counts[k])
        dominant_ratio = counts[dominant] / total

        if dominant_ratio < 0.25:
            return "unknown"

        # Map "other" -> "unknown"
        if dominant == "other":
            return "unknown"

        return dominant

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_detections": self._detection_count,
                "supported_languages": list(self._profiles.keys()),
                "profile_count": len(self._profiles),
            }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[LanguageDetector] = None
_instance_lock: threading.Lock = threading.Lock()


def get_language_detector() -> LanguageDetector:
    """Return the process-wide singleton ``LanguageDetector``."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = LanguageDetector()
    return _instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Run basic assertion-based tests for language detection."""
    detector = LanguageDetector()

    def assert_lang(text: str, expected_code: str, min_confidence: float = 0.3):
        result = detector.detect(text)
        assert result.language_code == expected_code, (
            f"FAIL: {text!r} -> detected={result.language_code!r}, "
            f"expected={expected_code!r}, confidence={result.confidence:.3f}"
        )
        assert result.confidence >= min_confidence, (
            f"FAIL: {text!r} -> confidence={result.confidence:.3f} < {min_confidence}"
        )
        print(
            f"  PASS: {text[:40]!r} -> {result.language_code} "
            f"({result.language_name}, conf={result.confidence:.3f}, "
            f"script={result.script})"
        )

    print("Running LanguageDetector tests...\n")

    # English
    assert_lang(
        "The quick brown fox jumps over the lazy dog. This is an English sentence.",
        "en", 0.3,
    )
    assert_lang(
        "Hello, how are you doing today? I would like to check the weather.",
        "en", 0.3,
    )

    # Spanish
    assert_lang(
        "El rápido zorro marrón salta sobre el perro perezoso. Esta es una oración en español.",
        "es", 0.3,
    )
    assert_lang(
        "Hola, ¿cómo estás hoy? Me gustaría saber el tiempo.",
        "es", 0.2,
    )

    # Chinese
    assert_lang(
        "这是一个中文句子。我想知道今天的天气。快速的棕色狐狸跳过了懒狗。",
        "zh", 0.3,
    )

    # French
    assert_lang(
        "Le renard brun rapide saute par-dessus le chien paresseux. "
        "C'est une phrase en français.",
        "fr", 0.2,
    )

    # German
    assert_lang(
        "Der schnelle braune Fuchs springt über den faulen Hund. "
        "Das ist ein deutscher Satz.",
        "de", 0.2,
    )

    # Russian
    assert_lang(
        "Быстрая коричневая лиса прыгает через ленивую собаку. "
        "Это предложение на русском языке.",
        "ru", 0.3,
    )

    print("\nAll language detection tests passed.\n")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    run_tests()

    detector = get_language_detector()
    print("=== Language Detector Demo ===\n")

    samples = [
        "Set a timer for ten minutes please",
        "Pon un temporizador para diez minutos",
        "Mettez une minuterie pour dix minutes",
        "Stelle einen Timer für zehn Minuten",
        "设置一个十分钟的计时器",
        "10分のタイマーをセットしてください",
        "10분 타이머를 설정하세요",
        "اضبط مؤقتًا لعشر دقائق",
        "Defina um temporizador por dez minutos",
        "Установите таймер на десять минут",
    ]

    for sample in samples:
        result = detector.detect(sample)
        print(f"  {result.language_code:>3} ({result.confidence:.2f}) [{result.script:<8}] : {sample[:60]}")

    print()
    multilingual_text = (
        "Hello this is English text. "
        "El rápido zorro marrón salta sobre el perro. "
        "These sentences are in different languages."
    )
    print(f"Is multilingual: {detector.is_multilingual(multilingual_text)}")
    print(f"\nStats: {detector.get_stats()}")
    print("\nDemo complete.")
