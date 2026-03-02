"""Cultural Adaptation module — AI Holographic Wristwatch.

Provides Hofstede-inspired cultural profiling and communication adaptation
so that Aria can tailor her communication style to the user's cultural context.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CulturalRegion(Enum):
    NORTH_AMERICA = "north_america"
    LATIN_AMERICA = "latin_america"
    WESTERN_EUROPE = "western_europe"
    EASTERN_EUROPE = "eastern_europe"
    EAST_ASIA = "east_asia"
    SOUTH_ASIA = "south_asia"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"
    OCEANIA = "oceania"


class FormalityStyle(Enum):
    VERY_FORMAL = "very_formal"
    FORMAL = "formal"
    NEUTRAL = "neutral"
    INFORMAL = "informal"
    CASUAL = "casual"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CulturalNorms:
    greeting_style: str
    personal_space: str
    communication_style: str
    time_orientation: str
    taboo_topics: List[str] = field(default_factory=list)
    politeness_markers: List[str] = field(default_factory=list)


@dataclass
class CulturalProfile:
    region: CulturalRegion
    formality_level: FormalityStyle
    directness: float          # 0.0 = very indirect, 1.0 = very direct
    uncertainty_avoidance: float   # 0.0 = tolerant of ambiguity, 1.0 = avoids uncertainty
    individualism: float       # 0.0 = collectivist, 1.0 = individualist
    cultural_norms: CulturalNorms


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

class CulturalAdapter:
    """Detect and adapt to the user's cultural context in real time."""

    _CULTURAL_PROFILES: Dict[CulturalRegion, CulturalProfile] = {
        CulturalRegion.NORTH_AMERICA: CulturalProfile(
            region=CulturalRegion.NORTH_AMERICA,
            formality_level=FormalityStyle.INFORMAL,
            directness=0.82,
            uncertainty_avoidance=0.46,
            individualism=0.91,
            cultural_norms=CulturalNorms(
                greeting_style="Casual handshake or wave; first names common",
                personal_space="Arm's length (~1 metre) preferred",
                communication_style="Direct, informal, results-oriented",
                time_orientation="Short-term, punctuality valued",
                taboo_topics=["salary details with strangers", "voting choices"],
                politeness_markers=["please", "thank you", "sure thing", "sounds good"],
            ),
        ),
        CulturalRegion.LATIN_AMERICA: CulturalProfile(
            region=CulturalRegion.LATIN_AMERICA,
            formality_level=FormalityStyle.INFORMAL,
            directness=0.50,
            uncertainty_avoidance=0.75,
            individualism=0.38,
            cultural_norms=CulturalNorms(
                greeting_style="Warm embrace or cheek kiss; first names after introduction",
                personal_space="Close physical proximity; touch is acceptable",
                communication_style="Expressive, relationship-first, context-rich",
                time_orientation="Polychronic; relationships over schedules",
                taboo_topics=["negative comments about family", "political criticism"],
                politeness_markers=["por favor", "gracias", "con mucho gusto", "claro que sí"],
            ),
        ),
        CulturalRegion.WESTERN_EUROPE: CulturalProfile(
            region=CulturalRegion.WESTERN_EUROPE,
            formality_level=FormalityStyle.FORMAL,
            directness=0.70,
            uncertainty_avoidance=0.62,
            individualism=0.78,
            cultural_norms=CulturalNorms(
                greeting_style="Handshake; titles used until invited to use first names",
                personal_space="Moderate distance; less physical contact",
                communication_style="Reserved but direct; privacy valued",
                time_orientation="Monochronic; punctuality expected",
                taboo_topics=["personal income", "religion in workplace"],
                politeness_markers=["please", "thank you", "I beg your pardon", "if you would"],
            ),
        ),
        CulturalRegion.EASTERN_EUROPE: CulturalProfile(
            region=CulturalRegion.EASTERN_EUROPE,
            formality_level=FormalityStyle.FORMAL,
            directness=0.65,
            uncertainty_avoidance=0.80,
            individualism=0.55,
            cultural_norms=CulturalNorms(
                greeting_style="Firm handshake; use of patronymics or titles",
                personal_space="Moderate; directness considered respectful",
                communication_style="Frank, structured, formal in professional settings",
                time_orientation="Past-oriented; tradition respected",
                taboo_topics=["WWII sensitivities", "political history"],
                politeness_markers=["please", "thank you", "allow me", "I would appreciate"],
            ),
        ),
        CulturalRegion.EAST_ASIA: CulturalProfile(
            region=CulturalRegion.EAST_ASIA,
            formality_level=FormalityStyle.VERY_FORMAL,
            directness=0.25,
            uncertainty_avoidance=0.70,
            individualism=0.20,
            cultural_norms=CulturalNorms(
                greeting_style="Bow or formal nod; family name used with honorifics",
                personal_space="Larger in formal contexts; context-dependent",
                communication_style="High-context, indirect, face-saving",
                time_orientation="Long-term; harmony and consensus valued",
                taboo_topics=["direct criticism", "political topics", "discussing failure publicly"],
                politeness_markers=["sumimasen", "onegaishimasu", "please", "I humbly suggest"],
            ),
        ),
        CulturalRegion.SOUTH_ASIA: CulturalProfile(
            region=CulturalRegion.SOUTH_ASIA,
            formality_level=FormalityStyle.FORMAL,
            directness=0.40,
            uncertainty_avoidance=0.60,
            individualism=0.48,
            cultural_norms=CulturalNorms(
                greeting_style="Namaste or handshake; elders addressed with titles",
                personal_space="Context-dependent; head wobble as acknowledgement",
                communication_style="High-context; indirect refusal common",
                time_orientation="Fluid; relationship and hierarchy valued",
                taboo_topics=["caste", "religious differences", "beef consumption"],
                politeness_markers=["please", "kindly", "as you wish", "I hope you don't mind"],
            ),
        ),
        CulturalRegion.MIDDLE_EAST: CulturalProfile(
            region=CulturalRegion.MIDDLE_EAST,
            formality_level=FormalityStyle.FORMAL,
            directness=0.45,
            uncertainty_avoidance=0.68,
            individualism=0.38,
            cultural_norms=CulturalNorms(
                greeting_style="'As-salamu alaykum'; right hand to heart; titles used",
                personal_space="Close among same gender; gender norms vary by country",
                communication_style="Relationship-centric; indirect on sensitive topics",
                time_orientation="Polychronic; hospitality prioritised",
                taboo_topics=["Israel-Palestine directly", "alcohol", "religious criticism"],
                politeness_markers=["inshallah", "please", "with respect", "if it pleases you"],
            ),
        ),
        CulturalRegion.AFRICA: CulturalProfile(
            region=CulturalRegion.AFRICA,
            formality_level=FormalityStyle.NEUTRAL,
            directness=0.55,
            uncertainty_avoidance=0.55,
            individualism=0.27,
            cultural_norms=CulturalNorms(
                greeting_style="Varies widely; greet elders first; Ubuntu philosophy",
                personal_space="Context and region dependent; community-oriented",
                communication_style="Storytelling tradition; community references",
                time_orientation="African time concept; relationships over schedules",
                taboo_topics=["colonialism jokes", "tribal conflicts"],
                politeness_markers=["please", "thank you", "I appreciate", "with respect"],
            ),
        ),
        CulturalRegion.OCEANIA: CulturalProfile(
            region=CulturalRegion.OCEANIA,
            formality_level=FormalityStyle.INFORMAL,
            directness=0.75,
            uncertainty_avoidance=0.51,
            individualism=0.80,
            cultural_norms=CulturalNorms(
                greeting_style="Casual handshake or wave; first names universal",
                personal_space="Arm's length; relaxed body language",
                communication_style="Direct, casual, understated humour",
                time_orientation="Present-focused; work-life balance valued",
                taboo_topics=["tall poppy syndrome topics", "political overclaiming"],
                politeness_markers=["no worries", "cheers", "please", "ta"],
            ),
        ),
    }

    _REGION_FROM_COUNTRY: Dict[str, CulturalRegion] = {
        # North America
        "US": CulturalRegion.NORTH_AMERICA,
        "CA": CulturalRegion.NORTH_AMERICA,
        "MX": CulturalRegion.LATIN_AMERICA,
        # Latin America
        "BR": CulturalRegion.LATIN_AMERICA,
        "AR": CulturalRegion.LATIN_AMERICA,
        "CO": CulturalRegion.LATIN_AMERICA,
        "CL": CulturalRegion.LATIN_AMERICA,
        "PE": CulturalRegion.LATIN_AMERICA,
        "VE": CulturalRegion.LATIN_AMERICA,
        # Western Europe
        "GB": CulturalRegion.WESTERN_EUROPE,
        "DE": CulturalRegion.WESTERN_EUROPE,
        "FR": CulturalRegion.WESTERN_EUROPE,
        "ES": CulturalRegion.WESTERN_EUROPE,
        "IT": CulturalRegion.WESTERN_EUROPE,
        "NL": CulturalRegion.WESTERN_EUROPE,
        "SE": CulturalRegion.WESTERN_EUROPE,
        "NO": CulturalRegion.WESTERN_EUROPE,
        "CH": CulturalRegion.WESTERN_EUROPE,
        # Eastern Europe
        "PL": CulturalRegion.EASTERN_EUROPE,
        "RU": CulturalRegion.EASTERN_EUROPE,
        "UA": CulturalRegion.EASTERN_EUROPE,
        "RO": CulturalRegion.EASTERN_EUROPE,
        "CZ": CulturalRegion.EASTERN_EUROPE,
        # East Asia
        "JP": CulturalRegion.EAST_ASIA,
        "CN": CulturalRegion.EAST_ASIA,
        "KR": CulturalRegion.EAST_ASIA,
        "TW": CulturalRegion.EAST_ASIA,
        # South Asia
        "IN": CulturalRegion.SOUTH_ASIA,
        "PK": CulturalRegion.SOUTH_ASIA,
        "BD": CulturalRegion.SOUTH_ASIA,
        "LK": CulturalRegion.SOUTH_ASIA,
        # Middle East
        "SA": CulturalRegion.MIDDLE_EAST,
        "AE": CulturalRegion.MIDDLE_EAST,
        "TR": CulturalRegion.MIDDLE_EAST,
        "EG": CulturalRegion.MIDDLE_EAST,
        "IR": CulturalRegion.MIDDLE_EAST,
        # Africa
        "NG": CulturalRegion.AFRICA,
        "ZA": CulturalRegion.AFRICA,
        "KE": CulturalRegion.AFRICA,
        "GH": CulturalRegion.AFRICA,
        "ET": CulturalRegion.AFRICA,
        # Oceania
        "AU": CulturalRegion.OCEANIA,
        "NZ": CulturalRegion.OCEANIA,
    }

    # Language code hints when country is unavailable
    _REGION_FROM_LANGUAGE: Dict[str, CulturalRegion] = {
        "ja": CulturalRegion.EAST_ASIA,
        "zh": CulturalRegion.EAST_ASIA,
        "ko": CulturalRegion.EAST_ASIA,
        "ar": CulturalRegion.MIDDLE_EAST,
        "hi": CulturalRegion.SOUTH_ASIA,
        "pt": CulturalRegion.LATIN_AMERICA,
        "es": CulturalRegion.LATIN_AMERICA,
        "ru": CulturalRegion.EASTERN_EUROPE,
        "de": CulturalRegion.WESTERN_EUROPE,
        "fr": CulturalRegion.WESTERN_EUROPE,
        "en": CulturalRegion.NORTH_AMERICA,
    }

    def __init__(self) -> None:
        self._current_profile: CulturalProfile = self._CULTURAL_PROFILES[CulturalRegion.NORTH_AMERICA]
        self._lock = threading.Lock()
        self._adapt_count: int = 0
        logger.debug("CulturalAdapter initialised with default NORTH_AMERICA profile")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_cultural_context(
        self,
        country_code: str,
        language_code: str,
    ) -> CulturalProfile:
        """Return the most appropriate CulturalProfile for given country/language codes.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g. "JP").
            language_code: BCP-47 language code (e.g. "ja").

        Returns:
            Matching CulturalProfile (falls back to NORTH_AMERICA if unknown).
        """
        country_upper = (country_code or "").upper()
        lang_lower = (language_code or "").lower().split("-")[0]

        region: Optional[CulturalRegion] = self._REGION_FROM_COUNTRY.get(country_upper)
        if region is None:
            region = self._REGION_FROM_LANGUAGE.get(lang_lower)
        if region is None:
            logger.warning(
                "Unknown country=%s lang=%s; defaulting to NORTH_AMERICA",
                country_upper, lang_lower,
            )
            region = CulturalRegion.NORTH_AMERICA

        profile = self._CULTURAL_PROFILES[region]
        with self._lock:
            self._current_profile = profile
        logger.info("Cultural context detected: %s", region.value)
        return profile

    def adapt_communication(self, text: str, profile: CulturalProfile) -> str:
        """Adjust a text response according to the supplied cultural profile.

        Modifications applied:
        - Adds opening politeness marker from the profile when directness is low.
        - Softens direct assertions with hedging language when directness < 0.4.
        - Appends a culturally appropriate closing courtesy when formality is high.

        Args:
            text: Raw response text to adapt.
            profile: Target cultural profile.

        Returns:
            Culturally adapted response text.
        """
        with self._lock:
            self._adapt_count += 1

        adapted = text.strip()

        # Prepend politeness marker for high-formality cultures
        if profile.formality_level in (FormalityStyle.VERY_FORMAL, FormalityStyle.FORMAL):
            if profile.cultural_norms.politeness_markers:
                marker = profile.cultural_norms.politeness_markers[0].capitalize()
                # Only add if the text doesn't already start with a marker
                if not any(adapted.lower().startswith(m.lower()) for m in profile.cultural_norms.politeness_markers):
                    adapted = f"{marker} — {adapted}"

        # Hedge direct statements for indirect cultures (directness < 0.4)
        if profile.directness < 0.40:
            direct_starters = [
                ("You should ", "You might consider "),
                ("You must ", "It may be worth "),
                ("Do this:", "One approach could be:"),
                ("The answer is ", "One way to look at it is "),
            ]
            for direct, hedged in direct_starters:
                if direct in adapted:
                    adapted = adapted.replace(direct, hedged)

        # Append closing courtesy for very formal profiles
        if profile.formality_level == FormalityStyle.VERY_FORMAL:
            if not adapted.endswith((".", "!", "?")):
                adapted += "."
            adapted += " Please let me know if further assistance is needed."

        logger.debug(
            "Communication adapted for region=%s directness=%.2f",
            profile.region.value, profile.directness,
        )
        return adapted

    def get_cultural_norms(self, region: CulturalRegion) -> CulturalNorms:
        """Return the CulturalNorms for a given region.

        Args:
            region: Target CulturalRegion.

        Returns:
            Corresponding CulturalNorms dataclass instance.
        """
        return self._CULTURAL_PROFILES[region].cultural_norms

    def set_active_profile(self, profile: CulturalProfile) -> None:
        """Manually override the active cultural profile.

        Args:
            profile: New CulturalProfile to use as default.
        """
        with self._lock:
            self._current_profile = profile
        logger.info("Active cultural profile overridden to: %s", profile.region.value)

    def get_active_profile(self) -> CulturalProfile:
        """Return the currently active CulturalProfile.

        Returns:
            The active CulturalProfile.
        """
        with self._lock:
            return self._current_profile

    def get_stats(self) -> Dict:
        """Return runtime statistics for the adapter.

        Returns:
            Dict with stats fields.
        """
        with self._lock:
            return {
                "active_region": self._current_profile.region.value,
                "active_formality": self._current_profile.formality_level.value,
                "active_directness": self._current_profile.directness,
                "adapt_count": self._adapt_count,
                "known_regions": len(self._CULTURAL_PROFILES),
                "known_country_codes": len(self._REGION_FROM_COUNTRY),
            }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_cultural_adapter_instance: Optional[CulturalAdapter] = None
_cultural_adapter_lock = threading.Lock()


def get_cultural_adapter() -> CulturalAdapter:
    """Return the process-wide singleton CulturalAdapter instance.

    Returns:
        Singleton CulturalAdapter.
    """
    global _cultural_adapter_instance
    if _cultural_adapter_instance is None:
        with _cultural_adapter_lock:
            if _cultural_adapter_instance is None:
                _cultural_adapter_instance = CulturalAdapter()
    return _cultural_adapter_instance
