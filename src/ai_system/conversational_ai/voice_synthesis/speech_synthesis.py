"""Speech synthesis module for the AI Holographic Wristwatch.

Provides a simulated text-to-speech pipeline. Because the wristwatch does not
run a full TTS neural network in this implementation layer, speech is
represented as a :class:`AudioData` object whose ``samples`` field contains a
sine-wave approximation of the spoken waveform at the voice's characteristic
fundamental frequency. Downstream hardware / firmware replaces this with real
samples at deployment time.

Thread-safe singleton available via :func:`get_speech_synthesizer`.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VoiceProfile:
    """Describes the characteristics of a synthesiser voice."""
    id: str
    name: str
    gender: str          # "female", "male", "neutral"
    language: str        # BCP-47 language tag, e.g. "en-US"
    pitch: float         # Fundamental frequency in Hz (e.g. 180.0 for a mid female voice)
    speed: float         # Default speaking rate multiplier (1.0 = normal)
    timbre: str          # Qualitative descriptor: "warm", "crisp", "smooth", "bright"

    def __repr__(self) -> str:
        return (
            f"VoiceProfile(id={self.id!r}, name={self.name!r}, "
            f"lang={self.language!r}, pitch={self.pitch}Hz)"
        )


@dataclass
class AudioData:
    """PCM audio data produced by the synthesiser."""
    samples: List[float]
    sample_rate: int = 22_050
    duration_secs: float = 0.0
    format: str = "wav"
    channels: int = 1

    def __repr__(self) -> str:
        return (
            f"AudioData(duration={self.duration_secs:.2f}s, "
            f"sample_rate={self.sample_rate}Hz, "
            f"samples={len(self.samples)})"
        )


@dataclass
class SynthesisRequest:
    """All parameters needed to synthesise a single utterance."""
    text: str
    voice_id: str = "voice_aria"
    pitch_adjust: float = 0.0     # Semitone offset from voice's base pitch
    speed: float = 1.0            # Speaking rate multiplier
    volume: float = 1.0           # Output amplitude multiplier [0.0, 2.0]
    priority: int = 1             # 1 = normal, 2 = high, 3 = urgent


# ---------------------------------------------------------------------------
# Voice catalogue
# ---------------------------------------------------------------------------

_AVAILABLE_VOICES: List[VoiceProfile] = [
    VoiceProfile(
        id="voice_aria",
        name="Aria",
        gender="female",
        language="en-US",
        pitch=210.0,
        speed=1.0,
        timbre="warm",
    ),
    VoiceProfile(
        id="voice_nova",
        name="Nova",
        gender="female",
        language="en-GB",
        pitch=190.0,
        speed=0.95,
        timbre="smooth",
    ),
    VoiceProfile(
        id="voice_echo",
        name="Echo",
        gender="male",
        language="en-US",
        pitch=120.0,
        speed=1.05,
        timbre="crisp",
    ),
    VoiceProfile(
        id="voice_orion",
        name="Orion",
        gender="male",
        language="en-GB",
        pitch=110.0,
        speed=0.90,
        timbre="bright",
    ),
    VoiceProfile(
        id="voice_sage",
        name="Sage",
        gender="neutral",
        language="en-US",
        pitch=155.0,
        speed=1.0,
        timbre="warm",
    ),
]

# Approximate syllables per second at speaking rate 1.0.
_SYLLABLES_PER_SECOND = 3.8
# Rough average syllable count per word.
_SYLLABLES_PER_WORD = 1.5
# Characters per word (for char-based estimation).
_CHARS_PER_WORD = 5.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpeechSynthesizer:
    """Converts text to simulated PCM audio using a wristwatch voice profile.

    The generated audio is a sine wave at the voice's fundamental frequency,
    amplitude-modulated by a simple exponential envelope to simulate natural
    onset and offset. Real deployment replaces this with a trained TTS model.

    Thread-safe; a shared singleton is available via
    :func:`get_speech_synthesizer`.
    """

    AVAILABLE_VOICES: List[VoiceProfile] = _AVAILABLE_VOICES

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current_voice: VoiceProfile = _AVAILABLE_VOICES[0]
        self._synthesis_count: int = 0
        self._total_duration: float = 0.0
        self._start_time: float = time.monotonic()
        logger.debug(
            "SpeechSynthesizer initialised with voice %r.", self._current_voice.id
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, request: SynthesisRequest) -> AudioData:
        """Synthesise speech for *request*.

        Generates a sine-wave approximation at the selected voice's base pitch
        (adjusted by ``pitch_adjust``). The number of samples corresponds to
        the estimated utterance duration.

        Args:
            request: A :class:`SynthesisRequest` specifying text, voice, and
                rendering parameters.

        Returns:
            An :class:`AudioData` object containing the PCM samples.
        """
        # Resolve voice.
        voice = self._get_voice_by_id(request.voice_id) or self._current_voice

        # Effective speaking speed.
        effective_speed = max(0.5, min(3.0, voice.speed * request.speed))

        # Estimate duration.
        duration = self.estimate_duration(request.text, effective_speed)

        # Compute fundamental frequency (pitch adjust in semitones → Hz).
        freq_hz = voice.pitch * (2 ** (request.pitch_adjust / 12.0))
        freq_hz = max(80.0, min(400.0, freq_hz))

        sample_rate = 22_050
        n_samples = int(duration * sample_rate)

        # Generate sine-wave samples with ADSR-like envelope.
        volume = max(0.0, min(2.0, request.volume))
        samples = self._generate_sine_wave(freq_hz, sample_rate, n_samples, volume)

        with self._lock:
            self._synthesis_count += 1
            self._total_duration += duration

        logger.debug(
            "synthesize() #%d: voice=%r text_len=%d duration=%.2fs samples=%d",
            self._synthesis_count,
            voice.id,
            len(request.text),
            duration,
            n_samples,
        )

        return AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration_secs=round(duration, 3),
            format="wav",
            channels=1,
        )

    def set_voice(self, voice_id: str) -> bool:
        """Switch the default voice to the one identified by *voice_id*.

        Returns:
            ``True`` if the voice was found and set, ``False`` otherwise.
        """
        voice = self._get_voice_by_id(voice_id)
        if voice is None:
            logger.warning("Voice ID %r not found; keeping current voice.", voice_id)
            return False
        with self._lock:
            self._current_voice = voice
        logger.info("Default voice switched to %r.", voice_id)
        return True

    def get_available_voices(self) -> List[VoiceProfile]:
        """Return the list of all available voice profiles."""
        return list(self.AVAILABLE_VOICES)

    def estimate_duration(self, text: str, speed: float = 1.0) -> float:
        """Estimate the speaking duration (in seconds) for *text*.

        Uses a word-count heuristic:
        ``duration = (word_count * syllables_per_word) / (syllables_per_second * speed)``

        Args:
            text: The text to estimate.
            speed: Speaking rate multiplier.

        Returns:
            Estimated duration in seconds (minimum 0.2 s).
        """
        speed = max(0.5, speed)
        word_count = max(1, len(text.split()))
        syllable_count = word_count * _SYLLABLES_PER_WORD
        raw_duration = syllable_count / (_SYLLABLES_PER_SECOND * speed)
        return max(0.2, raw_duration)

    def get_stats(self) -> Dict:
        """Return runtime statistics for this synthesiser instance."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return {
                "synthesis_count": self._synthesis_count,
                "total_duration_secs": round(self._total_duration, 2),
                "current_voice": self._current_voice.id,
                "available_voices": len(self.AVAILABLE_VOICES),
                "uptime_seconds": round(elapsed, 2),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_voice_by_id(self, voice_id: str) -> Optional[VoiceProfile]:
        """Look up a voice profile by ID."""
        for v in self.AVAILABLE_VOICES:
            if v.id == voice_id:
                return v
        return None

    def _generate_sine_wave(
        self,
        freq_hz: float,
        sample_rate: int,
        n_samples: int,
        volume: float,
    ) -> List[float]:
        """Generate a sine wave with a simple ADSR envelope.

        The attack is 5 ms, decay 10 ms, sustain at 0.8 amplitude,
        release 30 ms. This produces a more natural sound than a raw sine.
        """
        attack_samples = min(int(0.005 * sample_rate), n_samples)
        decay_samples  = min(int(0.010 * sample_rate), n_samples - attack_samples)
        release_samples = min(int(0.030 * sample_rate), n_samples)
        sustain_level = 0.8

        samples: List[float] = []
        two_pi_f_over_sr = 2.0 * math.pi * freq_hz / sample_rate

        for i in range(n_samples):
            # Envelope amplitude.
            if i < attack_samples:
                env = i / attack_samples if attack_samples > 0 else 1.0
            elif i < attack_samples + decay_samples:
                t = (i - attack_samples) / decay_samples if decay_samples > 0 else 0.0
                env = 1.0 - t * (1.0 - sustain_level)
            elif i >= n_samples - release_samples:
                t = (n_samples - i) / release_samples if release_samples > 0 else 0.0
                env = sustain_level * t
            else:
                env = sustain_level

            sample_value = volume * env * math.sin(two_pi_f_over_sr * i)
            samples.append(round(sample_value, 6))

        return samples


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_synthesizer_instance: Optional[SpeechSynthesizer] = None
_synthesizer_lock = threading.Lock()


def get_speech_synthesizer() -> SpeechSynthesizer:
    """Return the module-level :class:`SpeechSynthesizer` singleton.

    Thread-safe; the instance is created lazily on first call.
    """
    global _synthesizer_instance
    if _synthesizer_instance is None:
        with _synthesizer_lock:
            if _synthesizer_instance is None:
                _synthesizer_instance = SpeechSynthesizer()
    return _synthesizer_instance


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    synth = get_speech_synthesizer()

    print("=== SpeechSynthesizer Demo ===\n")
    print("Available voices:")
    for v in synth.get_available_voices():
        print(f"  {v.id:<18} {v.name:<8} {v.gender:<8} {v.language}  "
              f"{v.pitch}Hz  timbre={v.timbre}")

    requests = [
        SynthesisRequest(text="Your heart rate is 78 bpm.", voice_id="voice_aria"),
        SynthesisRequest(text="Timer set for ten minutes.", voice_id="voice_echo", speed=1.2),
        SynthesisRequest(text="Good morning! How are you feeling today?",
                         voice_id="voice_nova", pitch_adjust=2.0),
        SynthesisRequest(text="Battery low — please charge your watch.",
                         voice_id="voice_orion", volume=1.5),
    ]

    print("\nSynthesis results:")
    for req in requests:
        audio = synth.synthesize(req)
        est = synth.estimate_duration(req.text, req.speed)
        print(
            f"  voice={req.voice_id:<18} text={req.text[:45]!r:<50} "
            f"duration={audio.duration_secs:.2f}s  "
            f"samples={len(audio.samples)}"
        )

    print("\nStats:", synth.get_stats())
