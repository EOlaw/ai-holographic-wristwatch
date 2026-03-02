"""
Audio Sensor Package — AI Holographic Wristwatch

Exports all audio processing drivers:
- Microphone array (3-mic beamforming, VAD, DOA)
- Noise cancellation (feedforward ANC, spectral subtraction)
- Acoustic processing (MFCC, pitch, emotion, audio classification)
- Speaker identification (voiceprint, verification, diarization)
- Voice isolation (echo cancellation, wind suppression, speaker focus)
"""

from .microphone_array import (
    AudioQuality, SoundEvent,
    AudioFrame, MicArrayReading,
    VoiceActivityDetector,
    MicrophoneArray, get_microphone_array,
    run_microphone_array_tests,
    SAMPLE_RATE, FRAME_SIZE, MIC_COUNT,
)

from .noise_cancellation import (
    ANCMode, NoiseType,
    ANCProcessingResult,
    SpectralSubtractor, AdaptiveANCFilter,
    NoiseCancellation, get_noise_cancellation,
    run_noise_cancellation_tests,
)

from .acoustic_processing import (
    AudioClass, VocalEmotion,
    AcousticFeatures, AcousticProcessingReading,
    MFCCExtractor, PitchEstimator, SpectralAnalyzer,
    AcousticProcessor, get_acoustic_processor,
    run_acoustic_processing_tests,
    N_MFCC,
)

from .speaker_identification import (
    VerificationDecision, SpeakerState,
    SpeakerProfile, SpeakerSegment, SpeakerIdReading,
    DVectorExtractor, AntiSpoofingDetector,
    SpeakerIdentifier, get_speaker_identifier,
    run_speaker_identification_tests,
    EMBEDDING_DIM,
)

from .voice_isolation import (
    IsolationMode, EchoCancellerState,
    VoiceIsolationReading,
    AcousticEchoCanceller, WindNoiseDetector,
    VoiceIsolator, get_voice_isolator,
    run_voice_isolation_tests,
)

__version__ = "1.0.0"
__all__ = [
    # Microphone Array
    "AudioQuality", "SoundEvent", "AudioFrame", "MicArrayReading",
    "VoiceActivityDetector", "MicrophoneArray", "get_microphone_array",
    "run_microphone_array_tests", "SAMPLE_RATE", "FRAME_SIZE", "MIC_COUNT",
    # Noise Cancellation
    "ANCMode", "NoiseType", "ANCProcessingResult",
    "SpectralSubtractor", "AdaptiveANCFilter",
    "NoiseCancellation", "get_noise_cancellation", "run_noise_cancellation_tests",
    # Acoustic Processing
    "AudioClass", "VocalEmotion", "AcousticFeatures", "AcousticProcessingReading",
    "MFCCExtractor", "PitchEstimator", "SpectralAnalyzer",
    "AcousticProcessor", "get_acoustic_processor", "run_acoustic_processing_tests",
    "N_MFCC",
    # Speaker Identification
    "VerificationDecision", "SpeakerState", "SpeakerProfile",
    "SpeakerSegment", "SpeakerIdReading",
    "DVectorExtractor", "AntiSpoofingDetector",
    "SpeakerIdentifier", "get_speaker_identifier",
    "run_speaker_identification_tests", "EMBEDDING_DIM",
    # Voice Isolation
    "IsolationMode", "EchoCancellerState", "VoiceIsolationReading",
    "AcousticEchoCanceller", "WindNoiseDetector",
    "VoiceIsolator", "get_voice_isolator", "run_voice_isolation_tests",
]
