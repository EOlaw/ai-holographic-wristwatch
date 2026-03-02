"""Voice Synthesis subpackage — AI Holographic Wristwatch."""
from src.ai_system.conversational_ai.voice_synthesis.speech_synthesis import SpeechSynthesizer, VoiceProfile, AudioData, get_speech_synthesizer
from src.ai_system.conversational_ai.voice_synthesis.prosody_control import ProsodyController, ProsodyParams, get_prosody_controller
from src.ai_system.conversational_ai.voice_synthesis.emotion_in_speech import EmotionalSpeech, EmotionProfile, get_emotional_speech
from src.ai_system.conversational_ai.voice_synthesis.voice_personalization import VoicePersonalizer, get_voice_personalizer
from src.ai_system.conversational_ai.voice_synthesis.accent_adaptation import AccentAdapter, AccentProfile, get_accent_adapter
__all__ = ["SpeechSynthesizer", "VoiceProfile", "AudioData", "get_speech_synthesizer", "ProsodyController", "ProsodyParams", "get_prosody_controller", "EmotionalSpeech", "EmotionProfile", "get_emotional_speech", "VoicePersonalizer", "get_voice_personalizer", "AccentAdapter", "AccentProfile", "get_accent_adapter"]
