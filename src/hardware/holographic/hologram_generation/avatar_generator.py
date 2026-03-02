"""AI avatar appearance generation for the AI Holographic Wristwatch.

Generates 3D avatar mesh and appearance data from a high-level configuration.
Supports multiple visual styles and emotional expression presets.
"""

import threading
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class AvatarStyle(Enum):
    REALISTIC = "realistic"
    STYLIZED = "stylized"
    CARTOON = "cartoon"
    MINIMAL = "minimal"
    HOLOGRAPHIC_WIREFRAME = "holographic_wireframe"


class EmotionExpression(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    CONCERNED = "concerned"
    EXCITED = "excited"
    CALM = "calm"
    ATTENTIVE = "attentive"


@dataclass
class AvatarConfig:
    style: AvatarStyle
    emotion: EmotionExpression
    skin_tone: Tuple[float, float, float]
    hair_color: Tuple[float, float, float]
    eye_color: Tuple[float, float, float]
    height_m: float = 0.30
    enable_physics_hair: bool = False


@dataclass
class AvatarFrame:
    config: AvatarConfig
    vertex_count: int
    triangle_count: int
    texture_atlas_id: str
    blend_shape_weights: Dict[str, float]
    lod_level: int
    generation_time_ms: float
    generated_at: float = field(default_factory=time.time)


class AvatarGenerator:
    """Generates 3D avatar mesh from AvatarConfig using procedural techniques.

    Mesh complexity adapts to LOD level. Blend shapes encode expression weights
    for real-time expression control without regenerating full geometry.
    """

    LOD_VERTEX_COUNTS = [500, 2_000, 8_000, 25_000]

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._lod_level = 2
        self._last_frame: Optional[AvatarFrame] = None
        logger.info("AvatarGenerator initialized")

    def set_lod(self, lod_level: int) -> None:
        """Set Level-of-Detail (0 = lowest, 3 = highest)."""
        with self._lock:
            self._lod_level = max(0, min(3, lod_level))

    def generate(self, config: AvatarConfig) -> AvatarFrame:
        """Generate a full avatar frame from the given configuration."""
        with self._lock:
            t0 = time.monotonic()
            vertex_count = self._simulate_mesh_generation(config)
            triangle_count = vertex_count * 2 // 3
            blend_shapes = self._generate_blend_shapes(config.emotion)
            atlas_id = self._generate_texture_atlas(config)
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            frame = AvatarFrame(
                config=config,
                vertex_count=vertex_count,
                triangle_count=triangle_count,
                texture_atlas_id=atlas_id,
                blend_shape_weights=blend_shapes,
                lod_level=self._lod_level,
                generation_time_ms=elapsed_ms,
            )
            self._last_frame = frame
            logger.debug("Avatar generated: %d verts, %d tris in %.2f ms",
                         vertex_count, triangle_count, elapsed_ms)
            return frame

    def get_last_frame(self) -> Optional[AvatarFrame]:
        """Return the most recently generated avatar frame."""
        with self._lock:
            return self._last_frame

    def _simulate_mesh_generation(self, config: AvatarConfig) -> int:
        """Simulate procedural mesh generation — returns vertex count."""
        base = self.LOD_VERTEX_COUNTS[self._lod_level]
        style_mult = {
            AvatarStyle.MINIMAL: 0.3,
            AvatarStyle.CARTOON: 0.5,
            AvatarStyle.STYLIZED: 0.8,
            AvatarStyle.REALISTIC: 1.0,
            AvatarStyle.HOLOGRAPHIC_WIREFRAME: 0.4,
        }[config.style]
        return int(base * style_mult * random.uniform(0.9, 1.1))

    def _generate_blend_shapes(self, emotion: EmotionExpression) -> Dict[str, float]:
        """Produce expression blend shape weights for the given emotion."""
        presets = {
            EmotionExpression.NEUTRAL:   {"brow_raise": 0.0, "smile": 0.0, "eye_wide": 0.0},
            EmotionExpression.HAPPY:     {"brow_raise": 0.2, "smile": 0.8, "eye_wide": 0.1},
            EmotionExpression.CONCERNED: {"brow_raise": 0.0, "smile": 0.0, "brow_furrow": 0.6},
            EmotionExpression.EXCITED:   {"brow_raise": 0.7, "smile": 0.9, "eye_wide": 0.5},
            EmotionExpression.CALM:      {"brow_raise": 0.0, "smile": 0.2, "eye_wide": 0.0},
            EmotionExpression.ATTENTIVE: {"brow_raise": 0.3, "smile": 0.1, "eye_wide": 0.3},
        }
        return {k: v + random.uniform(-0.05, 0.05) for k, v in presets[emotion].items()}

    def _generate_texture_atlas(self, config: AvatarConfig) -> str:
        """Return a texture atlas identifier based on style and skin tone."""
        r, g, b = (int(c * 255) for c in config.skin_tone)
        return f"atlas_{config.style.value}_{r:02x}{g:02x}{b:02x}"


_GLOBAL_AVATAR_GENERATOR: Optional[AvatarGenerator] = None
_GLOBAL_AVATAR_GENERATOR_LOCK = threading.Lock()


def get_avatar_generator() -> AvatarGenerator:
    """Return the global AvatarGenerator singleton."""
    global _GLOBAL_AVATAR_GENERATOR
    with _GLOBAL_AVATAR_GENERATOR_LOCK:
        if _GLOBAL_AVATAR_GENERATOR is None:
            _GLOBAL_AVATAR_GENERATOR = AvatarGenerator()
    return _GLOBAL_AVATAR_GENERATOR


def run_avatar_generator_tests() -> None:
    """Self-test for AvatarGenerator."""
    logger.info("Running AvatarGenerator tests...")
    gen = AvatarGenerator()

    config = AvatarConfig(
        style=AvatarStyle.STYLIZED,
        emotion=EmotionExpression.HAPPY,
        skin_tone=(0.8, 0.65, 0.55),
        hair_color=(0.2, 0.1, 0.05),
        eye_color=(0.1, 0.4, 0.8),
    )
    gen.set_lod(1)
    frame = gen.generate(config)

    assert frame.vertex_count > 0
    assert frame.triangle_count > 0
    assert "smile" in frame.blend_shape_weights
    assert frame.lod_level == 1
    assert gen.get_last_frame() is frame

    logger.info("AvatarGenerator tests PASSED (%d verts)", frame.vertex_count)


if __name__ == "__main__":
    run_avatar_generator_tests()
