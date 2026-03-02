"""Rendering subpackage for the AI Holographic Wristwatch.

Exports quality adaptation, real-time processing, scene composition,
shader management, and volumetric rendering components.
"""

from src.hardware.holographic.rendering.quality_adaptation import (
    RenderQuality,
    QualityState,
    QualityAdaptationEngine,
    get_quality_adaptation_engine,
)
from src.hardware.holographic.rendering.real_time_processing import (
    ProcessingStage,
    ProcessingMode,
    FrameStats,
    RealTimeProcessor,
    get_real_time_processor,
)
from src.hardware.holographic.rendering.scene_composer import (
    LayerType,
    BlendMode,
    SceneObject,
    SceneComposition,
    SceneComposer,
    get_scene_composer,
)
from src.hardware.holographic.rendering.shader_manager import (
    ShaderType,
    ShaderEffect,
    ShaderProgram,
    ShaderManager,
    get_shader_manager,
)
from src.hardware.holographic.rendering.volume_renderer import (
    VolumeType,
    SamplingQuality,
    VolumeLayer,
    VolumeRenderStats,
    VolumeRenderer,
    get_volume_renderer,
)

__all__ = [
    "RenderQuality", "QualityState", "QualityAdaptationEngine", "get_quality_adaptation_engine",
    "ProcessingStage", "ProcessingMode", "FrameStats", "RealTimeProcessor",
    "get_real_time_processor",
    "LayerType", "BlendMode", "SceneObject", "SceneComposition", "SceneComposer",
    "get_scene_composer",
    "ShaderType", "ShaderEffect", "ShaderProgram", "ShaderManager", "get_shader_manager",
    "VolumeType", "SamplingQuality", "VolumeLayer", "VolumeRenderStats", "VolumeRenderer",
    "get_volume_renderer",
]
