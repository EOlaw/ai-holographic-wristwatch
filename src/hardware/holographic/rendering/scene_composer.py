"""Scene composer for the AI Holographic Wristwatch rendering pipeline.

Assembles holographic scene graphs from individual render objects, manages
draw order, visibility culling, and layer compositing for final output.
"""

import threading
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class LayerType(Enum):
    BACKGROUND = "background"
    AVATAR = "avatar"
    UI_OVERLAY = "ui_overlay"
    PARTICLE = "particle"
    DEBUG = "debug"


class BlendMode(Enum):
    OPAQUE = "opaque"
    ALPHA_BLEND = "alpha_blend"
    ADDITIVE = "additive"
    MULTIPLY = "multiply"


@dataclass
class SceneObject:
    object_id: str
    layer: LayerType
    blend_mode: BlendMode
    position: tuple
    scale: float
    visible: bool
    draw_order: int


@dataclass
class SceneComposition:
    object_count: int
    visible_count: int
    layer_counts: Dict[str, int]
    estimated_draw_calls: int
    composed_at: float = field(default_factory=time.time)


class SceneComposer:
    """Holographic scene graph manager with layer compositing and draw call batching.

    Maintains an ordered set of SceneObjects. Per frame, performs frustum
    culling (simulated) and produces a sorted draw list for the renderer.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._objects: Dict[str, SceneObject] = {}
        self._next_draw_order = 0
        logger.info("SceneComposer initialized")

    def add_object(
        self,
        object_id: str,
        layer: LayerType = LayerType.AVATAR,
        blend_mode: BlendMode = BlendMode.ALPHA_BLEND,
        position: tuple = (0.0, 0.0, 0.0),
        scale: float = 1.0,
    ) -> SceneObject:
        """Add or replace a scene object."""
        with self._lock:
            obj = SceneObject(
                object_id=object_id,
                layer=layer,
                blend_mode=blend_mode,
                position=position,
                scale=scale,
                visible=True,
                draw_order=self._next_draw_order,
            )
            self._next_draw_order += 1
            self._objects[object_id] = obj
            logger.debug("Scene object added: %s (layer=%s)", object_id, layer.value)
            return obj

    def remove_object(self, object_id: str) -> bool:
        """Remove a scene object by ID. Returns False if not found."""
        with self._lock:
            if object_id in self._objects:
                del self._objects[object_id]
                logger.debug("Scene object removed: %s", object_id)
                return True
            return False

    def set_visibility(self, object_id: str, visible: bool) -> None:
        """Toggle visibility of a scene object."""
        with self._lock:
            if object_id in self._objects:
                self._objects[object_id].visible = visible

    def compose(self) -> SceneComposition:
        """Cull invisible objects and produce the composited scene description."""
        with self._lock:
            visible = [o for o in self._objects.values() if o.visible]
            visible_culled = self._simulate_frustum_cull(visible)
            layer_counts: Dict[str, int] = {lt.value: 0 for lt in LayerType}
            for obj in visible_culled:
                layer_counts[obj.layer.value] += 1
            draw_calls = self._estimate_draw_calls(visible_culled)
            return SceneComposition(
                object_count=len(self._objects),
                visible_count=len(visible_culled),
                layer_counts=layer_counts,
                estimated_draw_calls=draw_calls,
            )

    def get_sorted_draw_list(self) -> List[SceneObject]:
        """Return visible objects sorted by layer then draw order."""
        layer_order = list(LayerType)
        with self._lock:
            visible = [o for o in self._objects.values() if o.visible]
            return sorted(visible, key=lambda o: (layer_order.index(o.layer), o.draw_order))

    def _simulate_frustum_cull(self, objects: List[SceneObject]) -> List[SceneObject]:
        """Simulate frustum culling — randomly removes ~10% of objects."""
        return [o for o in objects if random.random() > 0.1]

    def _estimate_draw_calls(self, objects: List[SceneObject]) -> int:
        """Estimate GPU draw calls based on batching by layer and blend mode."""
        batches = set((o.layer, o.blend_mode) for o in objects)
        return len(batches)


_GLOBAL_SCENE_COMPOSER: Optional[SceneComposer] = None
_GLOBAL_SCENE_COMPOSER_LOCK = threading.Lock()


def get_scene_composer() -> SceneComposer:
    """Return the global SceneComposer singleton."""
    global _GLOBAL_SCENE_COMPOSER
    with _GLOBAL_SCENE_COMPOSER_LOCK:
        if _GLOBAL_SCENE_COMPOSER is None:
            _GLOBAL_SCENE_COMPOSER = SceneComposer()
    return _GLOBAL_SCENE_COMPOSER


def run_scene_composer_tests() -> None:
    """Self-test for SceneComposer."""
    logger.info("Running SceneComposer tests...")
    composer = SceneComposer()

    obj = composer.add_object("avatar_01", layer=LayerType.AVATAR)
    assert obj.visible
    composer.add_object("ui_panel", layer=LayerType.UI_OVERLAY, blend_mode=BlendMode.ALPHA_BLEND)

    comp = composer.compose()
    assert comp.object_count == 2

    composer.set_visibility("avatar_01", False)
    draw_list = composer.get_sorted_draw_list()
    assert all(o.object_id != "avatar_01" for o in draw_list)

    assert composer.remove_object("ui_panel")
    assert not composer.remove_object("nonexistent")

    logger.info("SceneComposer tests PASSED")


if __name__ == "__main__":
    run_scene_composer_tests()
