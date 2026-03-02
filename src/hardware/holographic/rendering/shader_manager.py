"""Shader program manager for the AI Holographic Wristwatch rendering pipeline.

Manages compilation, caching, and hot-reloading of holographic shader programs.
Supports standard vertex/fragment shaders and holographic-specific effects.
"""

import threading
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class ShaderType(Enum):
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    COMPUTE = "compute"
    GEOMETRY = "geometry"


class ShaderEffect(Enum):
    HOLOGRAM_SCANLINE = "hologram_scanline"
    DEPTH_FOG = "depth_fog"
    CHROMATIC_ABERRATION = "chromatic_aberration"
    FRESNEL_EDGE = "fresnel_edge"
    TRANSPARENCY = "transparency"
    STANDARD = "standard"


@dataclass
class ShaderProgram:
    program_id: str
    effect: ShaderEffect
    shader_types: List[ShaderType]
    compiled: bool
    compile_time_ms: float
    uniform_count: int
    compiled_at: float = field(default_factory=time.time)


class ShaderManager:
    """Shader compilation cache and hot-reload manager for holographic effects.

    Maintains a registry of compiled shader programs keyed by effect name.
    Simulates GPU shader compilation and validates uniform bindings.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._programs: Dict[str, ShaderProgram] = {}
        self._active_program: Optional[str] = None
        logger.info("ShaderManager initialized")

    def compile_shader(
        self,
        program_id: str,
        effect: ShaderEffect,
        shader_types: Optional[List[ShaderType]] = None,
    ) -> ShaderProgram:
        """Compile (or retrieve cached) a shader program for the given effect."""
        with self._lock:
            if program_id in self._programs:
                logger.debug("Shader cache hit: %s", program_id)
                return self._programs[program_id]

            if shader_types is None:
                shader_types = [ShaderType.VERTEX, ShaderType.FRAGMENT]

            t0 = time.monotonic()
            compiled, uniform_count = self._simulate_compile(effect, shader_types)
            elapsed = (time.monotonic() - t0) * 1000.0

            prog = ShaderProgram(
                program_id=program_id,
                effect=effect,
                shader_types=shader_types,
                compiled=compiled,
                compile_time_ms=elapsed,
                uniform_count=uniform_count,
            )
            self._programs[program_id] = prog
            logger.info("Shader compiled: %s (%s) in %.2f ms", program_id, effect.value, elapsed)
            return prog

    def bind(self, program_id: str) -> bool:
        """Bind a shader program for the next draw call. Returns False if not found."""
        with self._lock:
            if program_id not in self._programs or not self._programs[program_id].compiled:
                logger.warning("Cannot bind unknown/uncompiled shader: %s", program_id)
                return False
            self._active_program = program_id
            logger.debug("Shader bound: %s", program_id)
            return True

    def invalidate(self, program_id: str) -> None:
        """Evict a shader from the cache, forcing recompilation on next use."""
        with self._lock:
            if program_id in self._programs:
                del self._programs[program_id]
                if self._active_program == program_id:
                    self._active_program = None
                logger.info("Shader invalidated: %s", program_id)

    def precompile_all_effects(self) -> int:
        """Compile one shader program for each known effect. Returns compiled count."""
        count = 0
        for effect in ShaderEffect:
            prog = self.compile_shader(f"auto_{effect.value}", effect)
            if prog.compiled:
                count += 1
        logger.info("Precompiled %d shader programs", count)
        return count

    def get_program(self, program_id: str) -> Optional[ShaderProgram]:
        """Retrieve a compiled shader program by ID."""
        with self._lock:
            return self._programs.get(program_id)

    def _simulate_compile(
        self, effect: ShaderEffect, shader_types: List[ShaderType]
    ):
        """Simulate GPU shader compilation — returns (success, uniform_count)."""
        success = random.random() > 0.02
        uniform_count = random.randint(4, 24)
        return success, uniform_count


_GLOBAL_SHADER_MANAGER: Optional[ShaderManager] = None
_GLOBAL_SHADER_MANAGER_LOCK = threading.Lock()


def get_shader_manager() -> ShaderManager:
    """Return the global ShaderManager singleton."""
    global _GLOBAL_SHADER_MANAGER
    with _GLOBAL_SHADER_MANAGER_LOCK:
        if _GLOBAL_SHADER_MANAGER is None:
            _GLOBAL_SHADER_MANAGER = ShaderManager()
    return _GLOBAL_SHADER_MANAGER


def run_shader_manager_tests() -> None:
    """Self-test for ShaderManager."""
    logger.info("Running ShaderManager tests...")
    mgr = ShaderManager()

    prog = mgr.compile_shader("holo_main", ShaderEffect.HOLOGRAM_SCANLINE)
    assert prog.program_id == "holo_main"
    assert prog.effect == ShaderEffect.HOLOGRAM_SCANLINE

    prog2 = mgr.compile_shader("holo_main", ShaderEffect.HOLOGRAM_SCANLINE)
    assert prog2.compiled_at == prog.compiled_at, "Should return cached program"

    if prog.compiled:
        assert mgr.bind("holo_main")
    assert not mgr.bind("does_not_exist")

    count = mgr.precompile_all_effects()
    assert count > 0

    mgr.invalidate("holo_main")
    assert mgr.get_program("holo_main") is None

    logger.info("ShaderManager tests PASSED")


if __name__ == "__main__":
    run_shader_manager_tests()
