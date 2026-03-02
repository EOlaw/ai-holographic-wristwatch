"""Real-time frame processing pipeline for the AI Holographic Wristwatch renderer.

Manages per-frame processing stages: geometry transform, rasterization, and
post-processing. Tracks frame timing statistics for adaptive quality feedback.
"""

import threading
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Deque
from collections import deque

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingStage(Enum):
    IDLE = "idle"
    GEOMETRY = "geometry"
    RASTERIZE = "rasterize"
    POST_PROCESS = "post_process"
    PRESENT = "present"


class ProcessingMode(Enum):
    SINGLE_BUFFER = "single_buffer"
    DOUBLE_BUFFER = "double_buffer"
    TRIPLE_BUFFER = "triple_buffer"


@dataclass
class FrameStats:
    frame_number: int
    stage_times_ms: dict
    total_ms: float
    fps: float
    dropped: bool
    timestamp: float = field(default_factory=time.time)


class RealTimeProcessor:
    """Per-frame real-time holographic rendering pipeline.

    Executes the geometry, rasterization, and post-processing stages for each
    frame and maintains a sliding window of frame timing statistics. Supports
    single, double, and triple buffering strategies.
    """

    HISTORY_SIZE = 60

    def __init__(self, mode: ProcessingMode = ProcessingMode.DOUBLE_BUFFER) -> None:
        self._lock = threading.RLock()
        self._mode = mode
        self._frame_number = 0
        self._stage = ProcessingStage.IDLE
        self._history: Deque[FrameStats] = deque(maxlen=self.HISTORY_SIZE)
        self._budget_ms = 16.7
        logger.info("RealTimeProcessor initialized (%s)", mode.value)

    def set_frame_budget(self, budget_ms: float) -> None:
        """Set frame time budget in milliseconds."""
        with self._lock:
            self._budget_ms = budget_ms

    def process_frame(self, scene_complexity: float = 1.0) -> FrameStats:
        """Process one holographic frame through all pipeline stages."""
        with self._lock:
            self._frame_number += 1
            stage_times: dict = {}
            t0 = time.monotonic()

            self._stage = ProcessingStage.GEOMETRY
            stage_times["geometry"] = self._simulate_geometry_pass(scene_complexity)

            self._stage = ProcessingStage.RASTERIZE
            stage_times["rasterize"] = self._simulate_rasterize_pass(scene_complexity)

            self._stage = ProcessingStage.POST_PROCESS
            stage_times["post_process"] = self._simulate_post_process_pass()

            self._stage = ProcessingStage.PRESENT
            stage_times["present"] = self._simulate_present()

            total_ms = (time.monotonic() - t0) * 1000.0
            dropped = total_ms > self._budget_ms
            if dropped:
                logger.debug("Frame %d dropped: %.1f ms > budget %.1f ms",
                             self._frame_number, total_ms, self._budget_ms)

            avg_fps = self._calculate_avg_fps(total_ms)
            stats = FrameStats(
                frame_number=self._frame_number,
                stage_times_ms=stage_times,
                total_ms=total_ms,
                fps=avg_fps,
                dropped=dropped,
            )
            self._history.append(stats)
            self._stage = ProcessingStage.IDLE
            return stats

    def get_average_fps(self) -> float:
        """Return rolling average FPS over recent frames."""
        with self._lock:
            if not self._history:
                return 0.0
            avg_ms = sum(s.total_ms for s in self._history) / len(self._history)
            return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    def _calculate_avg_fps(self, latest_ms: float) -> float:
        recent = list(self._history)[-10:] if self._history else []
        times = [s.total_ms for s in recent] + [latest_ms]
        avg = sum(times) / len(times)
        return 1000.0 / avg if avg > 0 else 0.0

    def _simulate_geometry_pass(self, complexity: float) -> float:
        """Simulate geometry transform pass timing."""
        return random.uniform(1.0, 4.0) * complexity

    def _simulate_rasterize_pass(self, complexity: float) -> float:
        """Simulate rasterization pass timing."""
        return random.uniform(2.0, 8.0) * complexity

    def _simulate_post_process_pass(self) -> float:
        """Simulate post-processing (bloom, tone-map) pass timing."""
        return random.uniform(0.5, 2.0)

    def _simulate_present(self) -> float:
        """Simulate frame presentation/flip timing."""
        return random.uniform(0.1, 0.5)


_GLOBAL_REAL_TIME_PROCESSOR: Optional[RealTimeProcessor] = None
_GLOBAL_REAL_TIME_PROCESSOR_LOCK = threading.Lock()


def get_real_time_processor() -> RealTimeProcessor:
    """Return the global RealTimeProcessor singleton."""
    global _GLOBAL_REAL_TIME_PROCESSOR
    with _GLOBAL_REAL_TIME_PROCESSOR_LOCK:
        if _GLOBAL_REAL_TIME_PROCESSOR is None:
            _GLOBAL_REAL_TIME_PROCESSOR = RealTimeProcessor()
    return _GLOBAL_REAL_TIME_PROCESSOR


def run_real_time_processing_tests() -> None:
    """Self-test for RealTimeProcessor."""
    logger.info("Running RealTimeProcessor tests...")
    proc = RealTimeProcessor(ProcessingMode.TRIPLE_BUFFER)
    proc.set_frame_budget(16.7)

    for _ in range(5):
        stats = proc.process_frame(scene_complexity=0.5)
        assert stats.frame_number > 0
        assert stats.total_ms > 0.0
        assert isinstance(stats.dropped, bool)

    fps = proc.get_average_fps()
    assert fps > 0.0, "FPS should be positive"

    logger.info("RealTimeProcessor tests PASSED (avg %.1f fps)", fps)


if __name__ == "__main__":
    run_real_time_processing_tests()
