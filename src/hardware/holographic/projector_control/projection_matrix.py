"""3D projection geometry engine for the AI Holographic Wristwatch.

Computes perspective transform, keystone correction, and homography matrices
required to accurately map holographic content onto the projection surface.
"""

import threading
import time
import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectionMatrix:
    perspective: List[List[float]]
    keystone_h: float
    keystone_v: float
    homography: List[List[float]]
    fov_deg: float
    aspect_ratio: float
    near_plane_mm: float
    far_plane_mm: float
    computed_at: float = field(default_factory=time.time)


def _identity_3x3() -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]


def _identity_4x4() -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


class ProjectionMatrixEngine:
    """Perspective transform, keystone correction, and homography computation.

    All matrices are row-major. Perspective uses a right-handed coordinate
    system with the Z-axis pointing out of the projector toward the viewer.
    """

    DEFAULT_FOV_DEG = 60.0
    DEFAULT_ASPECT = 16.0 / 9.0
    DEFAULT_NEAR_MM = 50.0
    DEFAULT_FAR_MM = 500.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._fov_deg = self.DEFAULT_FOV_DEG
        self._aspect = self.DEFAULT_ASPECT
        self._near_mm = self.DEFAULT_NEAR_MM
        self._far_mm = self.DEFAULT_FAR_MM
        self._keystone_h = 0.0
        self._keystone_v = 0.0
        self._matrix: Optional[ProjectionMatrix] = None
        logger.info("ProjectionMatrixEngine initialized")

    def configure(
        self,
        fov_deg: float = DEFAULT_FOV_DEG,
        aspect_ratio: float = DEFAULT_ASPECT,
        near_mm: float = DEFAULT_NEAR_MM,
        far_mm: float = DEFAULT_FAR_MM,
    ) -> None:
        """Set projection frustum parameters."""
        with self._lock:
            self._fov_deg = fov_deg
            self._aspect = aspect_ratio
            self._near_mm = near_mm
            self._far_mm = far_mm
            logger.debug("Projection configured: FoV=%.1f° AR=%.2f near=%.0f far=%.0f",
                         fov_deg, aspect_ratio, near_mm, far_mm)

    def set_keystone(self, horizontal: float, vertical: float) -> None:
        """Set keystone correction factors (-1.0 to 1.0)."""
        with self._lock:
            self._keystone_h = max(-1.0, min(1.0, horizontal))
            self._keystone_v = max(-1.0, min(1.0, vertical))

    def compute(self) -> ProjectionMatrix:
        """Compute and return the full projection matrix set."""
        with self._lock:
            persp = self._build_perspective_matrix()
            homo = self._build_homography(self._keystone_h, self._keystone_v)
            self._matrix = ProjectionMatrix(
                perspective=persp,
                keystone_h=self._keystone_h,
                keystone_v=self._keystone_v,
                homography=homo,
                fov_deg=self._fov_deg,
                aspect_ratio=self._aspect,
                near_plane_mm=self._near_mm,
                far_plane_mm=self._far_mm,
            )
            logger.debug("Projection matrix computed")
            return self._matrix

    def get_current_matrix(self) -> Optional[ProjectionMatrix]:
        """Return the last computed ProjectionMatrix."""
        with self._lock:
            return self._matrix

    def _build_perspective_matrix(self) -> List[List[float]]:
        """Build a 4x4 perspective projection matrix from current frustum params."""
        f = 1.0 / math.tan(math.radians(self._fov_deg) / 2.0)
        n, fa = self._near_mm, self._far_mm
        m = _identity_4x4()
        m[0][0] = f / self._aspect
        m[1][1] = f
        m[2][2] = (fa + n) / (n - fa)
        m[2][3] = (2.0 * fa * n) / (n - fa)
        m[3][2] = -1.0
        m[3][3] = 0.0
        return m

    def _build_homography(self, kh: float, kv: float) -> List[List[float]]:
        """Build a 3x3 homography matrix incorporating keystone correction."""
        h = _identity_3x3()
        h[2][0] = kh * 0.002
        h[2][1] = kv * 0.002
        return h


_GLOBAL_PROJECTION_MATRIX_ENGINE: Optional[ProjectionMatrixEngine] = None
_GLOBAL_PROJECTION_MATRIX_ENGINE_LOCK = threading.Lock()


def get_projection_matrix_engine() -> ProjectionMatrixEngine:
    """Return the global ProjectionMatrixEngine singleton."""
    global _GLOBAL_PROJECTION_MATRIX_ENGINE
    with _GLOBAL_PROJECTION_MATRIX_ENGINE_LOCK:
        if _GLOBAL_PROJECTION_MATRIX_ENGINE is None:
            _GLOBAL_PROJECTION_MATRIX_ENGINE = ProjectionMatrixEngine()
    return _GLOBAL_PROJECTION_MATRIX_ENGINE


def run_projection_matrix_tests() -> None:
    """Self-test for ProjectionMatrixEngine."""
    logger.info("Running ProjectionMatrixEngine tests...")
    engine = ProjectionMatrixEngine()

    engine.configure(fov_deg=75.0, aspect_ratio=4.0 / 3.0)
    engine.set_keystone(0.1, -0.05)

    matrix = engine.compute()
    assert matrix.fov_deg == 75.0
    assert matrix.aspect_ratio == 4.0 / 3.0
    assert len(matrix.perspective) == 4
    assert len(matrix.perspective[0]) == 4
    assert matrix.perspective[3][2] == -1.0

    assert len(matrix.homography) == 3
    assert engine.get_current_matrix() is not None

    logger.info("ProjectionMatrixEngine tests PASSED")


if __name__ == "__main__":
    run_projection_matrix_tests()
