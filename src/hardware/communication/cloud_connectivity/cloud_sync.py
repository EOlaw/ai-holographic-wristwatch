"""
Cloud Sync — AI Holographic Wristwatch

Synchronises watch data (health records, settings, contacts, media) with the
cloud using an eventually-consistent model.  Queued sync jobs are processed
by a background worker; conflict resolution uses a "last-writer-wins with
vector-clock tiebreaker" strategy.
"""

from __future__ import annotations

import json
import math
import queue
import random
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SyncStatus(Enum):
    IDLE = "idle"
    QUEUED = "queued"
    SYNCING = "syncing"
    COMPLETED = "completed"
    CONFLICT = "conflict"
    FAILED = "failed"
    PAUSED = "paused"


class SyncDirection(Enum):
    UPLOAD = "upload"
    DOWNLOAD = "download"
    BIDIRECTIONAL = "bidirectional"


class ConflictResolution(Enum):
    LAST_WRITER_WINS = "last_writer_wins"
    SERVER_WINS = "server_wins"
    CLIENT_WINS = "client_wins"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class VectorClock:
    """Logical clock for distributed conflict detection."""
    clocks: Dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> None:
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1

    def merge(self, other: "VectorClock") -> "VectorClock":
        merged = dict(self.clocks)
        for node, tick in other.clocks.items():
            merged[node] = max(merged.get(node, 0), tick)
        return VectorClock(clocks=merged)

    def happens_before(self, other: "VectorClock") -> bool:
        """Return True if self < other in causal order."""
        all_nodes = set(self.clocks) | set(other.clocks)
        dominated = False
        for node in all_nodes:
            if self.clocks.get(node, 0) > other.clocks.get(node, 0):
                return False
            if self.clocks.get(node, 0) < other.clocks.get(node, 0):
                dominated = True
        return dominated


@dataclass
class SyncJob:
    job_id: str
    data_type: str           # e.g. "health_records", "settings", "contacts"
    direction: SyncDirection
    payload: Dict[str, Any]
    vector_clock: VectorClock
    priority: int = 5        # 1 (highest) – 10 (lowest)
    status: SyncStatus = SyncStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class SyncConflict:
    conflict_id: str
    data_type: str
    local_clock: VectorClock
    remote_clock: VectorClock
    local_value: Any
    remote_value: Any
    detected_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[ConflictResolution] = None
    winning_value: Any = None


@dataclass
class SyncStatusReport:
    pending_jobs: int
    completed_today: int
    failed_today: int
    unresolved_conflicts: int
    last_sync_time: Optional[float]
    next_sync_time: Optional[float]
    bytes_uploaded_today: int
    bytes_downloaded_today: int


# ---------------------------------------------------------------------------
# CloudSync
# ---------------------------------------------------------------------------

class CloudSync:
    """
    Eventually-consistent cloud synchronisation engine.

    - Maintains a priority queue of pending SyncJobs.
    - Worker thread processes jobs at configurable throughput.
    - Conflict detection via vector clocks; resolution pluggable.
    - Exponential back-off on transient failures (max 3 retries).
    """

    _SIMULATED_BANDWIDTH_KBPS = 512.0
    _WORKER_POLL_INTERVAL_S = 0.5

    def __init__(
        self,
        node_id: Optional[str] = None,
        conflict_strategy: ConflictResolution = ConflictResolution.LAST_WRITER_WINS,
    ) -> None:
        self._node_id = node_id or f"watch-{uuid.uuid4().hex[:8]}"
        self._conflict_strategy = conflict_strategy
        self._lock = threading.RLock()
        self._job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._jobs: Dict[str, SyncJob] = {}
        self._conflicts: Dict[str, SyncConflict] = {}
        self._vector_clock = VectorClock()
        self._cloud_state: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._stats: Dict[str, Any] = {
            "completed_today": 0,
            "failed_today": 0,
            "bytes_up": 0,
            "bytes_down": 0,
            "last_sync": None,
        }
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="cloud-sync-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("CloudSync initialised for node %s", self._node_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def sync_data(
        self,
        data_type: str,
        payload: Dict[str, Any],
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        priority: int = 5,
    ) -> str:
        """
        Immediately enqueue a sync job and block until it completes (or fails).

        Returns the job_id of the resulting SyncJob.
        """
        job_id = self.queue_sync(data_type, payload, direction, priority)
        deadline = time.time() + 30.0
        while time.time() < deadline:
            job = self.get_sync_status(job_id)
            if job and job.status in (SyncStatus.COMPLETED, SyncStatus.FAILED):
                break
            time.sleep(0.1)
        return job_id

    def queue_sync(
        self,
        data_type: str,
        payload: Dict[str, Any],
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        priority: int = 5,
    ) -> str:
        """Enqueue a sync job and return its job_id without blocking."""
        with self._lock:
            self._vector_clock.increment(self._node_id)
            vc = VectorClock(clocks=dict(self._vector_clock.clocks))

        job = SyncJob(
            job_id=str(uuid.uuid4()),
            data_type=data_type,
            direction=direction,
            payload=payload,
            vector_clock=vc,
            priority=priority,
            status=SyncStatus.QUEUED,
        )

        with self._lock:
            self._jobs[job.job_id] = job

        # PriorityQueue uses (priority, tie-breaker, job) tuples
        self._job_queue.put((priority, time.time(), job))
        logger.debug("Queued sync job %s type=%s", job.job_id[:8], data_type)
        return job.job_id

    def get_sync_status(self, job_id: str) -> Optional[SyncJob]:
        """Return the current SyncJob record for the given job_id."""
        with self._lock:
            return self._jobs.get(job_id)

    def resolve_conflicts(
        self,
        conflict_id: str,
        resolution: ConflictResolution,
        chosen_value: Optional[Any] = None,
    ) -> bool:
        """
        Manually resolve a detected conflict.

        For MANUAL resolution, caller must supply chosen_value.
        Returns True on success.
        """
        with self._lock:
            conflict = self._conflicts.get(conflict_id)
            if conflict is None or conflict.resolved:
                return False

            if resolution == ConflictResolution.LOCAL_WINS:  # type: ignore[comparison-overlap]
                conflict.winning_value = conflict.local_value
            elif resolution == ConflictResolution.SERVER_WINS:
                conflict.winning_value = conflict.remote_value
            elif resolution == ConflictResolution.CLIENT_WINS:
                conflict.winning_value = conflict.local_value
            elif resolution == ConflictResolution.MANUAL:
                if chosen_value is None:
                    return False
                conflict.winning_value = chosen_value
            else:  # LAST_WRITER_WINS
                if conflict.local_clock.happens_before(conflict.remote_clock):
                    conflict.winning_value = conflict.remote_value
                else:
                    conflict.winning_value = conflict.local_value

            conflict.resolved = True
            conflict.resolution = resolution
            self._cloud_state[conflict.data_type]["value"] = conflict.winning_value

        logger.info("Conflict %s resolved via %s", conflict_id[:8], resolution.value)
        return True

    def get_status_report(self) -> SyncStatusReport:
        with self._lock:
            pending = self._job_queue.qsize()
            unresolved = sum(1 for c in self._conflicts.values() if not c.resolved)
            return SyncStatusReport(
                pending_jobs=pending,
                completed_today=self._stats["completed_today"],
                failed_today=self._stats["failed_today"],
                unresolved_conflicts=unresolved,
                last_sync_time=self._stats["last_sync"],
                next_sync_time=(self._stats["last_sync"] or time.time()) + 3600,
                bytes_uploaded_today=self._stats["bytes_up"],
                bytes_downloaded_today=self._stats["bytes_down"],
            )

    def stop(self) -> None:
        self._stop_event.set()

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                _, _, job = self._job_queue.get(timeout=self._WORKER_POLL_INTERVAL_S)
                self._process_job(job)
            except queue.Empty:
                pass

    def _process_job(self, job: SyncJob) -> None:
        with self._lock:
            job.status = SyncStatus.SYNCING

        try:
            payload_bytes = json.dumps(job.payload).encode()
            transfer_time = len(payload_bytes) / (self._SIMULATED_BANDWIDTH_KBPS * 128)
            time.sleep(min(transfer_time, 0.1))

            if job.direction in (SyncDirection.UPLOAD, SyncDirection.BIDIRECTIONAL):
                self._upload(job, payload_bytes)

            if job.direction in (SyncDirection.DOWNLOAD, SyncDirection.BIDIRECTIONAL):
                self._download(job)

            with self._lock:
                job.status = SyncStatus.COMPLETED
                job.completed_at = time.time()
                self._stats["completed_today"] += 1
                self._stats["last_sync"] = time.time()
                self._stats["bytes_up"] += len(payload_bytes)

            logger.debug("Sync job %s completed", job.job_id[:8])

        except Exception as exc:
            job.retry_count += 1
            if job.retry_count < job.max_retries:
                backoff = 2 ** job.retry_count
                time.sleep(min(backoff, 8))
                self._job_queue.put((job.priority, time.time(), job))
                logger.warning("Retry %d for job %s", job.retry_count, job.job_id[:8])
            else:
                with self._lock:
                    job.status = SyncStatus.FAILED
                    job.error = str(exc)
                    self._stats["failed_today"] += 1
                logger.error("Sync job %s failed: %s", job.job_id[:8], exc)

    def _upload(self, job: SyncJob, payload_bytes: bytes) -> None:
        """Simulate uploading payload to cloud; detect conflicts."""
        with self._lock:
            existing = self._cloud_state.get(job.data_type, {})
            remote_clock_data = existing.get("clock", {})
            remote_vc = VectorClock(clocks=remote_clock_data)

            if (
                existing
                and not job.vector_clock.happens_before(remote_vc)
                and not remote_vc.happens_before(job.vector_clock)
                and existing.get("value") != job.payload.get("value")
            ):
                # Concurrent writes — conflict!
                conflict = SyncConflict(
                    conflict_id=str(uuid.uuid4()),
                    data_type=job.data_type,
                    local_clock=job.vector_clock,
                    remote_clock=remote_vc,
                    local_value=job.payload,
                    remote_value=existing.get("value"),
                )
                self._conflicts[conflict.conflict_id] = conflict
                job.status = SyncStatus.CONFLICT
                # Auto-resolve with configured strategy
                self.resolve_conflicts(
                    conflict.conflict_id, self._conflict_strategy
                )

            self._cloud_state[job.data_type] = {
                "value": job.payload,
                "clock": job.vector_clock.clocks,
                "updated_at": time.time(),
            }

    def _download(self, job: SyncJob) -> None:
        """Simulate downloading the latest value from cloud."""
        with self._lock:
            remote = self._cloud_state.get(job.data_type)
        if remote:
            self._stats["bytes_down"] += len(json.dumps(remote).encode())


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_cloud_sync: Optional[CloudSync] = None
_sync_lock = threading.Lock()


def get_cloud_sync() -> CloudSync:
    global _cloud_sync
    with _sync_lock:
        if _cloud_sync is None:
            _cloud_sync = CloudSync()
    return _cloud_sync


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("AI Holographic Wristwatch — Cloud Sync Demo")
    print("=" * 55)

    cs = CloudSync(node_id="demo-watch")

    jid = cs.queue_sync("health_records", {"steps": 8500, "calories": 420})
    time.sleep(0.5)

    status = cs.get_sync_status(jid)
    print(f"Job {jid[:8]}: status={status.status.value}")

    report = cs.get_status_report()
    print(f"Completed today: {report.completed_today}")
    print(f"Pending: {report.pending_jobs}")

    cs.stop()
    print("Done.")
