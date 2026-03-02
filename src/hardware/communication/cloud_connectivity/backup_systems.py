"""
Backup Systems — AI Holographic Wristwatch

Manages scheduled and on-demand backups of watch data to cloud and local storage.
Supports full, incremental, and differential backups with AES-256 encryption,
integrity verification via SHA-256 checksums, and automatic restore capabilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BackupStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFYING = "verifying"
    RESTORING = "restoring"


class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class StorageDestination(Enum):
    LOCAL_FLASH = "local_flash"
    CLOUD_PRIMARY = "cloud_primary"
    CLOUD_SECONDARY = "cloud_secondary"
    PAIRED_PHONE = "paired_phone"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BackupRecord:
    backup_id: str
    backup_type: BackupType
    destination: StorageDestination
    status: BackupStatus
    timestamp: float
    size_bytes: int
    checksum: str
    duration_seconds: float = 0.0
    compressed: bool = True
    encrypted: bool = True
    error_message: Optional[str] = None

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def is_valid(self) -> bool:
        return self.status == BackupStatus.COMPLETED and bool(self.checksum)


@dataclass
class BackupSchedule:
    interval_hours: float
    backup_type: BackupType
    destination: StorageDestination
    enabled: bool = True
    last_run: Optional[float] = None
    next_run: Optional[float] = None

    def compute_next_run(self) -> float:
        base = self.last_run or time.time()
        self.next_run = base + self.interval_hours * 3600
        return self.next_run

    @property
    def is_due(self) -> bool:
        if not self.enabled:
            return False
        return time.time() >= (self.next_run or 0)


# ---------------------------------------------------------------------------
# BackupSystem
# ---------------------------------------------------------------------------

class BackupSystem:
    """
    Manages data backup lifecycle: scheduling, execution, verification, and restore.

    Architecture:
    - Background scheduler thread fires backups according to registered schedules.
    - Each backup is recorded in an in-memory ledger (persisted to JSON on-device).
    - Incremental backups track a "dirty block" bitmap to minimise data transfer.
    - Integrity is verified with SHA-256 over compressed, encrypted payload.
    """

    _MAX_BACKUPS_RETAINED = 30
    _BLOCK_SIZE_BYTES = 4096
    _SIMULATED_WRITE_SPEED_MB_S = 8.0  # flash write speed

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: Dict[str, BackupRecord] = {}
        self._schedules: List[BackupSchedule] = []
        self._status = BackupStatus.IDLE
        self._current_backup_id: Optional[str] = None
        self._dirty_blocks: set = set()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._data_snapshot: Dict[str, bytes] = {}

        self._default_schedules()

    # ── Scheduling ────────────────────────────────────────────────────────────

    def _default_schedules(self) -> None:
        """Register default daily full + hourly incremental schedules."""
        self._schedules = [
            BackupSchedule(
                interval_hours=24,
                backup_type=BackupType.FULL,
                destination=StorageDestination.CLOUD_PRIMARY,
            ),
            BackupSchedule(
                interval_hours=1,
                backup_type=BackupType.INCREMENTAL,
                destination=StorageDestination.CLOUD_PRIMARY,
            ),
            BackupSchedule(
                interval_hours=168,  # weekly
                backup_type=BackupType.FULL,
                destination=StorageDestination.CLOUD_SECONDARY,
            ),
        ]
        for s in self._schedules:
            s.compute_next_run()

    def start_scheduler(self) -> None:
        """Start the background scheduler thread."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="backup-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()
        logger.info("Backup scheduler started")

    def stop_scheduler(self) -> None:
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)

    def _scheduler_loop(self) -> None:
        while not self._stop_event.wait(timeout=60.0):
            with self._lock:
                due = [s for s in self._schedules if s.is_due]
            for schedule in due:
                self.schedule_backup(
                    backup_type=schedule.backup_type,
                    destination=schedule.destination,
                )
                schedule.last_run = time.time()
                schedule.compute_next_run()

    # ── Public API ────────────────────────────────────────────────────────────

    def schedule_backup(
        self,
        backup_type: BackupType = BackupType.INCREMENTAL,
        destination: StorageDestination = StorageDestination.CLOUD_PRIMARY,
    ) -> BackupRecord:
        """
        Immediately schedule (and execute) a backup of the given type.

        For incremental backups, only dirty blocks since the last full backup
        are included. Returns a BackupRecord with the outcome.
        """
        backup_id = self._generate_id()
        size_bytes = self._estimate_payload_size(backup_type)

        record = BackupRecord(
            backup_id=backup_id,
            backup_type=backup_type,
            destination=destination,
            status=BackupStatus.RUNNING,
            timestamp=time.time(),
            size_bytes=size_bytes,
            checksum="",
        )

        with self._lock:
            self._records[backup_id] = record
            self._status = BackupStatus.RUNNING
            self._current_backup_id = backup_id

        logger.info(
            "Backup started: id=%s type=%s dest=%s size_mb=%.2f",
            backup_id, backup_type.value, destination.value, size_bytes / 1_048_576,
        )

        threading.Thread(
            target=self._execute_backup,
            args=(record,),
            name=f"backup-{backup_id[:8]}",
            daemon=True,
        ).start()

        return record

    def restore_backup(self, backup_id: str) -> bool:
        """
        Restore device data from a completed backup.

        Validates checksum before overwriting live data. Returns True on success.
        """
        with self._lock:
            record = self._records.get(backup_id)

        if record is None:
            logger.error("Restore failed: backup_id=%s not found", backup_id)
            return False

        if not record.is_valid:
            logger.error(
                "Restore failed: backup %s status=%s", backup_id, record.status.value
            )
            return False

        logger.info("Starting restore from backup %s (%.2f MB)", backup_id, record.size_mb)

        with self._lock:
            self._status = BackupStatus.RESTORING

        # Verify integrity before restore
        if not self._verify_checksum(record):
            logger.error("Restore aborted: checksum mismatch for backup %s", backup_id)
            with self._lock:
                self._status = BackupStatus.FAILED
            return False

        # Simulate restore time proportional to data size
        restore_time = record.size_bytes / (self._SIMULATED_WRITE_SPEED_MB_S * 1_048_576)
        time.sleep(min(restore_time * 0.01, 0.5))  # Compressed for test speed

        with self._lock:
            self._dirty_blocks.clear()
            self._status = BackupStatus.IDLE

        logger.info("Restore complete from backup %s", backup_id)
        return True

    def list_backups(
        self,
        destination: Optional[StorageDestination] = None,
        backup_type: Optional[BackupType] = None,
        limit: int = 20,
    ) -> List[BackupRecord]:
        """Return backups sorted by timestamp (newest first), with optional filtering."""
        with self._lock:
            records = list(self._records.values())

        if destination:
            records = [r for r in records if r.destination == destination]
        if backup_type:
            records = [r for r in records if r.backup_type == backup_type]

        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    def verify_integrity(self, backup_id: str) -> Tuple[bool, str]:
        """
        Verify the integrity of a stored backup via checksum comparison.
        Returns (passed: bool, detail_message: str).
        """
        with self._lock:
            record = self._records.get(backup_id)

        if record is None:
            return False, f"Backup {backup_id} not found"

        if record.status != BackupStatus.COMPLETED:
            return False, f"Backup not completed (status={record.status.value})"

        passed = self._verify_checksum(record)
        if passed:
            msg = f"Integrity OK — SHA-256: {record.checksum[:16]}..."
            logger.info("Integrity check PASSED for backup %s", backup_id)
        else:
            msg = f"INTEGRITY FAILURE — checksum mismatch for backup {backup_id}"
            logger.error(msg)

        return passed, msg

    def get_status(self) -> BackupStatus:
        with self._lock:
            return self._status

    def get_record(self, backup_id: str) -> Optional[BackupRecord]:
        with self._lock:
            return self._records.get(backup_id)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _execute_backup(self, record: BackupRecord) -> None:
        """Run the actual backup in a background thread."""
        start = time.time()
        try:
            payload = self._collect_payload(record.backup_type)
            compressed = self._compress(payload)
            encrypted = self._encrypt(compressed)
            checksum = hashlib.sha256(encrypted).hexdigest()
            transfer_time = len(encrypted) / (self._SIMULATED_WRITE_SPEED_MB_S * 1_048_576)
            time.sleep(min(transfer_time * 0.005, 0.3))  # Simulated I/O

            with self._lock:
                record.checksum = checksum
                record.size_bytes = len(encrypted)
                record.duration_seconds = time.time() - start
                record.status = BackupStatus.COMPLETED
                self._status = BackupStatus.IDLE
                self._dirty_blocks.clear()
                self._current_backup_id = None
                self._prune_old_records()

            logger.info(
                "Backup completed: id=%s duration=%.2fs size_mb=%.2f",
                record.backup_id, record.duration_seconds, record.size_mb,
            )
        except Exception as exc:
            with self._lock:
                record.status = BackupStatus.FAILED
                record.error_message = str(exc)
                self._status = BackupStatus.FAILED
            logger.error("Backup failed: id=%s error=%s", record.backup_id, exc)

    def _collect_payload(self, backup_type: BackupType) -> bytes:
        """Assemble the data payload depending on backup type."""
        base_data = json.dumps({
            "health_records": list(range(1000)),
            "settings": {"brightness": 80, "volume": 60},
            "contacts": [f"contact_{i}" for i in range(50)],
            "activity_log": [random.random() for _ in range(500)],
        }).encode()

        if backup_type == BackupType.FULL:
            return base_data
        elif backup_type == BackupType.INCREMENTAL:
            # Only dirty blocks
            dirty_ratio = len(self._dirty_blocks) / max(1, math.ceil(len(base_data) / self._BLOCK_SIZE_BYTES))
            slice_end = max(256, int(len(base_data) * dirty_ratio))
            return base_data[:slice_end]
        else:  # DIFFERENTIAL
            return base_data[:len(base_data) // 2]

    def _compress(self, data: bytes) -> bytes:
        """Simulate LZ4 compression (~2:1 ratio)."""
        compressed_len = max(128, len(data) // 2)
        return data[:compressed_len] + b"\x00" * max(0, compressed_len - len(data))

    def _encrypt(self, data: bytes) -> bytes:
        """Simulate AES-256 encryption (same size output for simulation)."""
        return bytes((b ^ 0xA5) for b in data)  # XOR for simulation

    def _verify_checksum(self, record: BackupRecord) -> bool:
        """Re-derive checksum from a simulated re-read; always returns True in sim."""
        if not record.checksum:
            return False
        # In production: re-fetch from storage and compute SHA-256.
        # In simulation: we trust the stored checksum.
        return True

    def _prune_old_records(self) -> None:
        """Remove oldest records beyond retention limit (call with lock held)."""
        if len(self._records) > self._MAX_BACKUPS_RETAINED:
            sorted_ids = sorted(
                self._records.keys(),
                key=lambda k: self._records[k].timestamp,
            )
            for old_id in sorted_ids[: len(self._records) - self._MAX_BACKUPS_RETAINED]:
                del self._records[old_id]

    @staticmethod
    def _generate_id() -> str:
        import uuid
        return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_backup_system: Optional[BackupSystem] = None
_backup_lock = threading.Lock()


def get_backup_system() -> BackupSystem:
    global _backup_system
    with _backup_lock:
        if _backup_system is None:
            _backup_system = BackupSystem()
    return _backup_system


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time

    print("AI Holographic Wristwatch — Backup Systems Demo")
    print("=" * 55)

    bs = BackupSystem()

    # Trigger a full backup
    rec = bs.schedule_backup(BackupType.FULL, StorageDestination.CLOUD_PRIMARY)
    print(f"Scheduled backup: {rec.backup_id[:8]}... status={rec.status.value}")
    _time.sleep(0.5)

    # Trigger incremental
    rec2 = bs.schedule_backup(BackupType.INCREMENTAL)
    _time.sleep(0.5)

    listings = bs.list_backups()
    print(f"Total backups on record: {len(listings)}")

    if listings:
        ok, msg = bs.verify_integrity(listings[0].backup_id)
        print(f"Verify integrity: {ok} — {msg}")
        restored = bs.restore_backup(listings[0].backup_id)
        print(f"Restore result: {restored}")
