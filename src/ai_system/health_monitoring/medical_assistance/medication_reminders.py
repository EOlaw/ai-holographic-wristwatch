"""Medication Reminders — AI Holographic Wristwatch
Tracks medication schedules, sends reminders, and monitors adherence."""

from __future__ import annotations

import uuid
import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class ScheduleType(Enum):
    DAILY = "daily"
    TWICE_DAILY = "twice_daily"
    THREE_TIMES_DAILY = "three_times_daily"
    WEEKLY = "weekly"
    AS_NEEDED = "as_needed"
    WITH_MEALS = "with_meals"
    EVERY_N_HOURS = "every_n_hours"


@dataclass
class MedicationEntry:
    name: str
    dosage: str
    schedule_type: ScheduleType
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    times_per_day: int = 1
    next_due: float = field(default_factory=time.time)
    taken_count: int = 0
    missed_count: int = 0
    notes: str = ""
    active: bool = True


@dataclass
class MedicationLog:
    medication_id: str
    action: str
    timestamp: float
    notes: str


class MedicationReminderSystem:
    """Manages medication schedules, reminders, and adherence tracking."""

    # Interval in seconds for each schedule type
    _SCHEDULE_INTERVALS: Dict[ScheduleType, float] = {
        ScheduleType.DAILY: 86400.0,
        ScheduleType.TWICE_DAILY: 43200.0,
        ScheduleType.THREE_TIMES_DAILY: 28800.0,
        ScheduleType.WEEKLY: 604800.0,
        ScheduleType.AS_NEEDED: 0.0,
        ScheduleType.WITH_MEALS: 28800.0,  # roughly every 8 hours with meals
        ScheduleType.EVERY_N_HOURS: 14400.0,  # default 4 hours
    }

    def __init__(self) -> None:
        self._medications: Dict[str, MedicationEntry] = {}
        self._logs: List[MedicationLog] = []
        self._lock = threading.RLock()
        self._reminder_count: int = 0
        logger.info("MedicationReminderSystem initialized")

    def add_medication(
        self,
        name: str,
        dosage: str,
        schedule_type: ScheduleType,
        times_per_day: int = 1,
        notes: str = "",
    ) -> MedicationEntry:
        """Add a new medication to the tracking system."""
        with self._lock:
            entry = MedicationEntry(
                name=name,
                dosage=dosage,
                schedule_type=schedule_type,
                times_per_day=times_per_day,
                next_due=time.time(),
                notes=notes,
            )
            self._medications[entry.id] = entry
            self._log_action(entry.id, "added", f"Added {name} {dosage}")
            logger.info("Added medication: %s (%s) [%s]", name, dosage, entry.id)
            return entry

    def remove_medication(self, medication_id: str) -> bool:
        """Deactivate a medication entry (soft delete)."""
        with self._lock:
            entry = self._medications.get(medication_id)
            if entry is None:
                logger.warning("remove_medication: id not found: %s", medication_id)
                return False
            entry.active = False
            self._log_action(medication_id, "removed", "Medication deactivated")
            logger.info("Removed medication: %s", medication_id)
            return True

    def get_due_medications(self, window_minutes: int = 30) -> List[MedicationEntry]:
        """Return medications due within the next window_minutes, including overdue ones."""
        now = time.time()
        window_secs = window_minutes * 60.0
        due: List[MedicationEntry] = []
        with self._lock:
            for entry in self._medications.values():
                if not entry.active:
                    continue
                if entry.schedule_type == ScheduleType.AS_NEEDED:
                    continue
                if entry.next_due <= now + window_secs:
                    due.append(entry)
        due.sort(key=lambda e: e.next_due)
        logger.debug("Due medications (window=%dmin): %d found", window_minutes, len(due))
        return due

    def mark_taken(self, medication_id: str, notes: str = "") -> bool:
        """Mark a medication as taken and schedule the next dose."""
        with self._lock:
            entry = self._medications.get(medication_id)
            if entry is None or not entry.active:
                logger.warning("mark_taken: medication not found or inactive: %s", medication_id)
                return False
            entry.taken_count += 1
            entry.next_due = self._calculate_next_due(entry)
            self._reminder_count += 1
            self._log_action(medication_id, "taken", notes or f"Dose #{entry.taken_count} taken")
            logger.info(
                "Marked taken: %s (dose #%d), next_due in %.1f min",
                entry.name,
                entry.taken_count,
                (entry.next_due - time.time()) / 60.0,
            )
            return True

    def mark_skipped(self, medication_id: str) -> bool:
        """Mark a medication dose as skipped."""
        with self._lock:
            entry = self._medications.get(medication_id)
            if entry is None or not entry.active:
                logger.warning("mark_skipped: medication not found or inactive: %s", medication_id)
                return False
            entry.missed_count += 1
            entry.next_due = self._calculate_next_due(entry)
            self._log_action(medication_id, "skipped", f"Dose skipped; missed_count={entry.missed_count}")
            logger.info("Marked skipped: %s (missed #%d)", entry.name, entry.missed_count)
            return True

    def get_adherence_rate(self, medication_id: Optional[str] = None) -> float:
        """Return adherence rate (0.0–1.0). Pass medication_id for per-medication rate."""
        with self._lock:
            if medication_id is not None:
                entry = self._medications.get(medication_id)
                if entry is None:
                    return 0.0
                total = entry.taken_count + entry.missed_count
                if total == 0:
                    return 1.0
                return entry.taken_count / total

            # Overall adherence across all active medications
            total_taken = 0
            total_missed = 0
            for entry in self._medications.values():
                if entry.active:
                    total_taken += entry.taken_count
                    total_missed += entry.missed_count
            total = total_taken + total_missed
            if total == 0:
                return 1.0
            return total_taken / total

    def get_all_medications(self) -> List[MedicationEntry]:
        """Return all active medications."""
        with self._lock:
            return [e for e in self._medications.values() if e.active]

    def get_medication_log(self, medication_id: str, limit: int = 20) -> List[MedicationLog]:
        """Return recent log entries for a specific medication."""
        with self._lock:
            entries = [log for log in self._logs if log.medication_id == medication_id]
            entries.sort(key=lambda l: l.timestamp, reverse=True)
            return entries[:limit]

    def _calculate_next_due(self, entry: MedicationEntry) -> float:
        """Calculate the next due timestamp based on schedule type."""
        now = time.time()
        interval = self._SCHEDULE_INTERVALS.get(entry.schedule_type, 86400.0)
        if entry.schedule_type == ScheduleType.AS_NEEDED:
            return now + 86400.0  # 24h buffer for as-needed
        if entry.schedule_type == ScheduleType.EVERY_N_HOURS and entry.times_per_day > 0:
            # times_per_day used as "every N hours" when schedule is EVERY_N_HOURS
            interval = (24.0 / entry.times_per_day) * 3600.0
        return now + interval

    def _log_action(self, medication_id: str, action: str, notes: str) -> None:
        """Internal: append a log entry."""
        log = MedicationLog(
            medication_id=medication_id,
            action=action,
            timestamp=time.time(),
            notes=notes,
        )
        self._logs.append(log)

    def get_stats(self) -> Dict:
        """Return summary statistics for the reminder system."""
        with self._lock:
            active_meds = [e for e in self._medications.values() if e.active]
            overdue = [
                e for e in active_meds
                if e.schedule_type != ScheduleType.AS_NEEDED and e.next_due < time.time()
            ]
            return {
                "total_medications": len(self._medications),
                "active_medications": len(active_meds),
                "overdue_medications": len(overdue),
                "overall_adherence_rate": round(self.get_adherence_rate(), 3),
                "total_doses_taken": sum(e.taken_count for e in active_meds),
                "total_doses_missed": sum(e.missed_count for e in active_meds),
                "total_log_entries": len(self._logs),
                "reminder_count": self._reminder_count,
            }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_medication_reminder_system: Optional[MedicationReminderSystem] = None
_mrs_lock = threading.Lock()


def get_medication_reminder_system() -> MedicationReminderSystem:
    """Return the process-wide MedicationReminderSystem singleton."""
    global _medication_reminder_system
    with _mrs_lock:
        if _medication_reminder_system is None:
            _medication_reminder_system = MedicationReminderSystem()
    return _medication_reminder_system


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Run basic assertions to verify MedicationReminderSystem behavior."""
    system = MedicationReminderSystem()

    # Add medications
    med1 = system.add_medication("Aspirin", "100mg", ScheduleType.DAILY)
    med2 = system.add_medication("Vitamin D", "1000IU", ScheduleType.TWICE_DAILY, notes="With food")
    med3 = system.add_medication("Ibuprofen", "400mg", ScheduleType.AS_NEEDED)

    assert len(system.get_all_medications()) == 3, "Should have 3 active medications"

    # Mark taken / skipped
    assert system.mark_taken(med1.id, "Morning dose") is True
    assert med1.taken_count == 1

    assert system.mark_skipped(med2.id) is True
    assert med2.missed_count == 1

    # Adherence
    rate = system.get_adherence_rate()
    assert 0.0 <= rate <= 1.0, "Adherence rate must be between 0 and 1"

    per_med_rate = system.get_adherence_rate(med1.id)
    assert per_med_rate == 1.0, "med1 adherence should be 1.0 (1 taken, 0 missed)"

    # Remove medication
    assert system.remove_medication(med3.id) is True
    assert len(system.get_all_medications()) == 2, "Should have 2 active medications after removal"

    # Logs
    logs = system.get_medication_log(med1.id)
    assert len(logs) >= 1

    # Stats
    stats = system.get_stats()
    assert stats["active_medications"] == 2
    assert stats["total_doses_taken"] >= 1

    # Non-existent medication
    assert system.mark_taken("nonexistent") is False
    assert system.remove_medication("nonexistent") is False

    print("All MedicationReminderSystem tests passed.")


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_tests()

    system = get_medication_reminder_system()

    metformin = system.add_medication("Metformin", "500mg", ScheduleType.TWICE_DAILY, notes="With meals")
    lisinopril = system.add_medication("Lisinopril", "10mg", ScheduleType.DAILY, notes="Morning")
    vitamin_c = system.add_medication("Vitamin C", "500mg", ScheduleType.DAILY)

    print("\n--- Active Medications ---")
    for med in system.get_all_medications():
        print(f"  [{med.id}] {med.name} {med.dosage} — {med.schedule_type.value}")

    system.mark_taken(metformin.id, "Breakfast dose")
    system.mark_taken(lisinopril.id)
    system.mark_skipped(vitamin_c.id)

    print(f"\nOverall adherence: {system.get_adherence_rate():.1%}")
    print(f"Metformin adherence: {system.get_adherence_rate(metformin.id):.1%}")

    due = system.get_due_medications(window_minutes=60 * 24)
    print(f"\nMedications due in next 24h: {len(due)}")
    for med in due:
        mins = max(0, (med.next_due - time.time()) / 60)
        print(f"  {med.name}: due in {mins:.0f} min")

    print("\n--- Stats ---")
    for k, v in system.get_stats().items():
        print(f"  {k}: {v}")
