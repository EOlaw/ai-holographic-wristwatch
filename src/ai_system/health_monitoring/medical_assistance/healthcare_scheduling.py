"""Healthcare Scheduling — AI Holographic Wristwatch
Manages medical appointments, reminders, and appointment status tracking."""

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


class AppointmentType(Enum):
    GENERAL_CHECKUP = "general_checkup"
    SPECIALIST = "specialist"
    DENTAL = "dental"
    VISION = "vision"
    MENTAL_HEALTH = "mental_health"
    PHYSICAL_THERAPY = "physical_therapy"
    LAB_TEST = "lab_test"
    VACCINATION = "vaccination"


class AppointmentStatus(Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    MISSED = "missed"


@dataclass
class Appointment:
    provider: str
    appointment_type: AppointmentType
    scheduled_time: float  # Unix timestamp
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    location: str = ""
    notes: str = ""
    status: AppointmentStatus = AppointmentStatus.SCHEDULED
    reminder_sent: bool = False


class HealthcareScheduler:
    """Schedules and tracks medical appointments with reminder support."""

    # How many minutes before the appointment we consider it "upcoming soon"
    _REMINDER_WINDOW_HOURS: int = 24

    def __init__(self) -> None:
        self._appointments: Dict[str, Appointment] = {}
        self._lock = threading.RLock()
        logger.info("HealthcareScheduler initialized")

    def schedule_appointment(
        self,
        provider: str,
        appt_type: AppointmentType,
        scheduled_time: float,
        location: str = "",
        notes: str = "",
    ) -> Appointment:
        """Create and store a new appointment."""
        with self._lock:
            appt = Appointment(
                provider=provider,
                appointment_type=appt_type,
                scheduled_time=scheduled_time,
                location=location,
                notes=notes,
            )
            self._appointments[appt.id] = appt
            readable_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(scheduled_time))
            logger.info(
                "Appointment scheduled: %s with %s on %s [%s]",
                appt_type.value, provider, readable_time, appt.id,
            )
            return appt

    def cancel_appointment(self, appointment_id: str, reason: str = "") -> bool:
        """Cancel an appointment by ID."""
        with self._lock:
            appt = self._appointments.get(appointment_id)
            if appt is None:
                logger.warning("cancel_appointment: id not found: %s", appointment_id)
                return False
            if appt.status in (AppointmentStatus.COMPLETED, AppointmentStatus.CANCELLED):
                logger.warning(
                    "cancel_appointment: cannot cancel %s appointment [%s]",
                    appt.status.value, appointment_id,
                )
                return False
            appt.status = AppointmentStatus.CANCELLED
            if reason:
                appt.notes = (appt.notes + f" | Cancellation reason: {reason}").strip(" | ")
            logger.info("Appointment cancelled: %s [%s]", appt.provider, appointment_id)
            return True

    def complete_appointment(self, appointment_id: str, notes: str = "") -> bool:
        """Mark an appointment as completed."""
        with self._lock:
            appt = self._appointments.get(appointment_id)
            if appt is None:
                logger.warning("complete_appointment: id not found: %s", appointment_id)
                return False
            if appt.status == AppointmentStatus.CANCELLED:
                logger.warning(
                    "complete_appointment: appointment already cancelled [%s]", appointment_id
                )
                return False
            appt.status = AppointmentStatus.COMPLETED
            if notes:
                appt.notes = (appt.notes + f" | Completion notes: {notes}").strip(" | ")
            logger.info("Appointment completed: %s [%s]", appt.provider, appointment_id)
            return True

    def get_upcoming_appointments(self, days: int = 30) -> List[Appointment]:
        """Return upcoming appointments within the next `days` days, sorted by time."""
        now = time.time()
        cutoff = now + days * 86400.0
        with self._lock:
            upcoming = [
                appt for appt in self._appointments.values()
                if appt.status in (AppointmentStatus.SCHEDULED, AppointmentStatus.CONFIRMED)
                and now <= appt.scheduled_time <= cutoff
            ]
        upcoming.sort(key=lambda a: a.scheduled_time)
        logger.debug("Upcoming appointments (next %d days): %d found", days, len(upcoming))
        return upcoming

    def get_reminder_message(self, appointment_id: str) -> str:
        """Generate a human-readable reminder message for an appointment."""
        with self._lock:
            appt = self._appointments.get(appointment_id)
        if appt is None:
            return f"No appointment found with ID: {appointment_id}"

        readable_time = time.strftime("%A, %B %d at %I:%M %p", time.localtime(appt.scheduled_time))
        hours_until = (appt.scheduled_time - time.time()) / 3600.0

        if hours_until < 0:
            time_str = "This appointment has already passed."
        elif hours_until < 1:
            mins = int(hours_until * 60)
            time_str = f"In approximately {mins} minute(s)."
        elif hours_until < 24:
            time_str = f"In approximately {hours_until:.1f} hour(s)."
        else:
            days = hours_until / 24.0
            time_str = f"In approximately {days:.1f} day(s)."

        parts = [
            f"Reminder: {appt.appointment_type.value.replace('_', ' ').title()} appointment",
            f"with {appt.provider}",
            f"on {readable_time}.",
            time_str,
        ]
        if appt.location:
            parts.append(f"Location: {appt.location}.")
        if appt.notes:
            parts.append(f"Notes: {appt.notes}.")

        # Mark reminder as sent
        with self._lock:
            appt.reminder_sent = True

        return " ".join(parts)

    def get_overdue_appointments(self) -> List[Appointment]:
        """Return appointments that are past their scheduled time but not completed or cancelled."""
        now = time.time()
        with self._lock:
            overdue = [
                appt for appt in self._appointments.values()
                if appt.scheduled_time < now
                and appt.status in (AppointmentStatus.SCHEDULED, AppointmentStatus.CONFIRMED)
            ]
        # Mark overdue appointments as missed
        with self._lock:
            for appt in overdue:
                if appt.status != AppointmentStatus.MISSED:
                    appt.status = AppointmentStatus.MISSED
                    logger.warning("Appointment marked as missed: %s [%s]", appt.provider, appt.id)
        overdue.sort(key=lambda a: a.scheduled_time)
        return overdue

    def get_stats(self) -> Dict:
        """Return statistics about appointments."""
        with self._lock:
            all_appts = list(self._appointments.values())

        status_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for appt in all_appts:
            status_counts[appt.status.value] = status_counts.get(appt.status.value, 0) + 1
            type_counts[appt.appointment_type.value] = type_counts.get(appt.appointment_type.value, 0) + 1

        upcoming_count = len(self.get_upcoming_appointments(days=30))
        completed = status_counts.get("completed", 0)
        missed = status_counts.get("missed", 0)
        total_closed = completed + missed
        completion_rate = completed / total_closed if total_closed > 0 else 1.0

        return {
            "total_appointments": len(all_appts),
            "upcoming_30_days": upcoming_count,
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
            "completion_rate": round(completion_rate, 3),
            "reminders_sent": sum(1 for a in all_appts if a.reminder_sent),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_healthcare_scheduler: Optional[HealthcareScheduler] = None
_hs_lock = threading.Lock()


def get_healthcare_scheduler() -> HealthcareScheduler:
    """Return the process-wide HealthcareScheduler singleton."""
    global _healthcare_scheduler
    with _hs_lock:
        if _healthcare_scheduler is None:
            _healthcare_scheduler = HealthcareScheduler()
    return _healthcare_scheduler


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scheduler = get_healthcare_scheduler()

    print("=== Healthcare Scheduler Demo ===\n")

    now = time.time()

    # Schedule future appointments
    checkup = scheduler.schedule_appointment(
        "Dr. Williams", AppointmentType.GENERAL_CHECKUP,
        scheduled_time=now + 3 * 86400,  # 3 days from now
        location="City Medical Center, Room 204",
        notes="Annual physical"
    )
    dental = scheduler.schedule_appointment(
        "Dr. Park", AppointmentType.DENTAL,
        scheduled_time=now + 7 * 86400,  # 7 days from now
        location="Bright Smile Dental Clinic",
        notes="Routine cleaning"
    )
    lab = scheduler.schedule_appointment(
        "Quest Diagnostics", AppointmentType.LAB_TEST,
        scheduled_time=now + 1 * 86400,  # tomorrow
        location="Quest Lab, 100 Health Blvd",
        notes="Fasting blood panel"
    )
    past_appt = scheduler.schedule_appointment(
        "Dr. Nguyen", AppointmentType.SPECIALIST,
        scheduled_time=now - 3600,  # 1 hour ago
        location="Specialist Center",
    )

    # Print upcoming
    upcoming = scheduler.get_upcoming_appointments(days=30)
    print(f"Upcoming appointments (next 30 days): {len(upcoming)}")
    for appt in upcoming:
        days_from_now = (appt.scheduled_time - now) / 86400
        print(f"  [{appt.id}] {appt.appointment_type.value} with {appt.provider} in {days_from_now:.1f} days")

    # Reminder
    print(f"\nReminder for lab test:\n  {scheduler.get_reminder_message(lab.id)}")

    # Overdue
    overdue = scheduler.get_overdue_appointments()
    print(f"\nOverdue appointments: {len(overdue)}")
    for appt in overdue:
        print(f"  [{appt.id}] {appt.appointment_type.value} with {appt.provider} — {appt.status.value}")

    # Complete and cancel
    scheduler.complete_appointment(lab.id, notes="Results pending in 48h")
    scheduler.cancel_appointment(dental.id, reason="Provider unavailable")

    print("\n--- Stats ---")
    for k, v in scheduler.get_stats().items():
        print(f"  {k}: {v}")
