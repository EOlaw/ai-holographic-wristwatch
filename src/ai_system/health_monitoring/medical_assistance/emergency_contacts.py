"""Emergency Contacts — AI Holographic Wristwatch
Manages emergency contacts, simulates notifications, and locates nearby hospitals."""

from __future__ import annotations

import uuid
import threading
import time
import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class RelationshipType(Enum):
    FAMILY = "family"
    FRIEND = "friend"
    DOCTOR = "doctor"
    CAREGIVER = "caregiver"
    EMERGENCY_SERVICE = "emergency_service"


@dataclass
class Contact:
    name: str
    phone: str
    relationship: RelationshipType
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    email: str = ""
    is_primary: bool = False
    notes: str = ""


@dataclass
class HospitalInfo:
    name: str
    address: str
    phone: str
    distance_km: float
    trauma_level: int  # 1 = highest, 3 = basic


@dataclass
class NotificationResult:
    contact_id: str
    contact_name: str
    channel: str  # "sms", "call", "email"
    success: bool
    timestamp: float
    message: str


class EmergencyContactManager:
    """Manages emergency contacts and simulates emergency notification workflows."""

    # Static list of example hospitals (used for nearest-hospital lookup)
    _HOSPITALS: List[HospitalInfo] = [
        HospitalInfo(
            name="City General Hospital",
            address="123 Main St, Downtown",
            phone="555-001-0001",
            distance_km=1.2,
            trauma_level=1,
        ),
        HospitalInfo(
            name="St. Mary's Medical Center",
            address="456 Oak Ave, Midtown",
            phone="555-002-0002",
            distance_km=2.8,
            trauma_level=2,
        ),
        HospitalInfo(
            name="Northside Urgent Care",
            address="789 Pine Rd, North District",
            phone="555-003-0003",
            distance_km=3.5,
            trauma_level=3,
        ),
        HospitalInfo(
            name="Riverside Memorial Hospital",
            address="321 River Blvd, Riverside",
            phone="555-004-0004",
            distance_km=5.1,
            trauma_level=1,
        ),
        HospitalInfo(
            name="Eastside Community Clinic",
            address="654 East St, East End",
            phone="555-005-0005",
            distance_km=6.3,
            trauma_level=3,
        ),
    ]

    def __init__(self) -> None:
        self._contacts: Dict[str, Contact] = {}
        self._lock = threading.RLock()
        self._notification_log: List[NotificationResult] = []
        logger.info("EmergencyContactManager initialized")

    def add_contact(
        self,
        name: str,
        phone: str,
        relationship: RelationshipType,
        email: str = "",
        is_primary: bool = False,
        notes: str = "",
    ) -> Contact:
        """Add a new emergency contact."""
        with self._lock:
            # If this is set as primary, demote any existing primary
            if is_primary:
                for existing in self._contacts.values():
                    if existing.is_primary:
                        existing.is_primary = False
            contact = Contact(
                name=name,
                phone=phone,
                relationship=relationship,
                email=email,
                is_primary=is_primary,
                notes=notes,
            )
            self._contacts[contact.id] = contact
            logger.info(
                "Added contact: %s (%s, %s) primary=%s [%s]",
                name, phone, relationship.value, is_primary, contact.id,
            )
            return contact

    def remove_contact(self, contact_id: str) -> bool:
        """Remove a contact by ID."""
        with self._lock:
            if contact_id not in self._contacts:
                logger.warning("remove_contact: id not found: %s", contact_id)
                return False
            removed = self._contacts.pop(contact_id)
            logger.info("Removed contact: %s [%s]", removed.name, contact_id)
            return True

    def get_primary_contact(self) -> Optional[Contact]:
        """Return the designated primary emergency contact, or None."""
        with self._lock:
            for contact in self._contacts.values():
                if contact.is_primary:
                    return contact
            # Fall back to first contact if no primary is explicitly set
            if self._contacts:
                return next(iter(self._contacts.values()))
            return None

    def get_all_contacts(self) -> List[Contact]:
        """Return all contacts sorted by primary first, then name."""
        with self._lock:
            contacts = list(self._contacts.values())
        contacts.sort(key=lambda c: (not c.is_primary, c.name))
        return contacts

    def notify_emergency(self, health_event: str, location: str = "") -> Dict:
        """
        Simulate sending emergency notifications to all contacts.
        Returns a summary dict with notification results.
        """
        timestamp = time.time()
        results: List[NotificationResult] = []

        with self._lock:
            contacts = list(self._contacts.values())

        if not contacts:
            logger.warning("notify_emergency called but no contacts registered.")
            return {
                "success": False,
                "message": "No emergency contacts registered.",
                "notified_count": 0,
                "results": [],
            }

        location_str = f" at {location}" if location else ""
        message_body = (
            f"EMERGENCY ALERT: {health_event}{location_str}. "
            f"This is an automated message from AI Holographic Wristwatch. "
            f"Please check on the wearer immediately."
        )

        for contact in contacts:
            # Simulate SMS notification
            sms_result = NotificationResult(
                contact_id=contact.id,
                contact_name=contact.name,
                channel="sms",
                success=True,  # simulated success
                timestamp=timestamp,
                message=f"SMS to {contact.phone}: {message_body}",
            )
            results.append(sms_result)
            logger.info(
                "Simulated SMS to %s (%s): %s",
                contact.name, contact.phone, health_event,
            )

            # Simulate call for primary contact or doctors
            if contact.is_primary or contact.relationship == RelationshipType.DOCTOR:
                call_result = NotificationResult(
                    contact_id=contact.id,
                    contact_name=contact.name,
                    channel="call",
                    success=True,
                    timestamp=timestamp,
                    message=f"CALL initiated to {contact.phone}: {health_event}",
                )
                results.append(call_result)
                logger.info("Simulated CALL to %s (%s)", contact.name, contact.phone)

            # Simulate email if available
            if contact.email:
                email_result = NotificationResult(
                    contact_id=contact.id,
                    contact_name=contact.name,
                    channel="email",
                    success=True,
                    timestamp=timestamp,
                    message=f"Email to {contact.email}: {message_body}",
                )
                results.append(email_result)

        with self._lock:
            self._notification_log.extend(results)

        successful = sum(1 for r in results if r.success)
        return {
            "success": successful > 0,
            "message": f"Emergency alert sent for: {health_event}",
            "notified_count": len(contacts),
            "notification_count": len(results),
            "successful_notifications": successful,
            "health_event": health_event,
            "location": location,
            "timestamp": timestamp,
            "results": [
                {
                    "contact": r.contact_name,
                    "channel": r.channel,
                    "success": r.success,
                }
                for r in results
            ],
        }

    def get_nearest_hospital(self, latitude: float = 0.0, longitude: float = 0.0) -> HospitalInfo:
        """
        Return the nearest hospital from the static list.
        In a real implementation this would use GPS coordinates; here we return
        the hospital with the smallest stored distance_km.
        """
        if not self._HOSPITALS:
            # Fallback placeholder
            return HospitalInfo(
                name="Unknown Hospital",
                address="Unknown",
                phone="911",
                distance_km=0.0,
                trauma_level=1,
            )
        nearest = min(self._HOSPITALS, key=lambda h: h.distance_km)
        logger.info(
            "Nearest hospital: %s (%.1f km, trauma level %d)",
            nearest.name, nearest.distance_km, nearest.trauma_level,
        )
        return nearest

    def get_stats(self) -> Dict:
        """Return statistics about contacts and notifications."""
        with self._lock:
            contacts = list(self._contacts.values())
            primary = self.get_primary_contact()
            rel_counts: Dict[str, int] = {}
            for c in contacts:
                rel_counts[c.relationship.value] = rel_counts.get(c.relationship.value, 0) + 1
            return {
                "total_contacts": len(contacts),
                "primary_contact": primary.name if primary else None,
                "relationship_breakdown": rel_counts,
                "total_notifications_sent": len(self._notification_log),
                "successful_notifications": sum(1 for n in self._notification_log if n.success),
                "hospitals_in_database": len(self._HOSPITALS),
                "nearest_hospital": self._HOSPITALS[0].name if self._HOSPITALS else None,
            }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_emergency_contact_manager: Optional[EmergencyContactManager] = None
_ecm_lock = threading.Lock()


def get_emergency_contact_manager() -> EmergencyContactManager:
    """Return the process-wide EmergencyContactManager singleton."""
    global _emergency_contact_manager
    with _ecm_lock:
        if _emergency_contact_manager is None:
            _emergency_contact_manager = EmergencyContactManager()
    return _emergency_contact_manager


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    manager = get_emergency_contact_manager()

    print("=== Emergency Contact Manager Demo ===\n")

    # Add contacts
    alice = manager.add_contact(
        "Alice Smith", "555-100-0001", RelationshipType.FAMILY,
        email="alice@example.com", is_primary=True, notes="Spouse"
    )
    bob = manager.add_contact(
        "Bob Johnson", "555-100-0002", RelationshipType.FRIEND,
        notes="Best friend"
    )
    dr_chen = manager.add_contact(
        "Dr. Chen", "555-200-0001", RelationshipType.DOCTOR,
        email="dr.chen@clinic.example.com", notes="Primary care physician"
    )

    print("All contacts:")
    for c in manager.get_all_contacts():
        primary_tag = " [PRIMARY]" if c.is_primary else ""
        print(f"  {c.name} — {c.phone} ({c.relationship.value}){primary_tag}")

    primary = manager.get_primary_contact()
    print(f"\nPrimary contact: {primary.name if primary else 'None'}")

    # Simulate emergency notification
    print("\n--- Simulating Emergency Notification ---")
    result = manager.notify_emergency(
        "Irregular heart rate detected (180 bpm)",
        location="Home — 42 Holographic Way"
    )
    print(f"Notified: {result['notified_count']} contacts")
    print(f"Total notifications: {result['notification_count']}")
    for r in result["results"]:
        status = "OK" if r["success"] else "FAIL"
        print(f"  [{status}] {r['contact']} via {r['channel']}")

    # Nearest hospital
    hospital = manager.get_nearest_hospital()
    print(f"\nNearest hospital: {hospital.name} ({hospital.distance_km} km, trauma level {hospital.trauma_level})")
    print(f"  Address: {hospital.address}")
    print(f"  Phone:   {hospital.phone}")

    print("\n--- Stats ---")
    for k, v in manager.get_stats().items():
        print(f"  {k}: {v}")
