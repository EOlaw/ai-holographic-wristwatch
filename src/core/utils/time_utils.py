# src/core/utils/time_utils.py
"""
Sophisticated Time Management Utilities for AI Holographic Wristwatch System

This module provides comprehensive time handling capabilities including timezone 
management, scheduling algorithms, calendar integration, time synchronization,
duration calculations, recurring event management, and temporal data analysis.
"""

import time
import datetime
from datetime import timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import asyncio
import calendar
import math
from abc import ABC, abstractmethod
from zoneinfo import ZoneInfo
import bisect
from collections import defaultdict, deque
import heapq
import uuid

class TimeUnit(Enum):
    """Supported time units for duration calculations."""
    NANOSECOND = 1e-9
    MICROSECOND = 1e-6
    MILLISECOND = 1e-3
    SECOND = 1.0
    MINUTE = 60.0
    HOUR = 3600.0
    DAY = 86400.0
    WEEK = 604800.0
    MONTH = 2629746.0  # Average month in seconds
    YEAR = 31556952.0  # Average year in seconds

class RecurrencePattern(Enum):
    """Supported recurrence patterns for scheduling."""
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    HOURLY = "hourly"
    CUSTOM = "custom"

class TimeZoneRegion(Enum):
    """Common timezone regions for device management."""
    UTC = "UTC"
    US_EASTERN = "America/New_York"
    US_CENTRAL = "America/Chicago"
    US_MOUNTAIN = "America/Denver"
    US_PACIFIC = "America/Los_Angeles"
    EUROPE_LONDON = "Europe/London"
    EUROPE_PARIS = "Europe/Paris"
    ASIA_TOKYO = "Asia/Tokyo"
    ASIA_SHANGHAI = "Asia/Shanghai"
    AUSTRALIA_SYDNEY = "Australia/Sydney"

@dataclass
class TimeInterval:
    """Represents a time interval with start and end times."""
    start_time: datetime.datetime
    end_time: datetime.datetime
    timezone_info: Optional[str] = None
    
    def __post_init__(self):
        """Validate interval after initialization."""
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
    
    def duration(self) -> timedelta:
        """Calculate interval duration."""
        return self.end_time - self.start_time
    
    def duration_seconds(self) -> float:
        """Get duration in seconds as float."""
        return self.duration().total_seconds()
    
    def contains(self, timestamp: datetime.datetime) -> bool:
        """Check if timestamp falls within interval."""
        return self.start_time <= timestamp <= self.end_time
    
    def overlaps(self, other: 'TimeInterval') -> bool:
        """Check if this interval overlaps with another."""
        return (self.start_time < other.end_time and 
                other.start_time < self.end_time)
    
    def intersection(self, other: 'TimeInterval') -> Optional['TimeInterval']:
        """Calculate intersection with another interval."""
        if not self.overlaps(other):
            return None
        
        start = max(self.start_time, other.start_time)
        end = min(self.end_time, other.end_time)
        
        return TimeInterval(start, end, self.timezone_info)

@dataclass
class ScheduledEvent:
    """Represents a scheduled event with recurrence support."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    start_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    end_time: Optional[datetime.datetime] = None
    timezone_info: str = "UTC"
    recurrence: RecurrencePattern = RecurrencePattern.NONE
    recurrence_interval: int = 1
    recurrence_end: Optional[datetime.datetime] = None
    max_occurrences: Optional[int] = None
    is_enabled: bool = True
    priority: int = 1  # 1-10 scale
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_next_occurrence(self, after: Optional[datetime.datetime] = None) -> Optional[datetime.datetime]:
        """Calculate next occurrence of recurring event."""
        if self.recurrence == RecurrencePattern.NONE:
            reference_time = after or datetime.datetime.now(timezone.utc)
            return self.start_time if self.start_time > reference_time else None
        
        reference_time = after or datetime.datetime.now(timezone.utc)
        
        # Ensure we work with timezone-aware datetimes
        if self.start_time.tzinfo is None:
            event_tz = ZoneInfo(self.timezone_info)
            start_time = self.start_time.replace(tzinfo=event_tz)
        else:
            start_time = self.start_time
        
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        
        # Convert to same timezone for calculation
        start_time_utc = start_time.astimezone(timezone.utc)
        
        if reference_time <= start_time_utc:
            return start_time_utc
        
        # Calculate next occurrence based on recurrence pattern
        if self.recurrence == RecurrencePattern.DAILY:
            days_diff = (reference_time.date() - start_time_utc.date()).days
            next_occurrence_days = ((days_diff // self.recurrence_interval) + 1) * self.recurrence_interval
            next_occurrence = start_time_utc + timedelta(days=next_occurrence_days)
        
        elif self.recurrence == RecurrencePattern.WEEKLY:
            weeks_diff = (reference_time - start_time_utc).days // 7
            next_occurrence_weeks = ((weeks_diff // self.recurrence_interval) + 1) * self.recurrence_interval
            next_occurrence = start_time_utc + timedelta(weeks=next_occurrence_weeks)
        
        elif self.recurrence == RecurrencePattern.MONTHLY:
            months_since_start = ((reference_time.year - start_time_utc.year) * 12 + 
                                reference_time.month - start_time_utc.month)
            next_occurrence_months = ((months_since_start // self.recurrence_interval) + 1) * self.recurrence_interval
            
            target_year = start_time_utc.year + (start_time_utc.month + next_occurrence_months - 1) // 12
            target_month = (start_time_utc.month + next_occurrence_months - 1) % 12 + 1
            
            next_occurrence = start_time_utc.replace(year=target_year, month=target_month)
        
        elif self.recurrence == RecurrencePattern.YEARLY:
            years_diff = reference_time.year - start_time_utc.year
            next_occurrence_years = ((years_diff // self.recurrence_interval) + 1) * self.recurrence_interval
            next_occurrence = start_time_utc.replace(year=start_time_utc.year + next_occurrence_years)
        
        elif self.recurrence == RecurrencePattern.HOURLY:
            hours_diff = int((reference_time - start_time_utc).total_seconds() // 3600)
            next_occurrence_hours = ((hours_diff // self.recurrence_interval) + 1) * self.recurrence_interval
            next_occurrence = start_time_utc + timedelta(hours=next_occurrence_hours)
        
        else:
            return None
        
        # Check if within recurrence limits
        if self.recurrence_end and next_occurrence > self.recurrence_end:
            return None
        
        return next_occurrence
    
    def get_occurrences_in_range(self, start_range: datetime.datetime, 
                                end_range: datetime.datetime,
                                max_count: int = 1000) -> List[datetime.datetime]:
        """Get all occurrences within specified time range."""
        occurrences = []
        
        current_time = start_range
        count = 0
        
        while count < max_count:
            next_occurrence = self.get_next_occurrence(current_time)
            
            if not next_occurrence or next_occurrence > end_range:
                break
            
            occurrences.append(next_occurrence)
            current_time = next_occurrence + timedelta(seconds=1)
            count += 1
        
        return occurrences

class TimezoneManager:
    """Advanced timezone management for global device deployment."""
    
    def __init__(self):
        self.timezone_cache = {}
        self.location_timezone_mapping = self._build_location_mapping()
        self._lock = threading.Lock()
    
    def _build_location_mapping(self) -> Dict[str, str]:
        """Build mapping of locations to timezone identifiers."""
        return {
            # Major cities and their timezones
            'new_york': 'America/New_York',
            'los_angeles': 'America/Los_Angeles',
            'chicago': 'America/Chicago',
            'london': 'Europe/London',
            'paris': 'Europe/Paris',
            'tokyo': 'Asia/Tokyo',
            'sydney': 'Australia/Sydney',
            'shanghai': 'Asia/Shanghai',
            'dubai': 'Asia/Dubai',
            'mumbai': 'Asia/Kolkata',
            'sao_paulo': 'America/Sao_Paulo',
            'mexico_city': 'America/Mexico_City'
        }
    
    def get_timezone(self, timezone_identifier: str) -> ZoneInfo:
        """Get timezone object with caching."""
        with self._lock:
            if timezone_identifier not in self.timezone_cache:
                try:
                    self.timezone_cache[timezone_identifier] = ZoneInfo(timezone_identifier)
                except Exception:
                    # Fallback to UTC for invalid timezone
                    self.timezone_cache[timezone_identifier] = ZoneInfo('UTC')
            
            return self.timezone_cache[timezone_identifier]
    
    def convert_timezone(self, dt: datetime.datetime, target_timezone: str) -> datetime.datetime:
        """Convert datetime to target timezone."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        target_tz = self.get_timezone(target_timezone)
        return dt.astimezone(target_tz)
    
    def get_timezone_offset(self, timezone_identifier: str, 
                           at_time: Optional[datetime.datetime] = None) -> timedelta:
        """Get timezone offset from UTC at specified time."""
        target_tz = self.get_timezone(timezone_identifier)
        reference_time = at_time or datetime.datetime.now(timezone.utc)
        
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        
        localized_time = reference_time.astimezone(target_tz)
        return localized_time.utcoffset()
    
    def find_timezone_by_location(self, location: str) -> Optional[str]:
        """Find timezone identifier by location name."""
        location_key = location.lower().replace(' ', '_').replace('-', '_')
        return self.location_timezone_mapping.get(location_key)
    
    def calculate_time_difference(self, tz1: str, tz2: str,
                                at_time: Optional[datetime.datetime] = None) -> timedelta:
        """Calculate time difference between two timezones."""
        reference_time = at_time or datetime.datetime.now(timezone.utc)
        
        offset1 = self.get_timezone_offset(tz1, reference_time)
        offset2 = self.get_timezone_offset(tz2, reference_time)
        
        return offset2 - offset1

class TimeSync:
    """Network Time Protocol (NTP) and time synchronization utilities."""
    
    def __init__(self, ntp_servers: Optional[List[str]] = None):
        self.ntp_servers = ntp_servers or [
            'pool.ntp.org',
            'time.google.com',
            'time.apple.com',
            'time.windows.com'
        ]
        self.last_sync_time = None
        self.sync_offset = 0.0
        self.sync_accuracy = None
        self._lock = threading.Lock()
    
    async def sync_time_async(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Asynchronously synchronize time with NTP servers."""
        sync_results = []
        
        # Query multiple NTP servers for accuracy
        for server in self.ntp_servers[:3]:  # Use first 3 servers
            try:
                result = await self._query_ntp_server_async(server, timeout)
                sync_results.append(result)
            except Exception as e:
                sync_results.append({'server': server, 'error': str(e)})
        
        # Calculate best time offset
        valid_results = [r for r in sync_results if 'offset' in r]
        
        if not valid_results:
            raise TimeSyncError("Failed to synchronize with any NTP server")
        
        # Use median offset for robustness
        offsets = [r['offset'] for r in valid_results]
        median_offset = sorted(offsets)[len(offsets) // 2]
        
        with self._lock:
            self.sync_offset = median_offset
            self.last_sync_time = time.time()
            self.sync_accuracy = self._calculate_sync_accuracy(valid_results)
        
        return {
            'sync_offset': median_offset,
            'sync_accuracy': self.sync_accuracy,
            'servers_used': [r['server'] for r in valid_results],
            'sync_timestamp': self.last_sync_time
        }
    
    async def _query_ntp_server_async(self, server: str, timeout: float) -> Dict[str, Any]:
        """Query single NTP server asynchronously."""
        import socket
        
        # NTP packet format (48 bytes)
        ntp_packet = bytearray(48)
        ntp_packet[0] = 0x1b  # NTP version 3, client mode
        
        start_time = time.time()
        
        try:
            # Create socket and send NTP request
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            
            await asyncio.get_event_loop().run_in_executor(
                None, sock.sendto, ntp_packet, (server, 123)
            )
            
            response, _ = await asyncio.get_event_loop().run_in_executor(
                None, sock.recvfrom, 48
            )
            
            receive_time = time.time()
            
            # Parse NTP response
            transmit_timestamp = self._parse_ntp_timestamp(response[40:48])
            
            # Calculate offset (simplified NTP calculation)
            network_delay = (receive_time - start_time) / 2
            server_time = transmit_timestamp + network_delay
            local_time = receive_time
            offset = server_time - local_time
            
            sock.close()
            
            return {
                'server': server,
                'offset': offset,
                'delay': network_delay,
                'accuracy': abs(offset)
            }
            
        except Exception as e:
            if 'sock' in locals():
                sock.close()
            raise TimeSyncError(f"NTP query failed for {server}: {str(e)}")
    
    def _parse_ntp_timestamp(self, data: bytes) -> float:
        """Parse NTP timestamp format to Unix timestamp."""
        import struct
        
        # NTP timestamp: seconds since 1900-01-01
        ntp_time = struct.unpack('>I', data[:4])[0]
        ntp_fraction = struct.unpack('>I', data[4:8])[0]
        
        # Convert to Unix timestamp (seconds since 1970-01-01)
        ntp_epoch_offset = 2208988800  # Seconds between 1900 and 1970
        unix_time = ntp_time - ntp_epoch_offset
        fraction_seconds = ntp_fraction / (2**32)
        
        return unix_time + fraction_seconds
    
    def _calculate_sync_accuracy(self, sync_results: List[Dict[str, Any]]) -> float:
        """Calculate synchronization accuracy from multiple server results."""
        if len(sync_results) < 2:
            return sync_results[0].get('accuracy', 1.0)
        
        offsets = [r['offset'] for r in sync_results]
        return float(np.std(offsets))
    
    def get_synchronized_time(self) -> datetime.datetime:
        """Get current time adjusted for synchronization offset."""
        current_time = time.time()
        
        with self._lock:
            if self.last_sync_time and (current_time - self.last_sync_time) < 3600:  # 1 hour
                synchronized_time = current_time + self.sync_offset
            else:
                synchronized_time = current_time  # Use local time if sync is stale
        
        return datetime.datetime.fromtimestamp(synchronized_time, tz=timezone.utc)
    
    def is_sync_current(self, max_age_seconds: float = 3600) -> bool:
        """Check if time synchronization is current."""
        with self._lock:
            if not self.last_sync_time:
                return False
            
            return (time.time() - self.last_sync_time) < max_age_seconds

class EventScheduler:
    """Advanced event scheduling system with priority and conflict resolution."""
    
    def __init__(self, timezone_manager: TimezoneManager):
        self.timezone_manager = timezone_manager
        self.scheduled_events = {}
        self.event_queue = []  # Min-heap for efficient scheduling
        self.recurring_events = {}
        self.is_running = False
        self.scheduler_thread = None
        self._lock = threading.Lock()
        self.event_callbacks = defaultdict(list)
        self.conflict_resolution_policy = "priority_based"
    
    def schedule_event(self, event: ScheduledEvent) -> bool:
        """Schedule an event with conflict detection."""
        with self._lock:
            # Check for conflicts
            conflicts = self._detect_conflicts(event)
            
            if conflicts and not self._resolve_conflicts(event, conflicts):
                return False
            
            # Add event to storage
            self.scheduled_events[event.event_id] = event
            
            # Add to priority queue for next occurrence
            next_occurrence = event.get_next_occurrence()
            if next_occurrence:
                heapq.heappush(self.event_queue, (next_occurrence.timestamp(), event.event_id))
            
            # Track recurring events
            if event.recurrence != RecurrencePattern.NONE:
                self.recurring_events[event.event_id] = event
            
            return True
    
    def _detect_conflicts(self, new_event: ScheduledEvent) -> List[ScheduledEvent]:
        """Detect scheduling conflicts with existing events."""
        conflicts = []
        
        if not new_event.end_time:
            return conflicts  # Cannot detect conflicts for events without end time
        
        new_interval = TimeInterval(new_event.start_time, new_event.end_time)
        
        for existing_event in self.scheduled_events.values():
            if (existing_event.event_id != new_event.event_id and 
                existing_event.end_time):
                
                existing_interval = TimeInterval(existing_event.start_time, existing_event.end_time)
                
                if new_interval.overlaps(existing_interval):
                    conflicts.append(existing_event)
        
        return conflicts
    
    def _resolve_conflicts(self, new_event: ScheduledEvent, 
                          conflicts: List[ScheduledEvent]) -> bool:
        """Resolve scheduling conflicts based on policy."""
        if self.conflict_resolution_policy == "priority_based":
            # Allow if new event has higher priority than all conflicting events
            max_conflict_priority = max(event.priority for event in conflicts)
            return new_event.priority > max_conflict_priority
        
        elif self.conflict_resolution_policy == "reject_conflicts":
            return False  # Reject any event that creates conflicts
        
        elif self.conflict_resolution_policy == "allow_all":
            return True  # Allow all events regardless of conflicts
        
        else:
            return False
    
    def start_scheduler(self):
        """Start the event scheduler thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the event scheduler."""
        self.is_running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
    
    def _scheduler_loop(self):
        """Main scheduler loop for processing events."""
        while self.is_running:
            try:
                current_timestamp = time.time()
                events_to_process = []
                
                with self._lock:
                    # Get due events from queue
                    while (self.event_queue and 
                           self.event_queue[0][0] <= current_timestamp):
                        _, event_id = heapq.heappop(self.event_queue)
                        
                        if event_id in self.scheduled_events:
                            events_to_process.append(self.scheduled_events[event_id])
                
                # Process due events
                for event in events_to_process:
                    try:
                        self._execute_event(event)
                        
                        # Schedule next occurrence for recurring events
                        if event.recurrence != RecurrencePattern.NONE:
                            next_occurrence = event.get_next_occurrence(
                                datetime.datetime.fromtimestamp(current_timestamp, tz=timezone.utc)
                            )
                            if next_occurrence:
                                with self._lock:
                                    heapq.heappush(self.event_queue, 
                                                 (next_occurrence.timestamp(), event.event_id))
                        else:
                            # Remove non-recurring event after execution
                            with self._lock:
                                del self.scheduled_events[event.event_id]
                    
                    except Exception as e:
                        self._handle_event_error(event, e)
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Scheduler loop error: {e}")
                time.sleep(1.0)
    
    def _execute_event(self, event: ScheduledEvent):
        """Execute scheduled event and notify callbacks."""
        # Notify registered callbacks
        for callback in self.event_callbacks['all']:
            try:
                callback(event)
            except Exception as e:
                print(f"Event callback error: {e}")
        
        # Notify event-specific callbacks
        for callback in self.event_callbacks.get(event.event_id, []):
            try:
                callback(event)
            except Exception as e:
                print(f"Event-specific callback error: {e}")
    
    def _handle_event_error(self, event: ScheduledEvent, error: Exception):
        """Handle event execution errors."""
        print(f"Event execution error for {event.event_id}: {error}")
        
        # Log error through logging system if available
        try:
            from .logging_utils import get_logger
            logger = get_logger("event_scheduler")
            logger.error(f"Event execution failed: {event.title}", exception=error)
        except ImportError:
            pass
    
    def add_event_callback(self, callback: Callable[[ScheduledEvent], None],
                          event_id: Optional[str] = None):
        """Add callback for event notifications."""
        key = event_id if event_id else 'all'
        self.event_callbacks[key].append(callback)
    
    def remove_event(self, event_id: str) -> bool:
        """Remove scheduled event."""
        with self._lock:
            if event_id in self.scheduled_events:
                del self.scheduled_events[event_id]
                
                if event_id in self.recurring_events:
                    del self.recurring_events[event_id]
                
                # Remove from queue (will be skipped during processing)
                return True
        
        return False
    
    def get_upcoming_events(self, hours_ahead: int = 24) -> List[ScheduledEvent]:
        """Get events scheduled within specified time window."""
        end_time = datetime.datetime.now(timezone.utc) + timedelta(hours=hours_ahead)
        upcoming = []
        
        with self._lock:
            for event in self.scheduled_events.values():
                next_occurrence = event.get_next_occurrence()
                if next_occurrence and next_occurrence <= end_time:
                    upcoming.append(event)
        
        # Sort by next occurrence time
        upcoming.sort(key=lambda e: e.get_next_occurrence() or datetime.datetime.max)
        return upcoming

class DurationCalculator:
    """Utilities for duration calculations and time arithmetic."""
    
    @staticmethod
    def parse_duration_string(duration_str: str) -> timedelta:
        """Parse human-readable duration string to timedelta."""
        import re
        
        pattern = r'(?:(\d+(?:\.\d+)?)\s*(nanoseconds?|ns|microseconds?|us|milliseconds?|ms|seconds?|s|minutes?|m|hours?|h|days?|d|weeks?|w|months?|mo|years?|y))'
        matches = re.findall(pattern, duration_str.lower())
        
        if not matches:
            raise ValueError(f"Invalid duration format: {duration_str}")
        
        total_seconds = 0.0
        
        unit_multipliers = {
            'nanosecond': 1e-9, 'ns': 1e-9,
            'microsecond': 1e-6, 'us': 1e-6,
            'millisecond': 1e-3, 'ms': 1e-3,
            'second': 1.0, 's': 1.0,
            'minute': 60.0, 'm': 60.0,
            'hour': 3600.0, 'h': 3600.0,
            'day': 86400.0, 'd': 86400.0,
            'week': 604800.0, 'w': 604800.0,
            'month': 2629746.0, 'mo': 2629746.0,
            'year': 31556952.0, 'y': 31556952.0
        }
        
        for value_str, unit in matches:
            value = float(value_str)
            unit_clean = unit.rstrip('s')  # Remove plural 's'
            
            if unit_clean in unit_multipliers:
                total_seconds += value * unit_multipliers[unit_clean]
            else:
                raise ValueError(f"Unknown time unit: {unit}")
        
        return timedelta(seconds=total_seconds)
    
    @staticmethod
    def format_duration(duration: timedelta, precision: int = 2,
                       largest_unit: TimeUnit = TimeUnit.YEAR) -> str:
        """Format timedelta as human-readable string."""
        total_seconds = duration.total_seconds()
        
        if total_seconds < 0:
            return f"-{DurationCalculator.format_duration(-duration, precision, largest_unit)}"
        
        units = [
            (TimeUnit.YEAR, 'year'),
            (TimeUnit.MONTH, 'month'),
            (TimeUnit.WEEK, 'week'),
            (TimeUnit.DAY, 'day'),
            (TimeUnit.HOUR, 'hour'),
            (TimeUnit.MINUTE, 'minute'),
            (TimeUnit.SECOND, 'second'),
            (TimeUnit.MILLISECOND, 'millisecond'),
            (TimeUnit.MICROSECOND, 'microsecond'),
            (TimeUnit.NANOSECOND, 'nanosecond')
        ]
        
        # Find starting unit
        start_index = next((i for i, (unit, _) in enumerate(units) 
                          if unit == largest_unit), 0)
        
        parts = []
        remaining_seconds = total_seconds
        
        for i in range(start_index, len(units)):
            unit_enum, unit_name = units[i]
            unit_seconds = unit_enum.value
            
            if remaining_seconds >= unit_seconds:
                count = int(remaining_seconds // unit_seconds)
                remaining_seconds %= unit_seconds
                
                unit_str = unit_name if count == 1 else f"{unit_name}s"
                parts.append(f"{count} {unit_str}")
                
                if len(parts) >= precision:
                    break
        
        if not parts:
            if total_seconds < 1e-6:
                return f"{total_seconds * 1e9:.1f} nanoseconds"
            elif total_seconds < 1e-3:
                return f"{total_seconds * 1e6:.1f} microseconds"
            elif total_seconds < 1:
                return f"{total_seconds * 1e3:.1f} milliseconds"
            else:
                return f"{total_seconds:.3f} seconds"
        
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return f"{', '.join(parts[:-1])}, and {parts[-1]}"
    
    @staticmethod
    def add_business_days(start_date: datetime.date, business_days: int,
                         holidays: Optional[List[datetime.date]] = None) -> datetime.date:
        """Add business days to date, excluding weekends and holidays."""
        holidays = holidays or []
        current_date = start_date
        days_added = 0
        
        while days_added < business_days:
            current_date += timedelta(days=1)
            
            # Skip weekends (Monday=0, Sunday=6)
            if current_date.weekday() < 5 and current_date not in holidays:
                days_added += 1
        
        return current_date
    
    @staticmethod
    def calculate_business_hours(start_time: datetime.datetime, 
                               end_time: datetime.datetime,
                               business_start: datetime.time = datetime.time(9, 0),
                               business_end: datetime.time = datetime.time(17, 0),
                               holidays: Optional[List[datetime.date]] = None) -> float:
        """Calculate business hours between two timestamps."""
        holidays = holidays or []
        
        if start_time.date() == end_time.date():
            # Same day calculation
            if start_time.date().weekday() >= 5 or start_time.date() in holidays:
                return 0.0  # Weekend or holiday
            
            business_start_dt = datetime.datetime.combine(start_time.date(), business_start)
            business_end_dt = datetime.datetime.combine(start_time.date(), business_end)
            
            overlap_start = max(start_time, business_start_dt)
            overlap_end = min(end_time, business_end_dt)
            
            if overlap_start < overlap_end:
                return (overlap_end - overlap_start).total_seconds() / 3600.0
            else:
                return 0.0
        
        # Multi-day calculation
        total_hours = 0.0
        current_date = start_time.date()
        
        while current_date <= end_time.date():
            if current_date.weekday() < 5 and current_date not in holidays:
                day_start = (start_time if current_date == start_time.date() 
                           else datetime.datetime.combine(current_date, business_start))
                day_end = (end_time if current_date == end_time.date()
                          else datetime.datetime.combine(current_date, business_end))
                
                business_start_dt = datetime.datetime.combine(current_date, business_start)
                business_end_dt = datetime.datetime.combine(current_date, business_end)
                
                overlap_start = max(day_start, business_start_dt)
                overlap_end = min(day_end, business_end_dt)
                
                if overlap_start < overlap_end:
                    total_hours += (overlap_end - overlap_start).total_seconds() / 3600.0
            
            current_date += timedelta(days=1)
        
        return total_hours

class TemporalDataAnalyzer:
    """Analyze temporal patterns in time-series data."""
    
    def __init__(self, data_points: List[Tuple[datetime.datetime, float]]):
        self.data_points = sorted(data_points, key=lambda x: x[0])
        self.analysis_cache = {}
    
    def detect_periodic_patterns(self, min_period_hours: float = 1.0,
                               max_period_hours: float = 168.0) -> List[Dict[str, Any]]:
        """Detect periodic patterns in temporal data using FFT."""
        if len(self.data_points) < 10:
            return []
        
        # Convert to regular time series
        timestamps = [point[0].timestamp() for point in self.data_points]
        values = [point[1] for point in self.data_points]
        
        # Interpolate to regular intervals
        start_time = timestamps[0]
        end_time = timestamps[-1]
        duration = end_time - start_time
        
        if duration < min_period_hours * 3600:
            return []
        
        # Create regular time series with 1-minute resolution
        regular_timestamps = np.arange(start_time, end_time, 60)
        regular_values = np.interp(regular_timestamps, timestamps, values)
        
        # Apply FFT
        fft_values = np.fft.fft(regular_values)
        fft_frequencies = np.fft.fftfreq(len(regular_values), 60)  # 60-second intervals
        
        # Convert frequencies to periods (in hours)
        periods_hours = np.abs(1 / (fft_frequencies * 3600))
        
        # Find significant periodicities
        magnitude = np.abs(fft_values)
        
        # Filter by period range
        valid_indices = np.where((periods_hours >= min_period_hours) & 
                               (periods_hours <= max_period_hours))[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Get top periodicities by magnitude
        valid_magnitudes = magnitude[valid_indices]
        valid_periods = periods_hours[valid_indices]
        
        # Sort by magnitude and take top patterns
        sorted_indices = np.argsort(valid_magnitudes)[::-1]
        
        patterns = []
        for i in sorted_indices[:5]:  # Top 5 patterns
            idx = valid_indices[i]
            pattern = {
                'period_hours': float(valid_periods[i]),
                'strength': float(valid_magnitudes[i] / np.max(magnitude)),
                'frequency': float(fft_frequencies[idx]),
                'confidence': self._calculate_pattern_confidence(valid_periods[i], values)
            }
            
            if pattern['strength'] > 0.1:  # Only significant patterns
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_confidence(self, period_hours: float, values: List[float]) -> float:
        """Calculate confidence score for detected pattern."""
        period_samples = int(period_hours * 60)  # Convert to 1-minute samples
        
        if period_samples >= len(values):
            return 0.0
        
        # Calculate autocorrelation at the detected period
        correlation = np.corrcoef(values[:-period_samples], values[period_samples:])[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def detect_anomalous_timestamps(self, threshold_std: float = 2.0) -> List[datetime.datetime]:
        """Detect anomalous data points based on temporal patterns."""
        if len(self.data_points) < 5:
            return []
        
        values = [point[1] for point in self.data_points]
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        anomalous_timestamps = []
        
        for timestamp, value in self.data_points:
            z_score = abs(value - mean_value) / std_value if std_value > 0 else 0
            
            if z_score > threshold_std:
                anomalous_timestamps.append(timestamp)
        
        return anomalous_timestamps
    
    def calculate_temporal_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive temporal statistics."""
        if not self.data_points:
            return {}
        
        timestamps = [point[0] for point in self.data_points]
        values = [point[1] for point in self.data_points]
        
        # Time span analysis
        time_span = timestamps[-1] - timestamps[0]
        
        # Sampling rate analysis
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                    for i in range(len(timestamps) - 1)]
        
        # Value statistics
        value_stats = {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'range': float(np.max(values) - np.min(values))
        }
        
        # Temporal statistics
        temporal_stats = {
            'time_span_hours': time_span.total_seconds() / 3600,
            'average_sampling_interval': np.mean(intervals),
            'sampling_rate_hz': 1.0 / np.mean(intervals) if intervals else 0.0,
            'data_coverage': len(self.data_points) / (time_span.total_seconds() / np.mean(intervals)) if intervals else 0.0
        }
        
        return {
            'value_statistics': value_stats,
            'temporal_statistics': temporal_stats,
            'data_quality': {
                'completeness': min(1.0, temporal_stats['data_coverage']),
                'regularity': 1.0 - (np.std(intervals) / np.mean(intervals)) if intervals else 0.0,
                'anomaly_ratio': len(self.detect_anomalous_timestamps()) / len(self.data_points)
            }
        }

class CalendarIntegration:
    """Integration utilities for external calendar systems."""
    
    def __init__(self, timezone_manager: TimezoneManager):
        self.timezone_manager = timezone_manager
        self.calendar_providers = {}
        self.sync_intervals = {}
    
    def register_calendar_provider(self, provider_name: str, 
                                 sync_function: Callable[[], List[ScheduledEvent]],
                                 sync_interval_minutes: int = 30):
        """Register external calendar provider for synchronization."""
        self.calendar_providers[provider_name] = sync_function
        self.sync_intervals[provider_name] = sync_interval_minutes * 60
    
    async def sync_all_calendars(self) -> Dict[str, Any]:
        """Synchronize all registered calendar providers."""
        sync_results = {}
        
        for provider_name, sync_function in self.calendar_providers.items():
            try:
                start_time = time.time()
                events = await asyncio.get_event_loop().run_in_executor(
                    None, sync_function
                )
                sync_time = time.time() - start_time
                
                sync_results[provider_name] = {
                    'success': True,
                    'events_synced': len(events),
                    'sync_time_seconds': sync_time,
                    'last_sync': datetime.datetime.now(timezone.utc)
                }
                
            except Exception as e:
                sync_results[provider_name] = {
                    'success': False,
                    'error': str(e),
                    'last_sync': datetime.datetime.now(timezone.utc)
                }
        
        return sync_results
    
    def convert_external_event(self, external_event: Dict[str, Any],
                             provider_name: str) -> ScheduledEvent:
        """Convert external calendar event to internal format."""
        # Standard field mapping
        field_mappings = {
            'google': {
                'title': 'summary',
                'description': 'description',
                'start_time': 'start.dateTime',
                'end_time': 'end.dateTime',
                'timezone': 'start.timeZone'
            },
            'outlook': {
                'title': 'subject',
                'description': 'body.content',
                'start_time': 'start.dateTime',
                'end_time': 'end.dateTime',
                'timezone': 'start.timeZone'
            }
        }
        
        mapping = field_mappings.get(provider_name.lower(), {})
        
        # Extract fields using mapping
        title = self._extract_nested_field(external_event, mapping.get('title', 'title'))
        description = self._extract_nested_field(external_event, mapping.get('description', 'description'))
        
        # Parse start and end times
        start_time_str = self._extract_nested_field(external_event, mapping.get('start_time', 'start'))
        end_time_str = self._extract_nested_field(external_event, mapping.get('end_time', 'end'))
        
        start_time = self._parse_iso_datetime(start_time_str)
        end_time = self._parse_iso_datetime(end_time_str) if end_time_str else None
        
        # Extract timezone
        event_timezone = self._extract_nested_field(external_event, mapping.get('timezone', 'timezone'))
        if not event_timezone:
            event_timezone = 'UTC'
        
        return ScheduledEvent(
            title=title or "Untitled Event",
            description=description or "",
            start_time=start_time,
            end_time=end_time,
            timezone_info=event_timezone,
            metadata={'provider': provider_name, 'original_data': external_event}
        )
    
    def _extract_nested_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Extract field from nested dictionary using dot notation."""
        try:
            current = data
            for part in field_path.split('.'):
                current = current[part]
            return current
        except (KeyError, TypeError):
            return None
    
    def _parse_iso_datetime(self, datetime_str: str) -> datetime.datetime:
        """Parse ISO format datetime string."""
        try:
            # Handle various ISO formats
            if datetime_str.endswith('Z'):
                datetime_str = datetime_str[:-1] + '+00:00'
            
            return datetime.datetime.fromisoformat(datetime_str)
        except ValueError:
            # Fallback parsing
            try:
                return datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                raise ValueError(f"Cannot parse datetime: {datetime_str}")

class TimingProfiler:
    """Performance timing and profiling utilities."""
    
    def __init__(self, name: str = "default_profiler"):
        self.name = name
        self.timing_data = {}
        self.active_timers = {}
        self._lock = threading.Lock()
    
    def start_timer(self, operation_name: str) -> str:
        """Start timing an operation."""
        timer_id = str(uuid.uuid4())
        
        with self._lock:
            self.active_timers[timer_id] = {
                'operation_name': operation_name,
                'start_time': time.perf_counter(),
                'start_timestamp': datetime.datetime.now(timezone.utc)
            }
        
        return timer_id
    
    def stop_timer(self, timer_id: str) -> Optional[float]:
        """Stop timer and return elapsed time."""
        end_time = time.perf_counter()
        end_timestamp = datetime.datetime.now(timezone.utc)
        
        with self._lock:
            if timer_id not in self.active_timers:
                return None
            
            timer_info = self.active_timers.pop(timer_id)
            elapsed_time = end_time - timer_info['start_time']
            
            operation_name = timer_info['operation_name']
            
            # Store timing data
            if operation_name not in self.timing_data:
                self.timing_data[operation_name] = []
            
            self.timing_data[operation_name].append({
                'elapsed_time': elapsed_time,
                'start_timestamp': timer_info['start_timestamp'],
                'end_timestamp': end_timestamp
            })
        
        return elapsed_time
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        timer_id = self.start_timer(operation_name)
        try:
            yield timer_id
        finally:
            self.stop_timer(timer_id)
    
    def get_timing_statistics(self, operation_name: str) -> Optional[Dict[str, float]]:
        """Get timing statistics for specific operation."""
        with self._lock:
            if operation_name not in self.timing_data:
                return None
            
            timings = [entry['elapsed_time'] for entry in self.timing_data[operation_name]]
            
            return {
                'count': len(timings),
                'total_time': sum(timings),
                'average_time': np.mean(timings),
                'min_time': min(timings),
                'max_time': max(timings),
                'std_time': np.std(timings),
                'p50_time': np.percentile(timings, 50),
                'p95_time': np.percentile(timings, 95),
                'p99_time': np.percentile(timings, 99)
            }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all operations."""
        return {op_name: self.get_timing_statistics(op_name) 
                for op_name in self.timing_data.keys()}
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance timing report."""
        all_stats = self.get_all_statistics()
        
        # Find slowest operations
        slowest_operations = sorted(
            [(op, stats['average_time']) for op, stats in all_stats.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        # Find most frequent operations
        most_frequent = sorted(
            [(op, stats['count']) for op, stats in all_stats.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        # Calculate total profiled time
        total_profiled_time = sum(stats['total_time'] for stats in all_stats.values())
        
        return {
            'profiler_name': self.name,
            'total_operations': len(all_stats),
            'total_profiled_time': total_profiled_time,
            'slowest_operations': slowest_operations,
            'most_frequent_operations': most_frequent,
            'detailed_statistics': all_stats,
            'report_timestamp': datetime.datetime.now(timezone.utc).isoformat()
        }

class TimeWindowManager:
    """Manage time-based windows for data processing and analysis."""
    
    def __init__(self, window_size: timedelta, overlap: timedelta = timedelta(0)):
        self.window_size = window_size
        self.overlap = overlap
        self.active_windows = {}
        self.window_callbacks = defaultdict(list)
        self._lock = threading.Lock()
    
    def create_sliding_window(self, window_id: str, 
                            start_time: Optional[datetime.datetime] = None) -> 'SlidingTimeWindow':
        """Create a new sliding time window."""
        start_time = start_time or datetime.datetime.now(timezone.utc)
        
        window = SlidingTimeWindow(
            window_id=window_id,
            window_size=self.window_size,
            overlap=self.overlap,
            start_time=start_time
        )
        
        with self._lock:
            self.active_windows[window_id] = window
        
        return window
    
    def add_data_to_window(self, window_id: str, timestamp: datetime.datetime, 
                          data: Any) -> bool:
        """Add data point to specified time window."""
        with self._lock:
            if window_id not in self.active_windows:
                return False
            
            window = self.active_windows[window_id]
            window.add_data_point(timestamp, data)
            
            # Check if window is complete and notify callbacks
            if window.is_complete():
                for callback in self.window_callbacks[window_id]:
                    try:
                        callback(window.get_window_data())
                    except Exception as e:
                        print(f"Window callback error: {e}")
        
        return True
    
    def register_window_callback(self, window_id: str, 
                               callback: Callable[[List[Tuple[datetime.datetime, Any]]], None]):
        """Register callback for window completion events."""
        self.window_callbacks[window_id].append(callback)

class SlidingTimeWindow:
    """Sliding time window for temporal data collection."""
    
    def __init__(self, window_id: str, window_size: timedelta,
                 overlap: timedelta, start_time: datetime.datetime):
        self.window_id = window_id
        self.window_size = window_size
        self.overlap = overlap
        self.start_time = start_time
        self.end_time = start_time + window_size
        self.data_points = []
        self.is_closed = False
    
    def add_data_point(self, timestamp: datetime.datetime, data: Any):
        """Add data point to window if within time range."""
        if self.is_closed:
            return False
        
        if self.start_time <= timestamp <= self.end_time:
            self.data_points.append((timestamp, data))
            return True
        
        return False
    
    def is_complete(self) -> bool:
        """Check if window time range is complete."""
        current_time = datetime.datetime.now(timezone.utc)
        return current_time >= self.end_time
    
    def get_window_data(self) -> List[Tuple[datetime.datetime, Any]]:
        """Get all data points in window, sorted by timestamp."""
        return sorted(self.data_points, key=lambda x: x[0])
    
    def close_window(self):
        """Manually close the window."""
        self.is_closed = True
    
    def get_data_density(self) -> float:
        """Calculate data point density (points per second)."""
        if not self.data_points:
            return 0.0
        
        return len(self.data_points) / self.window_size.total_seconds()

class RecurringEventGenerator:
    """Generate recurring events based on complex patterns."""
    
    @staticmethod
    def generate_cron_schedule(cron_expression: str, 
                             start_time: datetime.datetime,
                             end_time: datetime.datetime,
                             timezone_str: str = "UTC") -> List[datetime.datetime]:
        """Generate schedule based on cron expression."""
        # Simplified cron parser for basic patterns
        # Format: minute hour day month day_of_week
        parts = cron_expression.split()
        
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 parts")
        
        minute, hour, day, month, day_of_week = parts
        
        schedule = []
        current_time = start_time.replace(second=0, microsecond=0)
        target_timezone = ZoneInfo(timezone_str)
        
        while current_time < end_time:
            if RecurringEventGenerator._matches_cron_pattern(current_time, parts):
                localized_time = current_time.astimezone(target_timezone)
                schedule.append(localized_time)
            
            current_time += timedelta(minutes=1)
        
        return schedule
    
    @staticmethod
    def _matches_cron_pattern(dt: datetime.datetime, cron_parts: List[str]) -> bool:
        """Check if datetime matches cron pattern."""
        minute, hour, day, month, day_of_week = cron_parts
        
        # Check minute
        if minute != '*' and int(minute) != dt.minute:
            return False
        
        # Check hour
        if hour != '*' and int(hour) != dt.hour:
            return False
        
        # Check day of month
        if day != '*' and int(day) != dt.day:
            return False
        
        # Check month
        if month != '*' and int(month) != dt.month:
            return False
        
        # Check day of week (0=Monday in Python, 0=Sunday in cron)
        if day_of_week != '*':
            cron_dow = int(day_of_week)
            python_dow = (dt.weekday() + 1) % 7  # Convert to Sunday=0 format
            if cron_dow != python_dow:
                return False
        
        return True
    
    @staticmethod
    def generate_business_hours_schedule(start_date: datetime.date,
                                       end_date: datetime.date,
                                       business_start: datetime.time,
                                       business_end: datetime.time,
                                       interval_minutes: int = 30,
                                       exclude_weekends: bool = True,
                                       holidays: Optional[List[datetime.date]] = None) -> List[datetime.datetime]:
        """Generate schedule during business hours."""
        holidays = holidays or []
        schedule = []
        
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends if requested
            if exclude_weekends and current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            # Skip holidays
            if current_date in holidays:
                current_date += timedelta(days=1)
                continue
            
            # Generate time slots for this day
            current_time = datetime.datetime.combine(current_date, business_start)
            end_time = datetime.datetime.combine(current_date, business_end)
            
            while current_time < end_time:
                schedule.append(current_time)
                current_time += timedelta(minutes=interval_minutes)
            
            current_date += timedelta(days=1)
        
        return schedule

class ChronoUtils:
    """Chronological utilities for time-based operations."""
    
    @staticmethod
    def calculate_age(birth_date: datetime.date, 
                     reference_date: Optional[datetime.date] = None) -> Dict[str, int]:
        """Calculate age with detailed breakdown."""
        reference_date = reference_date or datetime.date.today()
        
        if birth_date > reference_date:
            raise ValueError("Birth date cannot be in the future")
        
        # Calculate years
        years = reference_date.year - birth_date.year
        
        # Adjust if birthday hasn't occurred this year
        birthday_this_year = birth_date.replace(year=reference_date.year)
        if reference_date < birthday_this_year:
            years -= 1
            last_birthday = birth_date.replace(year=reference_date.year - 1)
        else:
            last_birthday = birthday_this_year
        
        # Calculate months and days since last birthday
        days_since_birthday = (reference_date - last_birthday).days
        
        # Approximate months (varying month lengths make this imprecise)
        months = days_since_birthday // 30
        days = days_since_birthday % 30
        
        return {
            'years': years,
            'months': months,
            'days': days,
            'total_days': (reference_date - birth_date).days
        }
    
    @staticmethod
    def find_optimal_meeting_time(participant_timezones: List[str],
                                participant_business_hours: List[Tuple[datetime.time, datetime.time]],
                                meeting_duration: timedelta,
                                preferred_date: datetime.date) -> List[Dict[str, Any]]:
        """Find optimal meeting times across multiple timezones."""
        if len(participant_timezones) != len(participant_business_hours):
            raise ValueError("Participant timezones and business hours must have same length")
        
        optimal_times = []
        
        # Generate potential time slots (15-minute intervals)
        base_datetime = datetime.datetime.combine(preferred_date, datetime.time(0, 0))
        time_slots = []
        
        for hour in range(24):
            for quarter in range(4):
                slot_time = base_datetime + timedelta(hours=hour, minutes=quarter * 15)
                time_slots.append(slot_time)
        
        # Evaluate each time slot
        for slot_time in time_slots:
            slot_end_time = slot_time + meeting_duration
            participants_available = 0
            participant_local_times = []
            
            for i, (tz, (business_start, business_end)) in enumerate(
                zip(participant_timezones, participant_business_hours)
            ):
                # Convert to participant's timezone
                tz_obj = ZoneInfo(tz)
                local_start = slot_time.replace(tzinfo=timezone.utc).astimezone(tz_obj)
                local_end = slot_end_time.replace(tzinfo=timezone.utc).astimezone(tz_obj)
                
                # Check if within business hours
                if (business_start <= local_start.time() <= business_end and
                    business_start <= local_end.time() <= business_end and
                    local_start.weekday() < 5):  # Weekday check
                    participants_available += 1
                    participant_local_times.append({
                        'participant_index': i,
                        'local_time': local_start,
                        'timezone': tz
                    })
            
            # Consider time slot if most participants available
            availability_ratio = participants_available / len(participant_timezones)
            
            if availability_ratio >= 0.5:  # At least 50% availability
                optimal_times.append({
                    'utc_time': slot_time,
                    'end_time': slot_end_time,
                    'participants_available': participants_available,
                    'availability_ratio': availability_ratio,
                    'participant_local_times': participant_local_times
                })
        
        # Sort by availability ratio and return best options
        optimal_times.sort(key=lambda x: x['availability_ratio'], reverse=True)
        return optimal_times[:10]  # Return top 10 options
    
    @staticmethod
    def calculate_time_until_next_occurrence(target_time: datetime.time,
                                          target_timezone: str = "UTC",
                                          reference_time: Optional[datetime.datetime] = None) -> timedelta:
        """Calculate time until next occurrence of daily time."""
        reference_time = reference_time or datetime.datetime.now(timezone.utc)
        target_tz = ZoneInfo(target_timezone)
        
        # Convert reference time to target timezone
        local_reference = reference_time.astimezone(target_tz)
        
        # Calculate target datetime for today
        target_today = datetime.datetime.combine(local_reference.date(), target_time)
        target_today = target_today.replace(tzinfo=target_tz)
        
        # If target time already passed today, use tomorrow
        if target_today <= local_reference:
            target_today += timedelta(days=1)
        
        # Convert back to UTC for calculation
        target_utc = target_today.astimezone(timezone.utc)
        reference_utc = reference_time.astimezone(timezone.utc)
        
        return target_utc - reference_utc

class TemporalCache:
    """Time-based cache with automatic expiration."""
    
    def __init__(self, default_ttl: timedelta = timedelta(hours=1)):
        self.default_ttl = default_ttl
        self.cache_data = {}
        self.expiry_times = {}
        self.access_times = {}
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._should_cleanup = True
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup."""
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup loop for expired entries."""
        while self._should_cleanup:
            try:
                self._cleanup_expired_entries()
                time.sleep(60)  # Cleanup every minute
            except Exception as e:
                print(f"Cache cleanup error: {e}")
                time.sleep(5)
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, expiry_time in self.expiry_times.items():
                if current_time > expiry_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.cache_data.pop(key, None)
                self.expiry_times.pop(key, None)
                self.access_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        current_time = time.time()
        
        with self._lock:
            if key in self.cache_data and current_time <= self.expiry_times[key]:
                self.access_times[key] = current_time
                return self.cache_data[key]
        
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None):
        """Put value in cache with specified TTL."""
        ttl = ttl or self.default_ttl
        current_time = time.time()
        expiry_time = current_time + ttl.total_seconds()
        
        with self._lock:
            self.cache_data[key] = value
            self.expiry_times[key] = expiry_time
            self.access_times[key] = current_time
    
    def invalidate(self, key: str) -> bool:
        """Manually invalidate cache entry."""
        with self._lock:
            if key in self.cache_data:
                del self.cache_data[key]
                self.expiry_times.pop(key, None)
                self.access_times.pop(key, None)
                return True
        
        return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache_data.clear()
            self.expiry_times.clear()
            self.access_times.clear()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        current_time = time.time()
        
        with self._lock:
            total_entries = len(self.cache_data)
            
            if total_entries == 0:
                return {
                    'total_entries': 0,
                    'average_age': 0,
                    'oldest_entry_age': 0,
                    'newest_entry_age': 0
                }
            
            ages = [current_time - access_time for access_time in self.access_times.values()]
            
            return {
                'total_entries': total_entries,
                'average_age': np.mean(ages),
                'oldest_entry_age': max(ages),
                'newest_entry_age': min(ages),
                'memory_estimate_kb': len(str(self.cache_data)) / 1024
            }

# Custom exceptions for time utilities
class TimeUtilsError(Exception):
    """Base exception for time utility errors."""
    pass

class TimeSyncError(TimeUtilsError):
    """Exception for time synchronization errors."""
    pass

class SchedulingError(TimeUtilsError):
    """Exception for event scheduling errors."""
    pass

class TimezoneError(TimeUtilsError):
    """Exception for timezone operation errors."""
    pass

class CalendarIntegrationError(TimeUtilsError):
    """Exception for calendar integration errors."""
    pass

# Global instances for system-wide time management
global_timezone_manager = TimezoneManager()
global_time_sync = TimeSync()
global_timing_profiler = TimingProfiler("global_profiler")
global_temporal_cache = TemporalCache()

# Utility functions for common time operations
def get_current_utc_time() -> datetime.datetime:
    """Get current UTC time with microsecond precision."""
    return datetime.datetime.now(timezone.utc)

def get_current_local_time(timezone_str: str) -> datetime.datetime:
    """Get current time in specified timezone."""
    utc_time = get_current_utc_time()
    return global_timezone_manager.convert_timezone(utc_time, timezone_str)

def parse_flexible_datetime(datetime_input: Union[str, int, float, datetime.datetime]) -> datetime.datetime:
    """Parse datetime from various input formats."""
    if isinstance(datetime_input, datetime.datetime):
        return datetime_input.replace(tzinfo=timezone.utc) if datetime_input.tzinfo is None else datetime_input
    
    elif isinstance(datetime_input, (int, float)):
        return datetime.datetime.fromtimestamp(datetime_input, tz=timezone.utc)
    
    elif isinstance(datetime_input, str):
        # Try multiple common formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y"
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.datetime.strptime(datetime_input, fmt)
                return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
            except ValueError:
                continue
        
        # Try ISO format parsing
        try:
            return datetime.datetime.fromisoformat(datetime_input.replace('Z', '+00:00'))
        except ValueError:
            pass
        
        raise ValueError(f"Unable to parse datetime: {datetime_input}")
    
    else:
        raise TypeError(f"Unsupported datetime input type: {type(datetime_input)}")

def create_time_range(start: Union[str, datetime.datetime], 
                     end: Union[str, datetime.datetime]) -> TimeInterval:
    """Create time range from flexible datetime inputs."""
    start_dt = parse_flexible_datetime(start)
    end_dt = parse_flexible_datetime(end)
    
    return TimeInterval(start_dt, end_dt)

def schedule_delayed_callback(callback: Callable[[], None], delay: timedelta) -> str:
    """Schedule callback to execute after specified delay."""
    callback_id = str(uuid.uuid4())
    
    def delayed_execution():
        time.sleep(delay.total_seconds())
        try:
            callback()
        except Exception as e:
            print(f"Delayed callback error: {e}")
    
    thread = threading.Thread(target=delayed_execution, daemon=True)
    thread.start()
    
    return callback_id

def benchmark_function_timing(func: Callable, iterations: int = 100,
                           warmup_iterations: int = 10) -> Dict[str, float]:
    """Benchmark function execution timing."""
    # Warmup runs
    for _ in range(warmup_iterations):
        try:
            func()
        except Exception:
            pass
    
    # Actual timing runs
    execution_times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        try:
            func()
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)
        except Exception as e:
            print(f"Benchmark execution error: {e}")
    
    if not execution_times:
        return {'error': 'All benchmark iterations failed'}
    
    return {
        'iterations': len(execution_times),
        'total_time': sum(execution_times),
        'average_time': np.mean(execution_times),
        'min_time': min(execution_times),
        'max_time': max(execution_times),
        'std_time': np.std(execution_times),
        'p50_time': np.percentile(execution_times, 50),
        'p95_time': np.percentile(execution_times, 95),
        'p99_time': np.percentile(execution_times, 99)
    }

def create_smart_reminder_system() -> 'SmartReminderSystem':
    """Create intelligent reminder system with context awareness."""
    return SmartReminderSystem(global_timezone_manager)

class SmartReminderSystem:
    """Intelligent reminder system with context-aware scheduling."""
    
    def __init__(self, timezone_manager: TimezoneManager):
        self.timezone_manager = timezone_manager
        self.reminders = {}
        self.user_patterns = {}
        self._lock = threading.Lock()
    
    def add_reminder(self, reminder_text: str, target_time: datetime.datetime,
                    user_id: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Add intelligent reminder with context awareness."""
        reminder_id = str(uuid.uuid4())
        
        reminder = {
            'id': reminder_id,
            'text': reminder_text,
            'target_time': target_time,
            'user_id': user_id,
            'context': context or {},
            'created_time': datetime.datetime.now(timezone.utc),
            'is_delivered': False,
            'optimal_delivery_time': self._calculate_optimal_delivery_time(
                user_id, target_time, context
            )
        }
        
        with self._lock:
            self.reminders[reminder_id] = reminder
        
        return reminder_id
    
    def _calculate_optimal_delivery_time(self, user_id: str, 
                                       target_time: datetime.datetime,
                                       context: Optional[Dict[str, Any]]) -> datetime.datetime:
        """Calculate optimal time to deliver reminder based on user patterns."""
        # Get user activity patterns
        patterns = self.user_patterns.get(user_id, {})
        
        # Default advance notice based on reminder type
        default_advance = timedelta(minutes=15)
        
        if context:
            reminder_type = context.get('type', 'general')
            
            if reminder_type == 'meeting':
                default_advance = timedelta(minutes=5)
            elif reminder_type == 'medication':
                default_advance = timedelta(minutes=0)  # Exact time
            elif reminder_type == 'task':
                default_advance = timedelta(minutes=30)
            elif reminder_type == 'event':
                default_advance = timedelta(hours=1)
        
        # Adjust based on user patterns
        user_preference = patterns.get('reminder_advance_minutes', default_advance.total_seconds() / 60)
        advance_time = timedelta(minutes=user_preference)
        
        # Consider user's typical active hours
        active_hours = patterns.get('active_hours', (7, 23))  # 7 AM to 11 PM default
        
        optimal_time = target_time - advance_time
        optimal_hour = optimal_time.hour
        
        # Adjust if outside active hours
        if optimal_hour < active_hours[0]:
            optimal_time = optimal_time.replace(hour=active_hours[0], minute=0)
        elif optimal_hour > active_hours[1]:
            # Deliver earlier the same day or next morning
            if optimal_time.date() == target_time.date():
                optimal_time = optimal_time.replace(hour=active_hours[1])
            else:
                next_day = optimal_time.date() + timedelta(days=1)
                optimal_time = datetime.datetime.combine(next_day, datetime.time(active_hours[0], 0))
                optimal_time = optimal_time.replace(tzinfo=optimal_time.tzinfo)
        
        return optimal_time
    
    def update_user_patterns(self, user_id: str, interaction_time: datetime.datetime,
                           response_time: Optional[float] = None):
        """Update user interaction patterns for better reminder timing."""
        with self._lock:
            if user_id not in self.user_patterns:
                self.user_patterns[user_id] = {
                    'active_hours': [7, 23],
                    'reminder_advance_minutes': 15,
                    'interaction_history': deque(maxlen=100),
                    'response_times': deque(maxlen=50)
                }
            
            patterns = self.user_patterns[user_id]
            patterns['interaction_history'].append(interaction_time)
            
            if response_time:
                patterns['response_times'].append(response_time)
            
            # Update active hours based on interaction history
            if len(patterns['interaction_history']) >= 10:
                interaction_hours = [dt.hour for dt in patterns['interaction_history']]
                patterns['active_hours'] = [
                    min(interaction_hours),
                    max(interaction_hours)
                ]
            
            # Adjust reminder advance based on response patterns
            if len(patterns['response_times']) >= 10:
                avg_response = np.mean(patterns['response_times'])
                if avg_response < 60:  # Quick responder
                    patterns['reminder_advance_minutes'] = 5
                elif avg_response > 300:  # Slow responder
                    patterns['reminder_advance_minutes'] = 60
    
    def get_pending_reminders(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pending reminders for user or all users."""
        current_time = datetime.datetime.now(timezone.utc)
        pending = []
        
        with self._lock:
            for reminder in self.reminders.values():
                if (reminder['is_delivered'] or 
                    reminder['optimal_delivery_time'] > current_time):
                    continue
                
                if user_id is None or reminder['user_id'] == user_id:
                    pending.append(reminder.copy())
        
        return sorted(pending, key=lambda r: r['optimal_delivery_time'])

# Performance monitoring decorators
def time_function_execution(operation_name: Optional[str] = None):
    """Decorator to automatically time function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with global_timing_profiler.time_operation(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def cache_with_ttl(ttl: timedelta, cache_key_generator: Optional[Callable] = None):
    """Decorator for caching function results with TTL."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_generator:
                cache_key = cache_key_generator(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Check cache
            cached_result = global_temporal_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            global_temporal_cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage and testing
    print("AI Holographic Wristwatch - Time Utilities Module")
    print("Testing time management capabilities...")
    
    # Test timezone management
    current_utc = get_current_utc_time()
    current_ny = get_current_local_time("America/New_York")
    print(f"UTC time: {current_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"NY time: {current_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Test event scheduling
    test_event = ScheduledEvent(
        title="Test Meeting",
        description="Daily standup meeting",
        start_time=datetime.datetime.now(timezone.utc) + timedelta(minutes=5),
        end_time=datetime.datetime.now(timezone.utc) + timedelta(minutes=35),
        recurrence=RecurrencePattern.DAILY
    )
    
    scheduler = EventScheduler(global_timezone_manager)
    scheduled = scheduler.schedule_event(test_event)
    print(f"Event scheduled successfully: {scheduled}")
    
    # Test duration calculations
    duration_str = "2 hours 30 minutes"
    parsed_duration = DurationCalculator.parse_duration_string(duration_str)
    formatted_duration = DurationCalculator.format_duration(parsed_duration)
    print(f"Duration parsing: '{duration_str}' -> {formatted_duration}")
    
    # Test temporal cache
    global_temporal_cache.put("test_key", {"data": "test_value"}, timedelta(seconds=5))
    cached_value = global_temporal_cache.get("test_key")
    print(f"Temporal cache test: {cached_value}")
    
    # Test timing profiler
    with global_timing_profiler.time_operation("test_operation"):
        time.sleep(0.01)  # Simulate work
    
    timing_stats = global_timing_profiler.get_timing_statistics("test_operation")
    print(f"Timing profiler test: {timing_stats}")
    
    print("Time utilities module initialized successfully.")