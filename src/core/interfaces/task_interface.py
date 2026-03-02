"""
Task Automation Interface Contracts for AI Holographic Wristwatch System

Defines the abstract contracts for the task execution, scheduling, and
workflow orchestration subsystems. The AI assistant uses these interfaces
to execute device actions, smart home commands, calendar operations, and
multi-step workflows without knowing which concrete backend is in use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..exceptions import AISystemError


# ============================================================================
# Enumerations
# ============================================================================

class TaskStatus(Enum):
    """Lifecycle state of a task."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Scheduling priority for a task."""
    EMERGENCY = 0
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TriggerType(Enum):
    """What causes a scheduled task to fire."""
    IMMEDIATE = "immediate"
    SCHEDULED_TIME = "scheduled_time"
    INTERVAL = "interval"
    CRON = "cron"
    EVENT = "event"
    SENSOR_THRESHOLD = "sensor_threshold"
    LOCATION = "location"
    CONTEXT = "context"


class TaskCategory(Enum):
    """Broad category for task classification."""
    DEVICE_CONTROL = "device_control"
    SMART_HOME = "smart_home"
    CALENDAR = "calendar"
    COMMUNICATION = "communication"
    HEALTH = "health"
    PRODUCTIVITY = "productivity"
    NAVIGATION = "navigation"
    MEDIA = "media"
    FINANCE = "finance"
    SYSTEM = "system"
    CUSTOM = "custom"


# ============================================================================
# Data Containers
# ============================================================================

@dataclass
class TaskTrigger:
    """Defines when a task should execute."""
    trigger_type: TriggerType
    scheduled_time: Optional[datetime] = None        # for SCHEDULED_TIME
    interval_seconds: Optional[float] = None         # for INTERVAL
    cron_expression: Optional[str] = None            # for CRON (e.g., "0 8 * * *")
    event_name: Optional[str] = None                 # for EVENT
    event_filter: Optional[Dict[str, Any]] = None    # event payload filter
    sensor_id: Optional[str] = None                  # for SENSOR_THRESHOLD
    threshold_value: Optional[float] = None          # threshold trigger level
    location_name: Optional[str] = None              # for LOCATION
    location_radius_m: Optional[float] = None
    max_executions: Optional[int] = None             # None = unlimited
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A unit of work to be executed by the task automation system."""
    task_id: str
    name: str
    category: TaskCategory
    action: str                                      # action handler identifier
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    trigger: Optional[TaskTrigger] = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    requires_confirmation: bool = False
    requires_permissions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None                # user_id or "ai_assistant"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Outcome of a completed or failed task execution."""
    task_id: str
    status: TaskStatus
    output: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_successful(self) -> bool:
        """Return True if the task completed without error."""
        return self.status == TaskStatus.COMPLETED and self.error_message is None


@dataclass
class WorkflowStep:
    """One step in a multi-step workflow."""
    step_id: str
    step_name: str
    task: Task
    depends_on: List[str] = field(default_factory=list)  # step_ids
    condition: Optional[str] = None      # expression evaluated before execution
    on_success: Optional[str] = None     # next step_id
    on_failure: Optional[str] = None     # fallback step_id


@dataclass
class Workflow:
    """A directed acyclic graph of tasks."""
    workflow_id: str
    name: str
    steps: List[WorkflowStep]
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WorkflowResult:
    """Outcome of an entire workflow execution."""
    workflow_id: str
    status: TaskStatus
    step_results: Dict[str, TaskResult] = field(default_factory=dict)
    total_execution_time_ms: float = 0.0
    failed_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledTask:
    """A task registered with the scheduler."""
    schedule_id: str
    task: Task
    trigger: TaskTrigger
    is_active: bool = True
    next_execution: Optional[datetime] = None
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# Task Executor Interface
# ============================================================================

class TaskExecutorInterface(ABC):
    """
    Contract for executing individual tasks.

    The executor handles action dispatch, permission checking, retry logic,
    and timeout enforcement. Concrete implementations may execute locally,
    delegate to cloud services, or interface with smart home hubs.
    """

    @abstractmethod
    async def execute(self, task: Task) -> TaskResult:
        """
        Execute a task immediately.

        Args:
            task: The Task to execute.
        Returns:
            TaskResult with status and output.
        Raises:
            AISystemError: If execution encounters an unhandled error.
        """
        ...

    @abstractmethod
    async def execute_with_confirmation(self, task: Task,
                                         confirmation_timeout_seconds: float = 30.0
                                         ) -> TaskResult:
        """
        Execute a task only after receiving user confirmation.

        Args:
            task:                         Task requiring user approval.
            confirmation_timeout_seconds: Max time to wait for confirmation.
        Returns:
            TaskResult — CANCELLED if user denies or times out.
        """
        ...

    @abstractmethod
    def cancel(self, task_id: str) -> bool:
        """
        Cancel a running or queued task.

        Args:
            task_id: ID of the task to cancel.
        Returns:
            True if cancelled successfully; False if task not found or already done.
        """
        ...

    @abstractmethod
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Return current status of a task.

        Args:
            task_id: Task identifier.
        Returns:
            TaskStatus, or None if task_id is unknown.
        """
        ...

    @abstractmethod
    def register_action_handler(self, action: str,
                                 handler: Callable) -> None:
        """
        Register a callable as the handler for a named action type.

        Args:
            action:  Action identifier string (e.g., "set_thermostat").
            handler: Async callable(task: Task) -> Any.
        """
        ...

    @abstractmethod
    def get_registered_actions(self) -> List[str]:
        """Return a list of all registered action identifiers."""
        ...

    @abstractmethod
    async def check_permissions(self, task: Task,
                                 user_id: str) -> bool:
        """
        Verify the user has the required permissions to run the task.

        Args:
            task:    The task to check.
            user_id: Requesting user identifier.
        Returns:
            True if all required permissions are granted.
        """
        ...


# ============================================================================
# Task Scheduler Interface
# ============================================================================

class TaskSchedulerInterface(ABC):
    """
    Contract for scheduling tasks to run at future times or on triggers.

    The scheduler maintains a priority queue of pending tasks and fires
    them when their trigger conditions are met.
    """

    @abstractmethod
    async def schedule(self, task: Task,
                       trigger: TaskTrigger) -> ScheduledTask:
        """
        Register a task with a trigger for future execution.

        Args:
            task:    The task definition.
            trigger: When and how the task should fire.
        Returns:
            ScheduledTask with assigned schedule_id and next_execution.
        """
        ...

    @abstractmethod
    async def cancel_schedule(self, schedule_id: str) -> bool:
        """
        Cancel a scheduled task by its schedule ID.

        Args:
            schedule_id: Identifier returned from schedule().
        Returns:
            True if found and cancelled.
        """
        ...

    @abstractmethod
    def get_scheduled_tasks(self,
                             category: Optional[TaskCategory] = None
                             ) -> List[ScheduledTask]:
        """
        List all registered scheduled tasks.

        Args:
            category: Optional filter by task category.
        Returns:
            List of ScheduledTask instances.
        """
        ...

    @abstractmethod
    def get_next_execution(self, schedule_id: str) -> Optional[datetime]:
        """
        Return the next scheduled execution time for a task.

        Args:
            schedule_id: Schedule identifier.
        Returns:
            Next execution datetime, or None if not scheduled.
        """
        ...

    @abstractmethod
    async def trigger_event(self, event_name: str,
                             event_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Fire all tasks listening for a named event.

        Args:
            event_name: Name of the event to fire.
            event_data: Optional payload passed to triggered tasks.
        Returns:
            Number of tasks triggered.
        """
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the scheduler event loop."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the scheduler and cancel all pending tasks."""
        ...


# ============================================================================
# Workflow Orchestrator Interface
# ============================================================================

class WorkflowOrchestratorInterface(ABC):
    """
    Contract for orchestrating multi-step, conditional task workflows.

    The orchestrator builds execution plans, manages dependencies between
    steps, handles partial failures, and provides rollback capabilities.
    """

    @abstractmethod
    async def execute_workflow(self, workflow: Workflow) -> WorkflowResult:
        """
        Execute all steps of a workflow, respecting dependencies.

        Args:
            workflow: Workflow DAG definition.
        Returns:
            WorkflowResult with per-step outcomes.
        """
        ...

    @abstractmethod
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow at the next safe checkpoint."""
        ...

    @abstractmethod
    async def resume_workflow(self, workflow_id: str) -> WorkflowResult:
        """Resume a paused workflow from where it stopped."""
        ...

    @abstractmethod
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow and undo completed steps if reversible."""
        ...

    @abstractmethod
    def get_workflow_status(self, workflow_id: str) -> Optional[TaskStatus]:
        """Return current status of a workflow execution."""
        ...

    @abstractmethod
    async def build_workflow_from_intent(self, intent: str,
                                          parameters: Dict[str, Any]
                                          ) -> Workflow:
        """
        Automatically construct a workflow from a natural language intent.

        Args:
            intent:     Intent identifier (e.g., "morning_routine").
            parameters: Parameters extracted from user input.
        Returns:
            A Workflow ready for execution.
        """
        ...


# ============================================================================
# Smart Home Control Interface
# ============================================================================

class SmartHomeControlInterface(ABC):
    """Contract for controlling smart home devices and platforms."""

    @abstractmethod
    async def get_devices(self) -> List[Dict[str, Any]]:
        """Return all discovered smart home devices."""
        ...

    @abstractmethod
    async def control_device(self, device_id: str,
                              command: str,
                              parameters: Optional[Dict[str, Any]] = None
                              ) -> TaskResult:
        """
        Send a command to a smart home device.

        Args:
            device_id:  Device identifier.
            command:    Command name (e.g., "turn_on", "set_temperature").
            parameters: Command parameters.
        Returns:
            TaskResult with execution outcome.
        """
        ...

    @abstractmethod
    async def get_device_state(self, device_id: str) -> Dict[str, Any]:
        """Return current state of a device."""
        ...

    @abstractmethod
    async def create_scene(self, scene_name: str,
                            device_states: List[Dict[str, Any]]) -> str:
        """Create a smart home scene. Returns scene_id."""
        ...

    @abstractmethod
    async def activate_scene(self, scene_id: str) -> TaskResult:
        """Activate a previously created scene."""
        ...

    @abstractmethod
    def get_supported_platforms(self) -> List[str]:
        """Return list of supported smart home platform names."""
        ...


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "TaskStatus", "TaskPriority", "TriggerType", "TaskCategory",
    "TaskTrigger", "Task", "TaskResult",
    "WorkflowStep", "Workflow", "WorkflowResult", "ScheduledTask",
    "TaskExecutorInterface", "TaskSchedulerInterface",
    "WorkflowOrchestratorInterface", "SmartHomeControlInterface",
]
