"""
Cloud Connectivity Sub-package — AI Holographic Wristwatch

Exports CloudSync, EdgeComputingManager, BackupSystem, and FailoverManager.
"""

from .cloud_sync import CloudSync, SyncStatus, SyncConflict, get_cloud_sync
from .edge_computing import EdgeComputingManager, TaskPriority, ComputeLocation, get_edge_computing_manager
from .backup_systems import BackupSystem, BackupRecord, BackupStatus, get_backup_system
from .failover_management import FailoverManager, ConnectionHealth, FailoverState, get_failover_manager

__all__ = [
    "CloudSync", "SyncStatus", "SyncConflict", "get_cloud_sync",
    "EdgeComputingManager", "TaskPriority", "ComputeLocation", "get_edge_computing_manager",
    "BackupSystem", "BackupRecord", "BackupStatus", "get_backup_system",
    "FailoverManager", "ConnectionHealth", "FailoverState", "get_failover_manager",
]
