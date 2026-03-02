"""
Hardware Communication Package — AI Holographic Wristwatch

Aggregates all communication subsystems:
- Wireless (Bluetooth, Wi-Fi, NFC, Cellular, Mesh)
- Cloud Connectivity (sync, edge computing, backup, failover)
- Device Pairing (smartphone, computer, multi-device, smart home)
"""

from .wireless import (
    BluetoothManager, get_bluetooth_manager,
    WifiController, get_wifi_controller,
    NFCController, get_nfc_controller,
    CellularIntegration, get_cellular_integration,
    MeshNetworking, get_mesh_networking,
)

from .cloud_connectivity import (
    CloudSync, get_cloud_sync,
    EdgeComputingManager, get_edge_computing_manager,
    BackupSystem, get_backup_system,
    FailoverManager, get_failover_manager,
)

from .device_pairing import (
    SmartphoneIntegration, get_smartphone_integration,
    ComputerSync, get_computer_sync,
    MultiDeviceCoordinator, get_multi_device_coordinator,
    SmartHomeController, get_smart_home_controller,
)

__all__ = [
    # Wireless
    "BluetoothManager", "get_bluetooth_manager",
    "WifiController", "get_wifi_controller",
    "NFCController", "get_nfc_controller",
    "CellularIntegration", "get_cellular_integration",
    "MeshNetworking", "get_mesh_networking",
    # Cloud
    "CloudSync", "get_cloud_sync",
    "EdgeComputingManager", "get_edge_computing_manager",
    "BackupSystem", "get_backup_system",
    "FailoverManager", "get_failover_manager",
    # Device pairing
    "SmartphoneIntegration", "get_smartphone_integration",
    "ComputerSync", "get_computer_sync",
    "MultiDeviceCoordinator", "get_multi_device_coordinator",
    "SmartHomeController", "get_smart_home_controller",
]
