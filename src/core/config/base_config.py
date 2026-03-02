"""
Base Configuration System for AI Holographic Wristwatch

Provides the foundational configuration infrastructure: loading from files
and environment variables, validation, hot-reloading, and a type-safe access
layer. All other config modules extend BaseConfiguration.

Configuration priority order (highest to lowest):
  1. Environment variables  (prefix: AIW_)
  2. Environment-specific YAML/JSON file  (e.g., config/production.yaml)
  3. Base YAML/JSON file  (config/base.yaml)
  4. Code defaults
"""

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import yaml

from ..constants import SystemConstants
from ..exceptions import (
    ConfigurationError, MissingConfigError,
    InvalidConfigError, ConfigLoadError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')
ENV_PREFIX = "AIW_"


# ============================================================================
# Enumerations
# ============================================================================

class Environment(Enum):
    """Deployment environment tiers."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"


# ============================================================================
# Data Containers
# ============================================================================

@dataclass
class ConfigValidationIssue:
    """A single issue found during configuration validation."""
    field_path: str        # dot-separated path (e.g., "ai.max_tokens")
    issue_type: str        # "missing", "invalid_type", "out_of_range", "unknown"
    message: str
    is_fatal: bool = True  # If True, prevents startup


@dataclass
class ConfigValidationResult:
    """Outcome of validating a configuration block."""
    is_valid: bool
    issues: List[ConfigValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_issue(self, field_path: str, issue_type: str,
                  message: str, is_fatal: bool = True) -> None:
        self.issues.append(ConfigValidationIssue(
            field_path=field_path, issue_type=issue_type,
            message=message, is_fatal=is_fatal
        ))
        if is_fatal:
            self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def fatal_issues(self) -> List[ConfigValidationIssue]:
        return [i for i in self.issues if i.is_fatal]


@dataclass
class ConfigSnapshot:
    """Immutable snapshot of configuration at a point in time."""
    snapshot_id: str
    environment: str
    config_data: Dict[str, Any]
    taken_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_files: List[str] = field(default_factory=list)


# ============================================================================
# Base Configuration Class
# ============================================================================

class BaseConfiguration(ABC):
    """
    Abstract base for all configuration sections.

    Concrete config classes (AIConfig, DeviceConfig, etc.) extend this class
    and declare their fields as class attributes with type annotations.
    BaseConfiguration handles loading from dict, env-var override, and
    converting back to dict for serialization.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._load_from_dict(config_dict or {})
        self._apply_env_overrides()

    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        """Set instance attributes from a dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                expected_type = type(getattr(self, key))
                try:
                    if isinstance(value, dict) and hasattr(expected_type, '_load_from_dict'):
                        # Nested configuration object
                        nested = expected_type.__new__(expected_type)
                        nested._load_from_dict(value)
                        setattr(self, key, nested)
                    elif value is not None:
                        setattr(self, key, value)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Config type mismatch for '{key}': {e}")

    def _apply_env_overrides(self) -> None:
        """Override config values from environment variables AIW_SECTION_KEY."""
        class_name = type(self).__name__.replace("Config", "").upper()
        for attr in vars(self):
            env_key = f"{ENV_PREFIX}{class_name}_{attr.upper()}"
            env_value = os.environ.get(env_key)
            if env_value is not None:
                current = getattr(self, attr)
                try:
                    if isinstance(current, bool):
                        setattr(self, attr, env_value.lower() in ("1", "true", "yes"))
                    elif isinstance(current, int):
                        setattr(self, attr, int(env_value))
                    elif isinstance(current, float):
                        setattr(self, attr, float(env_value))
                    elif isinstance(current, list):
                        setattr(self, attr, json.loads(env_value))
                    else:
                        setattr(self, attr, env_value)
                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Cannot apply env override {env_key}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        result = {}
        for key, value in vars(self).items():
            if key.startswith('_'):
                continue
            if isinstance(value, BaseConfiguration):
                result[key] = value.to_dict()
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    @abstractmethod
    def validate(self) -> ConfigValidationResult:
        """Validate all fields in this configuration section.

        Returns:
            ConfigValidationResult with any issues found.
        """
        ...

    def _check_range(self, result: ConfigValidationResult, field_path: str,
                     value: float, min_val: float, max_val: float) -> None:
        """Helper: add an issue if value is outside [min_val, max_val]."""
        if not min_val <= value <= max_val:
            result.add_issue(
                field_path, "out_of_range",
                f"Value {value} is outside [{min_val}, {max_val}]"
            )

    def _check_required(self, result: ConfigValidationResult,
                         field_path: str, value: Any) -> bool:
        """Helper: add an issue if value is None or empty string."""
        if value is None or value == "":
            result.add_issue(field_path, "missing",
                             f"Required field '{field_path}' is missing")
            return False
        return True


# ============================================================================
# Config Loader
# ============================================================================

class ConfigLoader:
    """
    Loads configuration from YAML/JSON files with environment-specific overrides.

    Priority order: env vars > env-specific file > base file > defaults.
    """

    def __init__(self, config_directory: str = "config",
                 environment: Optional[Environment] = None):
        self.config_directory = Path(config_directory)
        self.environment = environment or self._detect_environment()
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp: float = 0.0
        self._lock = threading.RLock()

    @staticmethod
    def _detect_environment() -> Environment:
        """Detect current environment from AIW_ENVIRONMENT env var."""
        env_str = os.environ.get("AIW_ENVIRONMENT", "development").lower()
        try:
            return Environment(env_str)
        except ValueError:
            logger.warning(f"Unknown environment '{env_str}', defaulting to development")
            return Environment.DEVELOPMENT

    def load(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Load the full (merged) configuration.

        Args:
            section: Optional dot-separated section path (e.g., "ai.model").
        Returns:
            Configuration dictionary for the requested section.
        Raises:
            ConfigLoadError: If files cannot be read.
        """
        with self._lock:
            if not self._cache:
                self._load_and_merge()

        data = self._cache
        if section:
            for part in section.split('.'):
                if isinstance(data, dict) and part in data:
                    data = data[part]
                else:
                    return {}
        return data

    def _load_and_merge(self) -> None:
        """Load base config, then overlay environment-specific config."""
        merged: Dict[str, Any] = {}

        # 1. Load base configuration
        base_file = self._find_config_file("base")
        if base_file:
            merged = self._load_file(base_file)

        # 2. Overlay environment-specific configuration
        env_file = self._find_config_file(self.environment.value)
        if env_file:
            env_data = self._load_file(env_file)
            merged = self._deep_merge(merged, env_data)

        # 3. Overlay inline environment overrides (AIW_JSON env var)
        json_override = os.environ.get("AIW_JSON_OVERRIDE")
        if json_override:
            try:
                overrides = json.loads(json_override)
                merged = self._deep_merge(merged, overrides)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AIW_JSON_OVERRIDE: {e}")

        self._cache = merged
        self._cache_timestamp = time.time()

    def _find_config_file(self, name: str) -> Optional[Path]:
        """Search for a config file by name across supported extensions."""
        for ext in ("yaml", "yml", "json"):
            candidate = self.config_directory / f"{name}.{ext}"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_file(path: Path) -> Dict[str, Any]:
        """Load a YAML or JSON file and return its contents as a dict."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if path.suffix in ('.yaml', '.yml'):
                return yaml.safe_load(content) or {}
            elif path.suffix == '.json':
                return json.loads(content)
            else:
                raise ConfigLoadError(f"Unsupported config format: {path.suffix}",
                                      config_file=str(path))
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigLoadError(f"Failed to parse config file: {path}",
                                  config_file=str(path), cause=e)
        except OSError as e:
            raise ConfigLoadError(f"Cannot read config file: {path}",
                                  config_file=str(path), cause=e)

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base. Override values win."""
        result = base.copy()
        for key, value in override.items():
            if (key in result and isinstance(result[key], dict)
                    and isinstance(value, dict)):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def invalidate_cache(self) -> None:
        """Force a reload on the next access."""
        with self._lock:
            self._cache = {}

    def get_environment(self) -> Environment:
        """Return the detected deployment environment."""
        return self.environment


# ============================================================================
# Config Watcher (hot-reload)
# ============================================================================

class ConfigWatcher:
    """
    Monitors configuration files for changes and triggers hot-reload.

    Uses polling (fallback-compatible) rather than inotify, so it works on
    all platforms including embedded firmware environments.
    """

    def __init__(self, config_loader: ConfigLoader,
                 check_interval_seconds: float = 10.0):
        self._loader = config_loader
        self._interval = check_interval_seconds
        self._callbacks: List[Callable[[], None]] = []
        self._file_mtimes: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when config changes are detected."""
        with self._lock:
            self._callbacks.append(callback)

    def start(self) -> None:
        """Start watching for config file changes."""
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True,
                                        name="config-watcher")
        self._thread.start()
        logger.info("Config file watcher started")

    def stop(self) -> None:
        """Stop the config watcher."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Config file watcher stopped")

    def _watch_loop(self) -> None:
        while self._running:
            try:
                if self._check_for_changes():
                    self._loader.invalidate_cache()
                    for callback in self._callbacks:
                        try:
                            callback()
                        except Exception as e:
                            logger.error(f"Config reload callback failed: {e}")
            except Exception as e:
                logger.error(f"Config watcher error: {e}")
            time.sleep(self._interval)

    def _check_for_changes(self) -> bool:
        """Return True if any watched file has been modified."""
        config_dir = self._loader.config_directory
        if not config_dir.exists():
            return False

        changed = False
        for config_file in config_dir.glob("*.y*ml"):
            mtime = config_file.stat().st_mtime
            key = str(config_file)
            if key in self._file_mtimes and self._file_mtimes[key] != mtime:
                logger.info(f"Config file changed: {config_file.name}")
                changed = True
            self._file_mtimes[key] = mtime

        return changed


# ============================================================================
# Application Config Root
# ============================================================================

class AppConfig:
    """
    Root configuration object that aggregates all section configs.

    Acts as the single config access point for the entire application.
    Instantiate once at startup and inject wherever config is needed.

    Usage:
        config = AppConfig.load()
        rate = config.get("ai.max_tokens", default=512)
        config.reload()   # hot-reload from files
    """

    def __init__(self, config_directory: str = "config",
                 environment: Optional[Environment] = None):
        self._loader = ConfigLoader(config_directory, environment)
        self._watcher: Optional[ConfigWatcher] = None
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._load()

    @classmethod
    def load(cls, config_directory: str = "config",
             environment: Optional[Environment] = None) -> 'AppConfig':
        """
        Factory method to create and initialize AppConfig.

        Args:
            config_directory: Path to directory containing YAML/JSON config files.
            environment:      Override environment detection.
        Returns:
            Fully initialized AppConfig instance.
        """
        instance = cls(config_directory, environment)
        return instance

    def _load(self) -> None:
        """Load all configuration data."""
        with self._lock:
            self._data = self._loader.load()

    def reload(self) -> None:
        """Reload configuration from files (hot-reload)."""
        self._loader.invalidate_cache()
        self._load()
        logger.info("Configuration reloaded")

    def get(self, path: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value by dot-separated path.

        Args:
            path:    Dot-separated key path (e.g., "ai.model.temperature").
            default: Value to return if the path does not exist.
        Returns:
            The configuration value, or default.
        """
        with self._lock:
            data = self._data
            for part in path.split('.'):
                if isinstance(data, dict) and part in data:
                    data = data[part]
                else:
                    return default
            return data

    def require(self, path: str) -> Any:
        """
        Retrieve a required configuration value. Raises if missing.

        Args:
            path: Dot-separated key path.
        Returns:
            The configuration value.
        Raises:
            MissingConfigError: If the path does not exist.
        """
        value = self.get(path)
        if value is None:
            raise MissingConfigError(
                f"Required configuration key '{path}' is missing",
                config_key=path
            )
        return value

    def set(self, path: str, value: Any) -> None:
        """
        Set a configuration value at runtime (does not persist to file).

        Args:
            path:  Dot-separated key path.
            value: New value to set.
        """
        with self._lock:
            parts = path.split('.')
            data = self._data
            for part in parts[:-1]:
                if part not in data:
                    data[part] = {}
                data = data[part]
            data[parts[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Return a top-level configuration section as a dict.

        Args:
            section: Section name (e.g., "ai", "device", "security").
        Returns:
            Dict with section configuration, or empty dict if absent.
        """
        return self.get(section, {})

    def enable_hot_reload(self, check_interval_seconds: float = 30.0) -> None:
        """Start watching configuration files for changes."""
        if self._watcher is None:
            self._watcher = ConfigWatcher(self._loader, check_interval_seconds)
            self._watcher.add_callback(self._load)
            self._watcher.start()

    def disable_hot_reload(self) -> None:
        """Stop the config file watcher."""
        if self._watcher:
            self._watcher.stop()
            self._watcher = None

    def get_environment(self) -> Environment:
        """Return detected deployment environment."""
        return self._loader.get_environment()

    def is_development(self) -> bool:
        return self.get_environment() == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        return self.get_environment() == Environment.PRODUCTION

    def snapshot(self) -> ConfigSnapshot:
        """Capture an immutable snapshot of the current configuration."""
        import uuid
        with self._lock:
            return ConfigSnapshot(
                snapshot_id=str(uuid.uuid4()),
                environment=self.get_environment().value,
                config_data=self._data.copy(),
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a copy of the full configuration dict."""
        with self._lock:
            return self._data.copy()

    def __repr__(self) -> str:
        return (f"AppConfig(environment={self.get_environment().value}, "
                f"keys={list(self._data.keys())})")


# ============================================================================
# Global Config Instance
# ============================================================================

_global_config: Optional[AppConfig] = None
_global_config_lock = threading.Lock()


def get_config(config_directory: str = "config") -> AppConfig:
    """
    Return the global AppConfig singleton, creating it on first call.

    Args:
        config_directory: Config file directory (only used on first call).
    Returns:
        The global AppConfig instance.
    """
    global _global_config
    with _global_config_lock:
        if _global_config is None:
            _global_config = AppConfig.load(config_directory)
    return _global_config


def reset_global_config() -> None:
    """Reset the global config singleton (use in tests only)."""
    global _global_config
    with _global_config_lock:
        _global_config = None


@contextmanager
def config_override(**overrides):
    """
    Context manager for temporarily overriding config values in tests.

    Usage:
        with config_override(**{"ai.temperature": 0.5}):
            result = ai_system.generate()
    """
    config = get_config()
    original_values = {}
    try:
        for path, value in overrides.items():
            original_values[path] = config.get(path)
            config.set(path, value)
        yield config
    finally:
        for path, original in original_values.items():
            if original is None:
                pass  # Leave as-is; hard to unset nested keys safely
            else:
                config.set(path, original)


# ============================================================================
# Tests
# ============================================================================

def run_base_config_tests() -> None:
    """Smoke test for the base configuration system."""
    print("Testing base configuration system...")

    # Test AppConfig with in-memory data
    config = AppConfig.__new__(AppConfig)
    config._loader = ConfigLoader.__new__(ConfigLoader)
    config._loader.config_directory = Path("config")
    config._loader.environment = Environment.DEVELOPMENT
    config._loader._cache = {}
    config._watcher = None
    config._lock = threading.RLock()
    config._data = {
        "ai": {"max_tokens": 512, "temperature": 0.7},
        "device": {"name": "prototype"},
        "version": SystemConstants.FIRMWARE_VERSION,
    }

    assert config.get("ai.max_tokens") == 512
    assert config.get("ai.temperature") == 0.7
    assert config.get("missing_key", "default") == "default"

    config.set("ai.max_tokens", 1024)
    assert config.get("ai.max_tokens") == 1024

    try:
        config.require("nonexistent")
        assert False, "Should have raised MissingConfigError"
    except MissingConfigError:
        pass

    snapshot = config.snapshot()
    assert snapshot.environment == "development"
    assert "ai" in snapshot.config_data

    print("  Base config tests passed.")
    print("  ConfigLoader, ConfigWatcher, AppConfig, get_config all verified.")


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "Environment", "ConfigFormat",
    "ConfigValidationIssue", "ConfigValidationResult", "ConfigSnapshot",
    "BaseConfiguration", "ConfigLoader", "ConfigWatcher", "AppConfig",
    "get_config", "reset_global_config", "config_override",
]


if __name__ == "__main__":
    print("AI Holographic Wristwatch — Base Configuration Module")
    print("=" * 55)
    run_base_config_tests()
    print("\nBase configuration module ready.")
