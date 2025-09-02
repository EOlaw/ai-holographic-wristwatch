# src/core/utils/logging_utils.py
"""
Advanced Logging Infrastructure for AI Holographic Wristwatch System

This module provides comprehensive logging capabilities including structured logging,
log level management, rotation policies, performance logging, security event logging,
AI decision logging, distributed tracing integration, and real-time log streaming.
"""

import logging
import logging.handlers
import json
import time
import threading
import asyncio
import traceback
import sys
import os
from typing import Tuple, Any, Dict, List, Optional, Union, Callable, TextIO
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from datetime import datetime, timezone
from pathlib import Path
import queue
import gzip
from contextlib import contextmanager
import functools

class LogLevel(Enum):
    """Enhanced log levels for the AI system."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    AI_DECISION = 25
    PERFORMANCE = 15
    SECURITY = 45
    AUDIT = 35

class LogCategory(Enum):
    """Log categories for better organization."""
    SYSTEM = "system"
    AI_CONVERSATION = "ai_conversation"
    SENSOR_DATA = "sensor_data"
    HOLOGRAPHIC = "holographic"
    SECURITY = "security" 
    PERFORMANCE = "performance"
    USER_INTERACTION = "user_interaction"
    DEVICE_HARDWARE = "device_hardware"
    NETWORK = "network"
    ERROR = "error"

@dataclass
class LogContext:
    """Contextual information for log entries."""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    execution_time: float
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    network_latency: Optional[float] = None
    cache_hit_rate: Optional[float] = None

@dataclass
class SecurityEvent:
    """Security event information for audit logging."""
    event_type: str
    severity: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    affected_resources: List[str] = field(default_factory=list)
    risk_score: Optional[int] = None
    mitigation_actions: List[str] = field(default_factory=list)

@dataclass
class StructuredLogEntry:
    """Structured log entry with comprehensive metadata."""
    timestamp: float = field(default_factory=time.time)
    level: str = "INFO"
    message: str = ""
    category: str = LogCategory.SYSTEM.value
    context: LogContext = field(default_factory=LogContext)
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Optional[PerformanceMetrics] = None
    security_event: Optional[SecurityEvent] = None
    exception_info: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for serialization."""
        entry_dict = asdict(self)
        entry_dict['timestamp'] = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        return entry_dict
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

class LogFormatter(logging.Formatter, ABC):
    """Abstract base class for custom log formatters."""
    
    @abstractmethod
    def format_structured(self, record: logging.LogRecord, 
                         log_entry: StructuredLogEntry) -> str:
        """Format structured log entry."""
        pass

class JSONFormatter(LogFormatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True, 
                 indent: Optional[int] = None):
        super().__init__()
        self.include_extra = include_extra
        self.indent = indent
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = getattr(record, 'structured_entry', None)
        
        if log_entry and isinstance(log_entry, StructuredLogEntry):
            return self.format_structured(record, log_entry)
        else:
            # Fallback for non-structured logs
            return self._format_standard_record(record)
    
    def format_structured(self, record: logging.LogRecord, 
                         log_entry: StructuredLogEntry) -> str:
        """Format structured log entry as JSON."""
        entry_dict = log_entry.to_dict()
        
        if not self.include_extra:
            entry_dict.pop('extra_fields', None)
        
        return json.dumps(entry_dict, indent=self.indent, default=str)
    
    def _format_standard_record(self, record: logging.LogRecord) -> str:
        """Format standard log record as JSON."""
        log_dict = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger_name': record.name,
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno
        }
        
        if record.exc_info:
            log_dict['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_dict, default=str)

class HumanReadableFormatter(LogFormatter):
    """Human-readable formatter for development and debugging."""
    
    def __init__(self, include_context: bool = True, 
                 include_performance: bool = True,
                 color_coding: bool = True):
        super().__init__()
        self.include_context = include_context
        self.include_performance = include_performance
        self.color_coding = color_coding
        
        # ANSI color codes
        self.colors = {
            'TRACE': '\033[90m',      # Dark gray
            'DEBUG': '\033[36m',      # Cyan
            'INFO': '\033[32m',       # Green
            'WARNING': '\033[33m',    # Yellow
            'ERROR': '\033[31m',      # Red
            'CRITICAL': '\033[35m',   # Magenta
            'SECURITY': '\033[41m',   # Red background
            'RESET': '\033[0m'        # Reset color
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record in human-readable format."""
        log_entry = getattr(record, 'structured_entry', None)
        
        if log_entry and isinstance(log_entry, StructuredLogEntry):
            return self.format_structured(record, log_entry)
        else:
            return self._format_standard_record(record)
    
    def format_structured(self, record: logging.LogRecord, 
                         log_entry: StructuredLogEntry) -> str:
        """Format structured log entry for human readability."""
        timestamp = datetime.fromtimestamp(log_entry.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Color coding
        color_start = ""
        color_end = ""
        if self.color_coding and log_entry.level in self.colors:
            color_start = self.colors[log_entry.level]
            color_end = self.colors['RESET']
        
        # Basic log line
        formatted = f"{color_start}{timestamp} [{log_entry.level:^9}] {log_entry.message}{color_end}"
        
        # Add category
        formatted += f" [cat:{log_entry.category}]"
        
        # Add context information
        if self.include_context and log_entry.context:
            context_parts = []
            if log_entry.context.user_id:
                context_parts.append(f"user:{log_entry.context.user_id[:8]}")
            if log_entry.context.device_id:
                context_parts.append(f"device:{log_entry.context.device_id[:8]}")
            if log_entry.context.component:
                context_parts.append(f"comp:{log_entry.context.component}")
            if log_entry.context.operation:
                context_parts.append(f"op:{log_entry.context.operation}")
            
            if context_parts:
                formatted += f" [{', '.join(context_parts)}]"
        
        # Add performance metrics
        if self.include_performance and log_entry.performance_metrics:
            metrics = log_entry.performance_metrics
            perf_parts = [f"time:{metrics.execution_time:.3f}s"]
            if metrics.memory_usage:
                perf_parts.append(f"mem:{metrics.memory_usage}KB")
            if metrics.cpu_usage:
                perf_parts.append(f"cpu:{metrics.cpu_usage:.1f}%")
            
            formatted += f" [perf: {', '.join(perf_parts)}]"
        
        # Add correlation ID
        if log_entry.context.correlation_id:
            formatted += f" [corr:{log_entry.context.correlation_id[:8]}]"
        
        # Add security event information
        if log_entry.security_event:
            formatted += f"\n  Security Event: {log_entry.security_event.event_type} "
            formatted += f"(severity: {log_entry.security_event.severity})"
            if log_entry.security_event.risk_score:
                formatted += f" Risk Score: {log_entry.security_event.risk_score}/10"
        
        # Add exception information
        if log_entry.exception_info:
            formatted += f"\n  Exception: {log_entry.exception_info['type']}: {log_entry.exception_info['message']}"
        
        # Add extra fields
        if log_entry.extra_fields:
            formatted += f"\n  Extra: {json.dumps(log_entry.extra_fields, default=str)}"
        
        return formatted
    
    def _format_standard_record(self, record: logging.LogRecord) -> str:
        """Format standard log record for human readability."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        color_start = ""
        color_end = ""
        if self.color_coding and record.levelname in self.colors:
            color_start = self.colors[record.levelname]
            color_end = self.colors['RESET']
        
        formatted = f"{color_start}{timestamp} [{record.levelname:^9}] {record.getMessage()}{color_end}"
        formatted += f" [{record.name}:{record.funcName}:{record.lineno}]"
        
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted

class AdvancedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Enhanced rotating file handler with compression and integrity checking."""
    
    def __init__(self, filename: str, mode: str = 'a', maxBytes: int = 100 * 1024 * 1024,
                 backupCount: int = 10, compress_rotated: bool = True,
                 integrity_check: bool = True):
        super().__init__(filename, mode, maxBytes, backupCount)
        self.compress_rotated = compress_rotated
        self.integrity_check = integrity_check
        self.file_checksums = {}
    
    def doRollover(self):
        """Perform log rotation with optional compression."""
        super().doRollover()
        
        # Compress rotated files if enabled
        if self.compress_rotated:
            self._compress_rotated_files()
        
        # Calculate checksums if integrity checking enabled
        if self.integrity_check:
            self._calculate_file_checksums()
    
    def _compress_rotated_files(self):
        """Compress rotated log files."""
        try:
            base_filename = self.baseFilename
            
            for i in range(1, self.backupCount + 1):
                rotated_file = f"{base_filename}.{i}"
                compressed_file = f"{rotated_file}.gz"
                
                if os.path.exists(rotated_file) and not os.path.exists(compressed_file):
                    with open(rotated_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            f_out.writelines(f_in)
                    
                    # Remove original file after compression
                    os.remove(rotated_file)
                    
        except Exception as e:
            # Log compression error to stderr to avoid recursion
            print(f"Log compression error: {e}", file=sys.stderr)
    
    def _calculate_file_checksums(self):
        """Calculate checksums for integrity verification."""
        try:
            import hashlib
            
            if os.path.exists(self.baseFilename):
                with open(self.baseFilename, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    self.file_checksums[self.baseFilename] = {
                        'checksum': file_hash,
                        'timestamp': time.time()
                    }
                    
        except Exception as e:
            print(f"Checksum calculation error: {e}", file=sys.stderr)

class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.queue_size = queue_size
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = None
        self.is_running = False
        self._start_worker()
    
    def _start_worker(self):
        """Start the asynchronous logging worker thread."""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def _worker_loop(self):
        """Worker thread loop for processing log records."""
        while self.is_running:
            try:
                # Get log record with timeout
                record = self.log_queue.get(timeout=1.0)
                
                if record is None:  # Shutdown signal
                    break
                
                # Process record with target handler
                self.target_handler.emit(record)
                self.log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                # Handle errors without causing recursion
                print(f"Async logging error: {e}", file=sys.stderr)
    
    def emit(self, record: logging.LogRecord):
        """Emit log record asynchronously."""
        try:
            if self.is_running:
                self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop log record if queue is full to prevent blocking
            print("Log queue full, dropping record", file=sys.stderr)
    
    def close(self):
        """Close the async handler gracefully."""
        self.is_running = False
        
        # Send shutdown signal
        try:
            self.log_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        # Close target handler
        self.target_handler.close()
        super().close()

class StreamingLogHandler(logging.Handler):
    """Real-time log streaming handler for monitoring systems."""
    
    def __init__(self, stream_callback: Callable[[Dict[str, Any]], None],
                 buffer_size: int = 100, flush_interval: float = 1.0):
        super().__init__()
        self.stream_callback = stream_callback
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.log_buffer = []
        self.last_flush = time.time()
        self._lock = threading.Lock()
    
    def emit(self, record: logging.LogRecord):
        """Buffer and stream log records."""
        try:
            # Convert record to dictionary
            log_dict = self._record_to_dict(record)
            
            with self._lock:
                self.log_buffer.append(log_dict)
                
                # Flush if buffer is full or interval elapsed
                current_time = time.time()
                should_flush = (len(self.log_buffer) >= self.buffer_size or 
                              current_time - self.last_flush >= self.flush_interval)
                
                if should_flush:
                    self._flush_buffer()
                    self.last_flush = current_time
                    
        except Exception as e:
            print(f"Streaming log error: {e}", file=sys.stderr)
    
    def _record_to_dict(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Convert log record to dictionary."""
        log_dict = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger_name': record.name,
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno
        }
        
        # Add structured entry if available
        structured_entry = getattr(record, 'structured_entry', None)
        if structured_entry:
            log_dict.update(structured_entry.to_dict())
        
        # Add exception info if present
        if record.exc_info:
            log_dict['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return log_dict
    
    def _flush_buffer(self):
        """Flush buffered log records to stream."""
        if not self.log_buffer:
            return
        
        try:
            # Send buffered logs to callback
            buffer_copy = self.log_buffer.copy()
            self.log_buffer.clear()
            
            # Call streaming callback with batch
            self.stream_callback(buffer_copy)
            
        except Exception as e:
            print(f"Log streaming callback error: {e}", file=sys.stderr)
    
    def flush(self):
        """Force flush of log buffer."""
        with self._lock:
            self._flush_buffer()
    
    def close(self):
        """Close handler and flush remaining logs."""
        self.flush()
        super().close()

class StructuredLogger:
    """Advanced structured logger with context management."""
    
    def __init__(self, name: str, base_context: Optional[LogContext] = None):
        self.logger = logging.getLogger(name)
        self.base_context = base_context or LogContext()
        self._context_stack = []
        self._local_context = threading.local()
    
    def _get_current_context(self) -> LogContext:
        """Get current logging context with inheritance."""
        # Start with base context
        context_dict = asdict(self.base_context)
        
        # Apply thread-local context
        local_context = getattr(self._local_context, 'context', None)
        if local_context:
            local_dict = asdict(local_context)
            context_dict.update({k: v for k, v in local_dict.items() if v is not None})
        
        return LogContext(**context_dict)
    
    @contextmanager
    def context(self, **context_fields):
        """Context manager for temporary logging context."""
        # Create new context with updated fields
        current_context = self._get_current_context()
        current_dict = asdict(current_context)
        current_dict.update(context_fields)
        new_context = LogContext(**current_dict)
        
        # Set thread-local context
        old_context = getattr(self._local_context, 'context', None)
        self._local_context.context = new_context
        
        try:
            yield
        finally:
            # Restore previous context
            self._local_context.context = old_context
    
    def set_context(self, **context_fields):
        """Set persistent context for current thread."""
        current_context = self._get_current_context()
        current_dict = asdict(current_context)
        current_dict.update(context_fields)
        self._local_context.context = LogContext(**current_dict)
    
    def clear_context(self):
        """Clear thread-local context."""
        self._local_context.context = None
    
    def _log_structured(self, level: str, message: str, category: LogCategory,
                       performance_metrics: Optional[PerformanceMetrics] = None,
                       security_event: Optional[SecurityEvent] = None,
                       exception_info: Optional[Exception] = None,
                       **extra_fields):
        """Log structured entry."""
        # Create structured log entry
        log_entry = StructuredLogEntry(
            level=level,
            message=message,
            category=category.value,
            context=self._get_current_context(),
            extra_fields=extra_fields,
            performance_metrics=performance_metrics,
            security_event=security_event
        )
        
        # Add exception information
        if exception_info:
            log_entry.exception_info = {
                'type': type(exception_info).__name__,
                'message': str(exception_info),
                'traceback': traceback.format_exc()
            }
        
        # Create log record
        record = logging.LogRecord(
            name=self.logger.name,
            level=getattr(logging, level),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        # Attach structured entry to record
        record.structured_entry = log_entry
        
        # Emit log record
        self.logger.handle(record)
    
    def trace(self, message: str, **kwargs):
        """Log trace level message."""
        self._log_structured("TRACE", message, LogCategory.SYSTEM, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self._log_structured("DEBUG", message, LogCategory.SYSTEM, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self._log_structured("INFO", message, LogCategory.SYSTEM, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self._log_structured("WARNING", message, LogCategory.SYSTEM, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error level message."""
        self._log_structured("ERROR", message, LogCategory.ERROR, 
                           exception_info=exception, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical level message."""
        self._log_structured("CRITICAL", message, LogCategory.ERROR,
                           exception_info=exception, **kwargs)
    
    def ai_decision(self, message: str, decision_data: Dict[str, Any], **kwargs):
        """Log AI decision with decision context."""
        kwargs.update(decision_data)
        self._log_structured("AI_DECISION", message, LogCategory.AI_CONVERSATION, **kwargs)
    
    def performance(self, message: str, metrics: PerformanceMetrics, **kwargs):
        """Log performance information."""
        self._log_structured("PERFORMANCE", message, LogCategory.PERFORMANCE,
                           performance_metrics=metrics, **kwargs)
    
    def security(self, message: str, security_event: SecurityEvent, **kwargs):
        """Log security event."""
        self._log_structured("SECURITY", message, LogCategory.SECURITY,
                           security_event=security_event, **kwargs)
    
    def audit(self, message: str, **kwargs):
        """Log audit information."""
        self._log_structured("AUDIT", message, LogCategory.SYSTEM, **kwargs)

class LoggingConfiguration:
    """Centralized logging configuration management."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict or self._default_config()
        self.configured_loggers = {}
        self.handlers = {}
    
    def _default_config(self) -> Dict[str, Any]:
        """Default logging configuration."""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': {
                    'class': 'logging_utils.JSONFormatter',
                    'indent': None
                },
                'human': {
                    'class': 'logging_utils.HumanReadableFormatter',
                    'include_context': True,
                    'color_coding': True
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'human',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging_utils.AdvancedRotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'json',
                    'filename': 'logs/ai_wristwatch.log',
                    'maxBytes': 100 * 1024 * 1024,  # 100MB
                    'backupCount': 10,
                    'compress_rotated': True
                },
                'security': {
                    'class': 'logging_utils.AdvancedRotatingFileHandler',
                    'level': 'SECURITY',
                    'formatter': 'json',
                    'filename': 'logs/security.log',
                    'maxBytes': 50 * 1024 * 1024,  # 50MB
                    'backupCount': 20
                },
                'performance': {
                    'class': 'logging_utils.AdvancedRotatingFileHandler',
                    'level': 'PERFORMANCE',
                    'formatter': 'json',
                    'filename': 'logs/performance.log',
                    'maxBytes': 200 * 1024 * 1024,  # 200MB
                    'backupCount': 5
                }
            },
            'loggers': {
                'ai_wristwatch': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file'],
                    'propagate': False
                },
                'ai_wristwatch.security': {
                    'level': 'INFO',
                    'handlers': ['security', 'console'],
                    'propagate': False
                },
                'ai_wristwatch.performance': {
                    'level': 'PERFORMANCE',
                    'handlers': ['performance'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'WARNING',
                'handlers': ['console']
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Create log directories
        self._create_log_directories()
        
        # Add custom log levels
        self._add_custom_log_levels()
        
        # Configure logging using dictConfig
        logging.config.dictConfig(self.config)
        
        # Setup signal handlers for log rotation
        self._setup_signal_handlers()
    
    def _create_log_directories(self):
        """Create necessary log directories."""
        log_files = []
        
        for handler_config in self.config.get('handlers', {}).values():
            if 'filename' in handler_config:
                log_files.append(handler_config['filename'])
        
        for log_file in log_files:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _add_custom_log_levels(self):
        """Add custom log levels to logging module."""
        for log_level in LogLevel:
            level_name = log_level.name
            level_value = log_level.value
            
            if not hasattr(logging, level_name):
                logging.addLevelName(level_value, level_name)
                
                # Add convenience method to Logger class
                def make_log_method(level_val):
                    def log_method(self, message, *args, **kwargs):
                        if self.isEnabledFor(level_val):
                            self._log(level_val, message, args, **kwargs)
                    return log_method
                
                setattr(logging.Logger, level_name.lower(), make_log_method(level_value))
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful log rotation."""
        import signal
        
        def rotate_logs(signum, frame):
            """Signal handler for log rotation."""
            for handler in logging.getLogger().handlers:
                if hasattr(handler, 'doRollover'):
                    handler.doRollover()
        
        # Register signal handler for SIGUSR1 (if available)
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, rotate_logs)
    
    def create_structured_logger(self, name: str, 
                                context: Optional[LogContext] = None) -> StructuredLogger:
        """Create structured logger with configuration."""
        if name not in self.configured_loggers:
            logger = StructuredLogger(name, context)
            logger.logger.setLevel(logging.DEBUG)
            self.configured_loggers[name] = logger
        
        return self.configured_loggers[name]
    
    def add_streaming_handler(self, logger_name: str, 
                            stream_callback: Callable[[Dict[str, Any]], None]):
        """Add streaming handler to specific logger."""
        logger = logging.getLogger(logger_name)
        handler = StreamingLogHandler(stream_callback)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        return handler
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging system statistics."""
        stats = {
            'configured_loggers': len(self.configured_loggers),
            'active_handlers': 0,
            'log_levels_used': set(),
            'handler_types': {}
        }
        
        for logger_name, logger in self.configured_loggers.items():
            for handler in logger.logger.handlers:
                stats['active_handlers'] += 1
                handler_type = type(handler).__name__
                stats['handler_types'][handler_type] = stats['handler_types'].get(handler_type, 0) + 1
        
        return stats

class PerformanceLogger:
    """Specialized logger for performance monitoring."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.operation_timings = {}
    
    @contextmanager
    def time_operation(self, operation_name: str, **context):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            with self.logger.context(operation=operation_name, **context):
                yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory if start_memory and end_memory else None
            )
            
            self.logger.performance(
                f"Operation '{operation_name}' completed",
                metrics,
                operation_name=operation_name
            )
            
            # Track operation timings
            if operation_name not in self.operation_timings:
                self.operation_timings[operation_name] = []
            self.operation_timings[operation_name].append(metrics.execution_time)
    
    def _get_memory_usage(self) -> Optional[int]:
        """Get current memory usage in KB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss // 1024
        except ImportError:
            return None
    
    def log_ai_inference_performance(self, model_name: str, input_size: int,
                                   inference_time: float, accuracy: Optional[float] = None):
        """Log AI inference performance metrics."""
        metrics = PerformanceMetrics(execution_time=inference_time)
        
        self.logger.performance(
            f"AI inference completed for model {model_name}",
            metrics,
            model_name=model_name,
            input_size=input_size,
            accuracy=accuracy
        )
    
    def log_sensor_processing_performance(self, sensor_type: str, data_points: int,
                                        processing_time: float, quality_score: float):
        """Log sensor data processing performance."""
        metrics = PerformanceMetrics(execution_time=processing_time)
        
        self.logger.performance(
            f"Sensor data processing completed for {sensor_type}",
            metrics,
            sensor_type=sensor_type,
            data_points=data_points,
            quality_score=quality_score
        )
    
    def log_holographic_rendering_performance(self, frame_count: int, render_time: float,
                                            resolution: Tuple[int, int], quality: str):
        """Log holographic rendering performance."""
        metrics = PerformanceMetrics(execution_time=render_time)
        
        self.logger.performance(
            f"Holographic rendering completed: {frame_count} frames",
            metrics,
            frame_count=frame_count,
            resolution=f"{resolution[0]}x{resolution[1]}",
            quality_level=quality,
            fps=frame_count / render_time if render_time > 0 else 0
        )

class SecurityAuditLogger:
    """Specialized logger for security events and audit trails."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.security_events = []
        self._lock = threading.Lock()
    
    def log_authentication_event(self, user_id: str, device_id: str, 
                                success: bool, method: str,
                                source_ip: Optional[str] = None):
        """Log authentication event."""
        event_type = "authentication_success" if success else "authentication_failure"
        severity = "info" if success else "warning"
        
        security_event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            affected_resources=[f"user:{user_id}", f"device:{device_id}"]
        )
        
        self.logger.security(
            f"Authentication {event_type} for user {user_id[:8]} on device {device_id[:8]}",
            security_event,
            user_id=user_id,
            device_id=device_id,
            auth_method=method
        )
        
        with self._lock:
            self.security_events.append({
                'event': security_event,
                'timestamp': time.time()
            })
    
    def log_data_access_event(self, user_id: str, resource_type: str, 
                             resource_id: str, operation: str,
                             success: bool, data_size: Optional[int] = None):
        """Log data access event for audit trail."""
        event_type = f"data_access_{operation}"
        severity = "info" if success else "error"
        
        security_event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            affected_resources=[f"{resource_type}:{resource_id}"]
        )
        
        extra_fields = {
            'user_id': user_id,
            'resource_type': resource_type,
            'resource_id': resource_id,
            'operation': operation,
            'success': success
        }
        
        if data_size:
            extra_fields['data_size'] = data_size
        
        self.logger.audit(
            f"Data access: {operation} on {resource_type} {resource_id[:8]}",
            **extra_fields
        )
    
    def log_security_violation(self, violation_type: str, description: str,
                             risk_score: int, source_ip: Optional[str] = None,
                             mitigation_actions: Optional[List[str]] = None):
        """Log security violation with high priority."""
        security_event = SecurityEvent(
            event_type=f"security_violation_{violation_type}",
            severity="critical",
            source_ip=source_ip,
            risk_score=risk_score,
            mitigation_actions=mitigation_actions or []
        )
        
        self.logger.security(
            f"Security violation detected: {description}",
            security_event,
            violation_type=violation_type,
            risk_score=risk_score
        )
    
    def log_privacy_event(self, event_type: str, user_id: str, 
                         data_type: str, purpose: str):
        """Log privacy-related events for compliance."""
        security_event = SecurityEvent(
            event_type=f"privacy_{event_type}",
            severity="info",
            affected_resources=[f"user_data:{user_id}"]
        )
        
        self.logger.audit(
            f"Privacy event: {event_type} for user {user_id[:8]}",
            user_id=user_id,
            data_type=data_type,
            purpose=purpose
        )

class DistributedTracing:
    """Distributed tracing integration for microservices logging."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.active_spans = {}
        self._span_stack = threading.local()
    
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None) -> str:
        """Start a new tracing span."""
        span_id = str(uuid.uuid4())
        trace_id = parent_span_id or str(uuid.uuid4())
        
        span_info = {
            'span_id': span_id,
            'trace_id': trace_id,
            'operation_name': operation_name,
            'service_name': self.service_name,
            'start_time': time.time(),
            'parent_span_id': parent_span_id
        }
        
        self.active_spans[span_id] = span_info
        
        # Set as current span in thread-local storage
        if not hasattr(self._span_stack, 'spans'):
            self._span_stack.spans = []
        self._span_stack.spans.append(span_id)
        
        return span_id
    
    def finish_span(self, span_id: str, success: bool = True, 
                   error_message: Optional[str] = None):
        """Finish tracing span with result."""
        if span_id not in self.active_spans:
            return
        
        span_info = self.active_spans[span_id]
        span_info.update({
            'end_time': time.time(),
            'duration': time.time() - span_info['start_time'],
            'success': success,
            'error_message': error_message
        })
        
        # Remove from thread-local stack
        if hasattr(self._span_stack, 'spans') and span_id in self._span_stack.spans:
            self._span_stack.spans.remove(span_id)
        
        # Log span completion
        logger = get_logger("distributed_tracing")
        logger.debug(
            f"Span completed: {span_info['operation_name']}",
            category=LogCategory.PERFORMANCE,
            trace_id=span_info['trace_id'],
            span_id=span_id,
            duration=span_info['duration'],
            success=success
        )
        
        # Clean up completed span
        del self.active_spans[span_id]
    
    def get_current_span_id(self) -> Optional[str]:
        """Get current active span ID."""
        if hasattr(self._span_stack, 'spans') and self._span_stack.spans:
            return self._span_stack.spans[-1]
        return None
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID."""
        current_span_id = self.get_current_span_id()
        if current_span_id and current_span_id in self.active_spans:
            return self.active_spans[current_span_id]['trace_id']
        return None
    
    @contextmanager
    def span(self, operation_name: str, **context):
        """Context manager for automatic span lifecycle."""
        span_id = self.start_span(operation_name)
        
        try:
            yield span_id
            self.finish_span(span_id, success=True)
        except Exception as e:
            self.finish_span(span_id, success=False, error_message=str(e))
            raise

# Global configuration and logger instances
global_logging_config = LoggingConfiguration()
global_tracing = DistributedTracing("ai_holographic_wristwatch")

def get_logger(name: str, context: Optional[LogContext] = None) -> StructuredLogger:
    """Get structured logger instance with optional context."""
    return global_logging_config.create_structured_logger(name, context)

def setup_logging_environment(config_path: Optional[str] = None,
                            log_level: str = "INFO",
                            enable_console: bool = True,
                            enable_file: bool = True,
                            log_directory: str = "logs"):
    """Setup complete logging environment for the application."""
    # Load custom configuration if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
        global_logging_config.config.update(custom_config)
    
    # Override with environment parameters
    if not enable_console:
        global_logging_config.config['loggers']['ai_wristwatch']['handlers'].remove('console')
    
    if not enable_file:
        handlers = global_logging_config.config['loggers']['ai_wristwatch']['handlers']
        handlers[:] = [h for h in handlers if h != 'file']
    
    # Update log directory
    if log_directory != "logs":
        for handler_config in global_logging_config.config['handlers'].values():
            if 'filename' in handler_config:
                filename = Path(handler_config['filename']).name
                handler_config['filename'] = str(Path(log_directory) / filename)
    
    # Setup logging
    global_logging_config.setup_logging()
    
    # Log initialization
    init_logger = get_logger("logging_system")
    init_logger.info(
        "Logging system initialized",
        category=LogCategory.SYSTEM,
        log_level=log_level,
        console_enabled=enable_console,
        file_enabled=enable_file,
        log_directory=log_directory
    )

def performance_logger_decorator(operation_name: str, category: LogCategory = LogCategory.PERFORMANCE):
    """Decorator for automatic performance logging."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f"performance.{func.__module__}")
            perf_logger = PerformanceLogger(logger)
            
            with perf_logger.time_operation(operation_name, function_name=func.__name__):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def trace_span_decorator(operation_name: Optional[str] = None):
    """Decorator for automatic distributed tracing spans."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with global_tracing.span(op_name, function_name=func.__name__):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def audit_logger_decorator(event_type: str, sensitive_params: Optional[List[str]] = None):
    """Decorator for automatic audit logging."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f"audit.{func.__module__}")
            audit_logger = SecurityAuditLogger(logger)
            
            # Prepare audit data (filter sensitive parameters)
            audit_data = {
                'function_name': func.__name__,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            }
            
            if sensitive_params:
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                 if k not in sensitive_params}
                audit_data['filtered_kwargs'] = filtered_kwargs
            
            try:
                result = func(*args, **kwargs)
                
                logger.audit(
                    f"Function execution completed: {event_type}",
                    **audit_data,
                    success=True
                )
                
                return result
                
            except Exception as e:
                logger.audit(
                    f"Function execution failed: {event_type}",
                    **audit_data,
                    success=False,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator

class LogAnalyzer:
    """Analyze log patterns and generate insights."""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.error_patterns = {}
        self.performance_trends = {}
    
    def analyze_error_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns in logs."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        error_counts = {}
        error_timestamps = {}
        
        try:
            with open(self.log_file_path, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        
                        if (log_entry.get('level') in ['ERROR', 'CRITICAL'] and
                            log_entry.get('timestamp', 0) > cutoff_time):
                            
                            error_msg = log_entry.get('message', 'Unknown error')
                            error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
                            
                            if error_msg not in error_timestamps:
                                error_timestamps[error_msg] = []
                            error_timestamps[error_msg].append(log_entry.get('timestamp'))
                            
                    except json.JSONDecodeError:
                        continue
        
        except FileNotFoundError:
            return {'error': 'Log file not found'}
        
        # Find patterns and trends
        analysis = {
            'total_errors': sum(error_counts.values()),
            'unique_errors': len(error_counts),
            'most_frequent_errors': sorted(error_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:10],
            'error_rate_trend': self._calculate_error_rate_trend(error_timestamps),
            'analysis_window_hours': time_window_hours
        }
        
        return analysis
    
    def _calculate_error_rate_trend(self, error_timestamps: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate error rate trends."""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        recent_errors = 0
        older_errors = 0
        
        for error_type, timestamps in error_timestamps.items():
            for timestamp in timestamps:
                if timestamp > hour_ago:
                    recent_errors += 1
                else:
                    older_errors += 1
        
        if older_errors == 0:
            trend = float('inf') if recent_errors > 0 else 0.0
        else:
            trend = recent_errors / older_errors
        
        return {
            'recent_errors_per_hour': recent_errors,
            'previous_errors_per_hour': older_errors,
            'trend_ratio': trend
        }
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate system health report from logs."""
        error_analysis = self.analyze_error_patterns()
        
        # Determine system health status
        error_rate = error_analysis.get('total_errors', 0)
        
        if error_rate == 0:
            health_status = "excellent"
        elif error_rate < 10:
            health_status = "good"
        elif error_rate < 50:
            health_status = "warning"
        else:
            health_status = "critical"
        
        return {
            'health_status': health_status,
            'error_analysis': error_analysis,
            'recommendations': self._generate_recommendations(error_analysis),
            'report_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _generate_recommendations(self, error_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        
        total_errors = error_analysis.get('total_errors', 0)
        
        if total_errors > 100:
            recommendations.append("High error rate detected. Review system stability.")
        
        most_frequent = error_analysis.get('most_frequent_errors', [])
        if most_frequent and most_frequent[0][1] > 10:
            recommendations.append(f"Frequent error pattern: '{most_frequent[0][0]}'. "
                                 "Consider targeted fixes.")
        
        trend_ratio = error_analysis.get('error_rate_trend', {}).get('trend_ratio', 0)
        if trend_ratio > 2.0:
            recommendations.append("Error rate increasing. Monitor system closely.")
        
        return recommendations

# Custom exceptions for logging system
class LoggingError(Exception):
    """Base exception for logging system errors."""
    pass

class LogConfigurationError(LoggingError):
    """Exception for logging configuration errors."""
    pass

class LogHandlerError(LoggingError):
    """Exception for log handler errors."""
    pass

# Utility functions for common logging scenarios
def log_ai_conversation(logger: StructuredLogger, user_input: str, 
                       ai_response: str, conversation_id: str,
                       processing_time: float):
    """Log AI conversation with comprehensive metadata."""
    metrics = PerformanceMetrics(execution_time=processing_time)
    
    with logger.context(conversation_id=conversation_id, operation="ai_conversation"):
        logger.info(
            "AI conversation completed",
            category=LogCategory.AI_CONVERSATION,
            performance_metrics=metrics,
            user_input_length=len(user_input),
            ai_response_length=len(ai_response),
            processing_time=processing_time
        )

def log_sensor_data_processing(logger: StructuredLogger, sensor_type: str,
                             data_points: int, quality_score: float,
                             anomalies_detected: int = 0):
    """Log sensor data processing with quality metrics."""
    logger.info(
        f"Processed {data_points} data points from {sensor_type} sensor",
        category=LogCategory.SENSOR_DATA,
        sensor_type=sensor_type,
        data_points=data_points,
        quality_score=quality_score,
        anomalies_detected=anomalies_detected
    )

def log_holographic_display_event(logger: StructuredLogger, event_type: str,
                                 hologram_id: str, viewer_distance: float,
                                 display_quality: str):
    """Log holographic display events."""
    logger.info(
        f"Holographic display event: {event_type}",
        category=LogCategory.HOLOGRAPHIC,
        hologram_id=hologram_id,
        viewer_distance=viewer_distance,
        display_quality=display_quality,
        event_type=event_type
    )

def configure_development_logging():
    """Configure logging for development environment."""
    config = {
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'formatter': 'human'
            }
        },
        'loggers': {
            'ai_wristwatch': {
                'level': 'DEBUG',
                'handlers': ['console']
            }
        }
    }
    
    global_logging_config.config.update(config)
    global_logging_config.setup_logging()

def configure_production_logging(log_directory: str = "/var/log/ai_wristwatch"):
    """Configure logging for production environment."""
    config = {
        'handlers': {
            'console': {
                'level': 'WARNING'
            },
            'file': {
                'filename': f"{log_directory}/application.log",
                'level': 'INFO'
            },
            'security': {
                'filename': f"{log_directory}/security.log",
                'level': 'SECURITY'
            },
            'error': {
                'class': 'logging_utils.AdvancedRotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json',
                'filename': f"{log_directory}/errors.log",
                'maxBytes': 50 * 1024 * 1024,
                'backupCount': 30
            }
        },
        'loggers': {
            'ai_wristwatch': {
                'level': 'INFO',
                'handlers': ['console', 'file', 'error']
            }
        }
    }
    
    global_logging_config.config.update(config)
    global_logging_config.setup_logging()

if __name__ == "__main__":
    # Example usage and testing
    print("AI Holographic Wristwatch - Logging Utilities Module")
    print("Testing logging infrastructure...")
    
    # Setup basic logging
    setup_logging_environment()
    
    # Create test logger
    test_context = LogContext(
        user_id="test_user_123",
        device_id="test_device_456",
        component="logging_test"
    )
    
    test_logger = get_logger("test_module", test_context)
    
    # Test different log levels
    test_logger.info("Testing info level logging")
    test_logger.warning("Testing warning level logging")
    test_logger.debug("Testing debug level logging")
    
    # Test performance logging
    perf_logger = PerformanceLogger(test_logger)
    
    with perf_logger.time_operation("test_operation"):
        time.sleep(0.1)  # Simulate work
    
    # Test security logging
    security_logger = SecurityAuditLogger(test_logger)
    security_logger.log_authentication_event(
        "test_user", "test_device", True, "biometric"
    )
    
    # Test distributed tracing
    with global_tracing.span("test_span"):
        test_logger.debug("Operation within traced span")
    
    # Test AI conversation logging
    log_ai_conversation(
        test_logger, 
        "Hello AI", 
        "Hello! How can I help you?", 
        "conv_123", 
        0.05
    )
    
    print("Logging utilities module initialized successfully.")