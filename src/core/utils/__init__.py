# src/core/utils/__init__.py
"""
Core Utilities Package for AI Holographic Wristwatch System

This package provides comprehensive utility functions and classes for the AI Holographic
Wristwatch system, including mathematical computations, data manipulation, input validation,
logging infrastructure, time management, and memory optimization.

The utilities are designed to work together to provide a robust foundation for the
entire system, with emphasis on performance, reliability, and maintainability.
"""

from typing import Any, Dict, List, Optional, Union, Callable

# ============================================================================
# Mathematical Utilities
# ============================================================================
from .math_utils import (
    # Core mathematical classes
    Vector3D,
    Quaternion,
    AdvancedLinearAlgebra,
    HolographicCalculations,
    AdvancedSignalProcessing,
    AdvancedOptimization,
    ProbabilityDistributions,
    AdvancedStatistics,
    GeometricCalculations,
    NumericalIntegration,
    NumericalDifferentiation,
    NumericalRootFinding,
    InterpolationMethods,
    AdvancedNumericalMethods,
    
    # Performance and profiling
    MathPerformanceProfiler,
    MathPerformanceStats,
    
    # Utility functions
    profile_math_operation,
    memoize_math_function,
    math_performance_context,
    create_rotation_matrix_from_vectors,
    solve_cubic_equation,
    run_math_utils_tests,
    
    # Global instances
    global_math_profiler,
    global_performance_stats,
    
    # Exceptions
    MathematicalError,
    ConvergenceError,
    NumericalInstabilityError,
    InvalidDomainError,
)

# ============================================================================
# Data Manipulation Utilities  
# ============================================================================
from .data_utils import (
    # Serialization and formats
    SerializationFormat,
    CompressionType,
    DataIntegrityLevel,
    CachePolicy,
    
    # Core data classes
    AdvancedDataSerializer,
    DataIntegrityManager,
    AdvancedCacheManager,
    DataPipelineProcessor,
    DataSchema,
    DataValidationResult,
    
    # Utility functions
    serialize_with_integrity,
    deserialize_with_verification,
    create_data_pipeline,
    validate_data_schema,
    run_comprehensive_data_tests,
    
    # Global instances
    global_serializer,
    global_integrity_manager,
    global_cache,
    
    # Exceptions
    DataOperationError,
    DataSerializationError,
    DataIntegrityError,
    DataTransformationError,
    DataValidationError,
    DataProcessingError,
    CacheError,
)

# ============================================================================
# Input Validation Utilities
# ============================================================================
from .validation_utils import (
    # Validation enums and types
    ValidationSeverity,
    ValidationType,
    
    # Core validation classes
    ValidationIssue,
    ValidationResult,
    BaseValidator,
    TypeValidator,
    RangeValidator,
    PatternValidator,
    SecurityValidator,
    CompositeValidator,
    CustomRuleEngine,
    DataSanitizer,
    
    # Pre-configured validators
    CommonValidators,
    
    # Utility functions
    validate_with_context,
    batch_validate,
    create_sensor_data_validator,
    sanitize_user_input_comprehensive,
    validate_ai_conversation_input,
    
    # Global instances
    global_validation_cache,
    global_validation_profiler,
    global_validation_aggregator,
    
    # Custom validation utilities
    SensorDataValidator,
    AIConversationValidator,
    HolographicDataValidator,
)

# ============================================================================
# Advanced Logging Infrastructure
# ============================================================================
from .logging_utils import (
    # Logging enums and levels
    LogLevel,
    LogCategory,
    
    # Core logging classes
    LogContext,
    PerformanceMetrics,
    SecurityEvent,
    StructuredLogEntry,
    StructuredLogger,
    LoggingConfiguration,
    PerformanceLogger,
    SecurityAuditLogger,
    LogAnalyzer,
    
    # Formatters and handlers
    JSONFormatter,
    ColoredConsoleFormatter,
    AdvancedRotatingFileHandler,
    StreamingLogHandler,
    
    # Utility functions
    log_ai_conversation,
    log_sensor_data_processing,
    log_holographic_display_event,
    configure_development_logging,
    configure_production_logging,
    
    # Decorators
    performance_logger_decorator,
    trace_span_decorator,
    audit_logger_decorator,
    
    # Global logging instances
    global_logging_config,
    global_tracing,
    
    # Exceptions
    LoggingError,
    LogConfigurationError,
    LogHandlerError,
)

# ============================================================================
# Time Management Utilities
# ============================================================================
from .time_utils import (
    # Time enums and types
    TimeUnit,
    RecurrencePattern,
    TimeZoneRegion,
    
    # Core time classes
    TimeInterval,
    ScheduledEvent,
    TimezoneManager,
    EventScheduler,
    DurationCalculator,
    TimingProfiler,
    TimeSync,
    TemporalCache,
    ChronoUtils,
    CronScheduler,
    BusinessHoursCalculator,
    SmartReminderSystem,
    
    # Time utility functions
    get_current_utc_time,
    get_current_local_time,
    parse_flexible_datetime,
    create_time_range,
    schedule_delayed_callback,
    benchmark_function_timing,
    create_smart_reminder_system,
    
    # Global time management instances
    global_timezone_manager,
    global_time_sync,
    global_timing_profiler,
    global_temporal_cache,
    
    # Decorators
    time_operation,
    cache_with_ttl,
    
    # Custom exceptions
    TimeUtilsError,
    TimeSyncError,
    SchedulingError,
    TimezoneError,
    CalendarIntegrationError,
)

# ============================================================================
# Memory Management Utilities
# ============================================================================
from .memory_utils import (
    # Memory enums and types
    MemoryUnit,
    CachePolicy as MemoryCachePolicy,
    GCStrategy,
    
    # Core memory classes
    MemoryStats,
    CacheEntry,
    MemoryPool,
    AdvancedCache,
    SmartCache,
    MemoryProfiler,
    GarbageCollectionOptimizer,
    MemoryLeakDetector,
    ResourceMonitor,
    MemoryOptimizer,
    TypedObjectPool,
    MemoryMappedBuffer,
    
    # Memory utility functions
    get_memory_stats,
    format_memory_size,
    measure_memory_usage,
    find_memory_leaks,
    clear_all_caches,
    
    # Advanced memory functions
    create_ai_model_cache,
    create_sensor_data_pool,
    create_holographic_buffer,
    profile_memory_usage,
    emergency_memory_cleanup,
    get_global_memory_status,
    configure_global_memory_management,
    
    # Context managers
    memory_monitoring_context,
    temporary_memory_limit,
    garbage_collection_disabled,
    
    # Global memory management instances
    global_memory_optimizer,
    global_leak_detector,
    global_resource_monitor,
    
    # Configuration
    MEMORY_MANAGEMENT_CONFIG,
)

# ============================================================================
# Package-level Utility Functions
# ============================================================================

def initialize_core_utils(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize all core utility systems with configuration.
    
    This function sets up the logging system, memory management, time synchronization,
    and mathematical profiling based on the provided configuration parameters.
    
    Args:
        config: Optional configuration dictionary containing system-specific settings
                for each utility module
        
    Returns:
        Dictionary containing initialization status and system information for each
        initialized component
    """
    config = config or {}
    initialization_status = {}
    
    try:
        # Initialize logging system first
        logging_config = config.get('logging', {})
        global_logging_config.config.update(logging_config)
        global_logging_config.setup_logging()
        initialization_status['logging'] = 'initialized'
        
        # Get system logger for subsequent operations
        system_logger = global_logging_config.create_structured_logger('system')
        
        # Initialize memory management
        memory_config = config.get('memory', {})
        configure_global_memory_management(memory_config)
        global_resource_monitor.start_monitoring()
        initialization_status['memory'] = 'initialized'
        
        # Initialize time synchronization
        time_config = config.get('time', {})
        if time_config.get('enable_sync', True):
            # Start time synchronization in background
            import asyncio
            try:
                asyncio.create_task(global_time_sync.sync_time_async())
            except RuntimeError:
                # Handle case where no event loop is running
                pass
        initialization_status['time_sync'] = 'initialized'
        
        # Initialize mathematical profiler
        math_config = config.get('math', {})
        global_math_profiler.clear_statistics()
        initialization_status['math_profiler'] = 'initialized'
        
        # Initialize validation system
        validation_config = config.get('validation', {})
        # Validation system is stateless, but we can note it's available
        initialization_status['validation'] = 'initialized'
        
        system_logger.info("Core utilities initialized successfully", 
                          extra={'initialization_status': initialization_status})
        
    except Exception as e:
        error_msg = f"Failed to initialize core utilities: {str(e)}"
        if 'system_logger' in locals():
            system_logger.error(error_msg, exc_info=True)
        else:
            print(error_msg)
        initialization_status['error'] = str(e)
    
    return initialization_status

def get_system_health_report() -> Dict[str, Any]:
    """
    Generate comprehensive system health report from all utility modules.
    
    This function collects health metrics from memory management, performance
    profiling, time synchronization, and logging systems to provide a complete
    view of system status.
    
    Returns:
        Dictionary containing comprehensive health metrics including memory usage,
        performance statistics, synchronization status, and logging activity
    """
    health_report = {
        'timestamp': get_current_utc_time().isoformat(),
        'system_status': 'healthy'
    }
    
    try:
        # Memory health metrics
        memory_status = get_global_memory_status()
        current_memory = memory_status['system_memory']
        health_report['memory'] = {
            'status': 'healthy' if current_memory.get('usage_percent', 0) < 85 else 'warning',
            'usage_percent': current_memory.get('usage_percent', 0),
            'available_gb': current_memory.get('available_gb', 0),
            'cache_efficiency': memory_status.get('cache_stats', {})
        }
        
        # Performance metrics
        math_performance = global_math_profiler.get_performance_report()
        health_report['performance'] = {
            'status': 'healthy',
            'math_operations': len(math_performance),
            'profiler_overhead': global_math_profiler._profiling_overhead if hasattr(global_math_profiler, '_profiling_overhead') else 0
        }
        
        # Time synchronization status
        health_report['time_sync'] = {
            'status': 'synced' if global_time_sync.is_sync_current() else 'out_of_sync',
            'last_sync': global_time_sync.last_sync_time,
            'sync_accuracy': global_time_sync.get_sync_accuracy()
        }
        
        # Logging system status
        health_report['logging'] = {
            'status': 'operational',
            'configured_loggers': len(global_logging_config.configured_loggers),
            'configuration_status': 'loaded'
        }
        
        # Validation system status
        health_report['validation'] = {
            'status': 'operational',
            'cache_enabled': hasattr(global_validation_cache, 'cache_enabled'),
            'profiler_active': hasattr(global_validation_profiler, 'profiling_enabled')
        }
        
        # Overall system status determination
        warning_conditions = [
            health_report['memory']['status'] == 'warning',
            health_report['time_sync']['status'] == 'out_of_sync'
        ]
        
        if any(warning_conditions):
            health_report['system_status'] = 'warning'
            
    except Exception as e:
        health_report['system_status'] = 'error'
        health_report['error'] = str(e)
        # Use basic logging if structured logger fails
        print(f"Failed to generate health report: {str(e)}")
    
    return health_report

def cleanup_core_utils() -> bool:
    """
    Cleanup and shutdown all core utility systems gracefully.
    
    This function performs orderly shutdown of all utility systems including
    stopping monitoring threads, clearing caches, saving state, and releasing
    resources.
    
    Returns:
        Boolean indicating whether all cleanup operations completed successfully
    """
    cleanup_success = True
    
    try:
        # Stop resource monitoring
        global_resource_monitor.stop_monitoring()
        
        # Cleanup memory management
        emergency_cleanup_result = emergency_memory_cleanup()
        if not cleanup_success:
            cleanup_success = cleanup_success and emergency_cleanup_result.get('success', False)
        
        # Clear all caches
        clear_all_caches()
        
        # Stop leak detection
        if hasattr(global_leak_detector, 'stop_monitoring'):
            global_leak_detector.stop_monitoring()
        
        # Cleanup mathematical profiler
        global_math_profiler.clear_statistics()
        
        # Clear temporal cache
        global_temporal_cache.clear()
        
        # Stop time synchronization
        if hasattr(global_time_sync, 'stop_sync'):
            global_time_sync.stop_sync()
        
        # Final log message before cleanup
        system_logger = global_logging_config.create_structured_logger('system')
        system_logger.info("Core utilities cleanup completed successfully")
        
        # Cleanup logging (do this last)
        for logger_name, logger in global_logging_config.configured_loggers.items():
            for handler in logger.logger.handlers[:]:
                try:
                    handler.close()
                    logger.logger.removeHandler(handler)
                except Exception as e:
                    print(f"Error closing handler for {logger_name}: {str(e)}")
                    cleanup_success = False
            
    except Exception as e:
        cleanup_success = False
        print(f"Error during core utilities cleanup: {str(e)}")
    
    return cleanup_success

# ============================================================================
# Convenience Functions for Common Operations
# ============================================================================

def get_logger(name: str, context: Optional[LogContext] = None) -> StructuredLogger:
    """
    Convenience function to get a configured structured logger.
    
    Args:
        name: Logger name, typically module or component name
        context: Optional logging context for structured metadata
        
    Returns:
        Configured StructuredLogger instance
    """
    return global_logging_config.create_structured_logger(name, context)

def validate_input(data: Any, validators: List[BaseValidator], 
                  context: Optional[str] = None) -> ValidationResult:
    """
    Convenience function for input validation with caching and profiling.
    
    Args:
        data: Data to validate
        validators: List of validator instances to apply
        context: Optional context for profiling and logging
        
    Returns:
        Comprehensive validation result with issues and metadata
    """
    return validate_with_context(data, validators, context)

def serialize_data(data: Any, format_type: SerializationFormat = SerializationFormat.JSON,
                  compression: Optional[CompressionType] = None) -> bytes:
    """
    Convenience function for data serialization.
    
    Args:
        data: Data to serialize
        format_type: Serialization format to use
        compression: Optional compression algorithm
        
    Returns:
        Serialized data as bytes
    """
    return global_serializer.serialize(data, format_type, compression)

def deserialize_data(data: bytes, format_type: SerializationFormat = SerializationFormat.JSON,
                    compression: Optional[CompressionType] = None) -> Any:
    """
    Convenience function for data deserialization.
    
    Args:
        data: Serialized data bytes
        format_type: Expected serialization format
        compression: Compression algorithm used
        
    Returns:
        Deserialized data object
    """
    return global_serializer.deserialize(data, format_type, compression)

def create_smart_cache(name: str, max_size_mb: int = 100) -> SmartCache:
    """
    Convenience function to create an optimized smart cache.
    
    Args:
        name: Cache identifier name
        max_size_mb: Maximum cache size in megabytes
        
    Returns:
        Configured SmartCache instance
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    return SmartCache(name=name, max_size_bytes=max_size_bytes)

def monitor_performance(operation_name: str):
    """
    Convenience decorator for performance monitoring.
    
    Args:
        operation_name: Name to identify the operation in performance reports
        
    Returns:
        Decorator function for automatic performance tracking
    """
    return performance_logger_decorator(operation_name)

# ============================================================================
# Module Metadata and Exports
# ============================================================================

__version__ = "1.0.0"
__author__ = "AI Holographic Wristwatch Development Team"
__description__ = "Core utilities package providing comprehensive foundational functionality"

# Define package-level exports for cleaner imports
__all__ = [
    # Mathematical utilities
    'Vector3D', 'Quaternion', 'AdvancedLinearAlgebra', 'HolographicCalculations',
    'MathPerformanceProfiler', 'profile_math_operation', 'memoize_math_function',
    'global_math_profiler', 'math_performance_context', 'create_rotation_matrix_from_vectors',
    'solve_cubic_equation', 'run_math_utils_tests',
    
    # Data manipulation utilities
    'SerializationFormat', 'CompressionType', 'DataIntegrityLevel', 'CachePolicy',
    'AdvancedDataSerializer', 'serialize_with_integrity', 'deserialize_with_verification',
    'create_data_pipeline', 'validate_data_schema', 'global_serializer',
    
    # Validation utilities
    'ValidationSeverity', 'ValidationType', 'ValidationResult', 'ValidationIssue',
    'BaseValidator', 'TypeValidator', 'RangeValidator', 'PatternValidator',
    'SecurityValidator', 'CompositeValidator', 'validate_with_context', 'batch_validate',
    'create_sensor_data_validator', 'CommonValidators',
    
    # Logging infrastructure
    'LogLevel', 'LogCategory', 'StructuredLogger', 'LoggingConfiguration',
    'PerformanceLogger', 'SecurityAuditLogger', 'log_ai_conversation',
    'log_sensor_data_processing', 'log_holographic_display_event',
    'global_logging_config', 'performance_logger_decorator',
    
    # Time management utilities
    'TimeUnit', 'RecurrencePattern', 'TimeInterval', 'ScheduledEvent',
    'EventScheduler', 'TimezoneManager', 'get_current_utc_time', 'parse_flexible_datetime',
    'create_time_range', 'global_timezone_manager', 'global_time_sync', 'time_operation',
    'cache_with_ttl', 'create_smart_reminder_system',
    
    # Memory management utilities
    'MemoryUnit', 'MemoryStats', 'SmartCache', 'MemoryProfiler', 'AdvancedCache',
    'get_memory_stats', 'format_memory_size', 'create_ai_model_cache',
    'emergency_memory_cleanup', 'get_global_memory_status', 'global_memory_optimizer',
    'memory_monitoring_context', 'profile_memory_usage',
    
    # Package-level functions
    'initialize_core_utils', 'get_system_health_report', 'cleanup_core_utils',
    
    # Convenience functions
    'get_logger', 'validate_input', 'serialize_data', 'deserialize_data',
    'create_smart_cache', 'monitor_performance'
]

# Package initialization message
print("AI Holographic Wristwatch Core Utilities Package Loaded")
print(f"Version: {__version__}")
print("Modules: math_utils, data_utils, validation_utils, logging_utils, time_utils, memory_utils")