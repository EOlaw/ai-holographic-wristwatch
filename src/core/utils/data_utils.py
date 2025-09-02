"""
Robust Data Manipulation Utilities for AI Holographic Wristwatch System

This module provides comprehensive data handling capabilities including serialization,
validation, transformation pipelines, caching, and streaming data management.
"""

import json
import pickle
import gzip
import hashlib
import base64
import struct
from typing import Tuple, Any, Dict, List, Optional, Union, Callable, Iterator, BinaryIO
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
import asyncio
import threading
import time
from collections import OrderedDict, deque
from abc import ABC, abstractmethod
import weakref
import numpy as np
import scipy.stats
import copy

class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    BINARY = "binary"
    COMPRESSED_JSON = "compressed_json"
    COMPRESSED_PICKLE = "compressed_pickle"

class CompressionType(Enum):
    """Supported compression algorithms."""
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"

@dataclass
class DataValidationResult:
    """Result of data validation operations."""
    is_valid: bool
    error_messages: List[str]
    warnings: List[str]
    validated_data: Optional[Any] = None

@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    max_size: int = 0

class DataTransformer(ABC):
    """Abstract base class for data transformation operations."""
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform input data."""
        pass
    
    @abstractmethod
    def inverse_transform(self, data: Any) -> Any:
        """Apply inverse transformation."""
        pass

class NormalizationTransformer(DataTransformer):
    """Data normalization transformer."""
    
    def __init__(self, method: str = 'minmax', feature_range: Tuple[float, float] = (0, 1)):
        self.method = method
        self.feature_range = feature_range
        self.min_vals = None
        self.max_vals = None
        self.mean_vals = None
        self.std_vals = None
        self.is_fitted = False
    
    def fit(self, data: np.ndarray):
        """Fit transformer parameters to data."""
        if self.method == 'minmax':
            self.min_vals = np.min(data, axis=0)
            self.max_vals = np.max(data, axis=0)
        elif self.method == 'standard':
            self.mean_vals = np.mean(data, axis=0)
            self.std_vals = np.std(data, axis=0)
        
        self.is_fitted = True
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization transformation."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transformation")
        
        if self.method == 'minmax':
            range_vals = self.max_vals - self.min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            normalized = (data - self.min_vals) / range_vals
            return normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        elif self.method == 'standard':
            std_vals = self.std_vals.copy()
            std_vals[std_vals == 0] = 1  # Avoid division by zero
            return (data - self.mean_vals) / std_vals
        
        return data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply inverse normalization."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before inverse transformation")
        
        if self.method == 'minmax':
            denormalized = ((data - self.feature_range[0]) / 
                          (self.feature_range[1] - self.feature_range[0]))
            return denormalized * (self.max_vals - self.min_vals) + self.min_vals
        
        elif self.method == 'standard':
            return data * self.std_vals + self.mean_vals
        
        return data

class DataSerializer:
    """Advanced data serialization with multiple format support."""
    
    @staticmethod
    def serialize(data: Any, format_type: SerializationFormat, 
                 compression: Optional[CompressionType] = None) -> bytes:
        """Serialize data to bytes using specified format."""
        try:
            # Convert dataclass objects to dictionaries
            if is_dataclass(data):
                data = asdict(data)
            
            if format_type == SerializationFormat.JSON:
                serialized = json.dumps(data, default=DataSerializer._json_serializer).encode('utf-8')
            elif format_type == SerializationFormat.PICKLE:
                serialized = pickle.dumps(data)
            elif format_type == SerializationFormat.BINARY:
                serialized = DataSerializer._serialize_binary(data)
            elif format_type == SerializationFormat.COMPRESSED_JSON:
                json_data = json.dumps(data, default=DataSerializer._json_serializer).encode('utf-8')
                serialized = gzip.compress(json_data)
            elif format_type == SerializationFormat.COMPRESSED_PICKLE:
                pickle_data = pickle.dumps(data)
                serialized = gzip.compress(pickle_data)
            else:
                raise ValueError(f"Unsupported serialization format: {format_type}")
            
            # Apply additional compression if specified
            if compression and format_type not in [SerializationFormat.COMPRESSED_JSON, 
                                                  SerializationFormat.COMPRESSED_PICKLE]:
                serialized = DataSerializer._apply_compression(serialized, compression)
            
            return serialized
            
        except Exception as e:
            raise DataSerializationError(f"Serialization failed: {str(e)}")
    
    @staticmethod
    def deserialize(data: bytes, format_type: SerializationFormat,
                   compression: Optional[CompressionType] = None) -> Any:
        """Deserialize bytes to Python object using specified format."""
        try:
            # Apply decompression if specified
            if compression and format_type not in [SerializationFormat.COMPRESSED_JSON,
                                                  SerializationFormat.COMPRESSED_PICKLE]:
                data = DataSerializer._apply_decompression(data, compression)
            
            if format_type == SerializationFormat.JSON:
                return json.loads(data.decode('utf-8'))
            elif format_type == SerializationFormat.PICKLE:
                return pickle.loads(data)
            elif format_type == SerializationFormat.BINARY:
                return DataSerializer._deserialize_binary(data)
            elif format_type == SerializationFormat.COMPRESSED_JSON:
                decompressed = gzip.decompress(data)
                return json.loads(decompressed.decode('utf-8'))
            elif format_type == SerializationFormat.COMPRESSED_PICKLE:
                decompressed = gzip.decompress(data)
                return pickle.loads(decompressed)
            else:
                raise ValueError(f"Unsupported deserialization format: {format_type}")
                
        except Exception as e:
            raise DataSerializationError(f"Deserialization failed: {str(e)}")
    
    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, np.ndarray):
            return {'__numpy_array__': True, 'data': obj.tolist(), 'dtype': str(obj.dtype)}
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    @staticmethod
    def _serialize_binary(data: Any) -> bytes:
        """Custom binary serialization for performance-critical data."""
        if isinstance(data, (int, float)):
            return struct.pack('d', float(data))
        elif isinstance(data, str):
            encoded = data.encode('utf-8')
            return struct.pack('I', len(encoded)) + encoded
        elif isinstance(data, np.ndarray):
            header = struct.pack('II', *data.shape)
            return header + data.tobytes()
        else:
            # Fallback to pickle for complex objects
            return pickle.dumps(data)
    
    @staticmethod
    def _deserialize_binary(data: bytes) -> Any:
        """Custom binary deserialization."""
        try:
            # Try as float first
            if len(data) == 8:
                return struct.unpack('d', data)[0]
            # Try as string
            elif len(data) > 4:
                str_len = struct.unpack('I', data[:4])[0]
                if len(data) == 4 + str_len:
                    return data[4:].decode('utf-8')
            # Fallback to pickle
            return pickle.loads(data)
        except:
            return pickle.loads(data)
    
    @staticmethod
    def _apply_compression(data: bytes, compression: CompressionType) -> bytes:
        """Apply compression to data."""
        if compression == CompressionType.GZIP:
            return gzip.compress(data)
        else:
            # Other compression types would be implemented here
            return data
    
    @staticmethod
    def _apply_decompression(data: bytes, compression: CompressionType) -> bytes:
        """Apply decompression to data."""
        if compression == CompressionType.GZIP:
            return gzip.decompress(data)
        else:
            # Other decompression types would be implemented here
            return data

class DataIntegrityChecker:
    """Data integrity verification and validation."""
    
    @staticmethod
    def calculate_checksum(data: Union[bytes, str], algorithm: str = 'sha256') -> str:
        """Calculate checksum for data integrity verification."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        hash_algorithms = {
            'md5': hashlib.md5,
            'sha1': hashlib.sha1,
            'sha256': hashlib.sha256,
            'sha512': hashlib.sha512
        }
        
        if algorithm not in hash_algorithms:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_obj = hash_algorithms[algorithm]()
        hash_obj.update(data)
        return hash_obj.hexdigest()
    
    @staticmethod
    def verify_checksum(data: Union[bytes, str], expected_checksum: str, 
                       algorithm: str = 'sha256') -> bool:
        """Verify data integrity using checksum."""
        calculated_checksum = DataIntegrityChecker.calculate_checksum(data, algorithm)
        return calculated_checksum == expected_checksum
    
    @staticmethod
    def add_integrity_wrapper(data: Any, format_type: SerializationFormat) -> dict:
        """Add integrity metadata to serialized data."""
        serialized_data = DataSerializer.serialize(data, format_type)
        checksum = DataIntegrityChecker.calculate_checksum(serialized_data)
        
        return {
            'data': base64.b64encode(serialized_data).decode('utf-8'),
            'checksum': checksum,
            'format': format_type.value,
            'timestamp': time.time(),
            'version': '1.0'
        }
    
    @staticmethod
    def extract_and_verify(wrapped_data: dict, format_type: SerializationFormat) -> Any:
        """Extract and verify data from integrity wrapper."""
        data_bytes = base64.b64decode(wrapped_data['data'])
        expected_checksum = wrapped_data['checksum']
        
        if not DataIntegrityChecker.verify_checksum(data_bytes, expected_checksum):
            raise DataIntegrityError("Data integrity check failed")
        
        return DataSerializer.deserialize(data_bytes, format_type)

class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.stats = CacheStatistics(max_size=max_size)
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.stats.hits += 1
                return value
            else:
                self.stats.misses += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self._lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used item
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
                self.stats.evictions += 1
            
            self.cache[key] = value
            self.stats.total_size = len(self.cache)
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self.cache.clear()
            self.stats = CacheStatistics(max_size=self.max_size)
    
    def get_statistics(self) -> CacheStatistics:
        """Get cache performance statistics."""
        with self._lock:
            self.stats.total_size = len(self.cache)
            return self.stats

class DataPipeline:
    """Data transformation pipeline for processing workflows."""
    
    def __init__(self, name: str = "default_pipeline"):
        self.name = name
        self.transformers = []
        self.performance_metrics = {}
    
    def add_transformer(self, transformer: Callable[[Any], Any], 
                       name: Optional[str] = None):
        """Add transformation step to pipeline."""
        step_name = name or f"step_{len(self.transformers)}"
        self.transformers.append((step_name, transformer))
    
    def process(self, data: Any, monitor_performance: bool = True) -> Any:
        """Process data through transformation pipeline."""
        result = data
        
        for step_name, transformer in self.transformers:
            start_time = time.perf_counter()
            
            try:
                result = transformer(result)
            except Exception as e:
                raise DataTransformationError(f"Pipeline step '{step_name}' failed: {str(e)}")
            
            if monitor_performance:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                if step_name not in self.performance_metrics:
                    self.performance_metrics[step_name] = []
                self.performance_metrics[step_name].append(execution_time)
        
        return result
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for pipeline steps."""
        summary = {}
        
        for step_name, times in self.performance_metrics.items():
            summary[step_name] = {
                'total_executions': len(times),
                'total_time': sum(times),
                'average_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times)
            }
        
        return summary

class StreamingDataHandler:
    """Handle streaming data with buffering and real-time processing."""
    
    def __init__(self, buffer_size: int = 10000, 
                 processing_interval: float = 0.1):
        self.buffer_size = buffer_size
        self.processing_interval = processing_interval
        self.buffer = deque(maxlen=buffer_size)
        self.processors = []
        self.is_active = False
        self._processing_thread = None
        self._lock = threading.Lock()
        self.statistics = {
            'total_items_processed': 0,
            'buffer_overflows': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0
        }
    
    def add_processor(self, processor: Callable[[Any], None], 
                     name: Optional[str] = None):
        """Add data processor to streaming handler."""
        processor_name = name or f"processor_{len(self.processors)}"
        self.processors.append((processor_name, processor))
    
    def start_streaming(self):
        """Start streaming data processing."""
        if self.is_active:
            return
        
        self.is_active = True
        self._processing_thread = threading.Thread(target=self._process_stream)
        self._processing_thread.start()
    
    def stop_streaming(self):
        """Stop streaming data processing."""
        self.is_active = False
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join()
    
    def add_data(self, data: Any):
        """Add data to streaming buffer."""
        with self._lock:
            if len(self.buffer) >= self.buffer_size:
                self.statistics['buffer_overflows'] += 1
            self.buffer.append(data)
    
    def _process_stream(self):
        """Internal streaming data processing loop."""
        while self.is_active:
            try:
                if self.buffer:
                    with self._lock:
                        if self.buffer:
                            data_batch = list(self.buffer)
                            self.buffer.clear()
                        else:
                            data_batch = []
                    
                    if data_batch:
                        start_time = time.perf_counter()
                        
                        for processor_name, processor in self.processors:
                            try:
                                for item in data_batch:
                                    processor(item)
                            except Exception as e:
                                self.statistics['processing_errors'] += 1
                                print(f"Processing error in {processor_name}: {str(e)}")
                        
                        end_time = time.perf_counter()
                        processing_time = end_time - start_time
                        
                        # Update statistics
                        self.statistics['total_items_processed'] += len(data_batch)
                        current_avg = self.statistics['average_processing_time']
                        total_processed = self.statistics['total_items_processed']
                        self.statistics['average_processing_time'] = (
                            (current_avg * (total_processed - len(data_batch)) + 
                             processing_time) / total_processed
                        )
                
                time.sleep(self.processing_interval)
                
            except Exception as e:
                self.statistics['processing_errors'] += 1
                print(f"Streaming processing error: {str(e)}")

class DataConverter:
    """Utilities for converting between different data formats."""
    
    @staticmethod
    def numpy_to_list(array: np.ndarray) -> List[Any]:
        """Convert numpy array to Python list."""
        return array.tolist()
    
    @staticmethod
    def list_to_numpy(data: List[Any], dtype: Optional[str] = None) -> np.ndarray:
        """Convert Python list to numpy array."""
        return np.array(data, dtype=dtype)
    
    @staticmethod
    def dict_to_object(data: Dict[str, Any], target_class: type) -> Any:
        """Convert dictionary to object instance."""
        if is_dataclass(target_class):
            return target_class(**data)
        else:
            obj = target_class()
            for key, value in data.items():
                setattr(obj, key, value)
            return obj
    
    @staticmethod
    def object_to_dict(obj: Any) -> Dict[str, Any]:
        """Convert object to dictionary."""
        if is_dataclass(obj):
            return asdict(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise ValueError(f"Cannot convert {type(obj)} to dictionary")
    
    @staticmethod
    def flatten_nested_dict(nested_dict: Dict[str, Any], 
                          separator: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary structure."""
        def _flatten(obj: Any, parent_key: str = '') -> Dict[str, Any]:
            items = []
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{parent_key}{separator}{key}" if parent_key else key
                    items.extend(_flatten(value, new_key).items())
            else:
                return {parent_key: obj}
            
            return dict(items)
        
        return _flatten(nested_dict)
    
    @staticmethod
    def unflatten_dict(flat_dict: Dict[str, Any], 
                      separator: str = '.') -> Dict[str, Any]:
        """Unflatten flattened dictionary structure."""
        result = {}
        
        for key, value in flat_dict.items():
            parts = key.split(separator)
            current = result
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
        
        return result

class DataValidator:
    """Comprehensive data validation framework."""
    
    def __init__(self):
        self.validation_rules = {}
        self.custom_validators = {}
    
    def add_validation_rule(self, field_name: str, validator: Callable[[Any], bool],
                          error_message: str):
        """Add validation rule for a field."""
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        
        self.validation_rules[field_name].append((validator, error_message))
    
    def add_custom_validator(self, name: str, validator: Callable[[Any], DataValidationResult]):
        """Add custom validator function."""
        self.custom_validators[name] = validator
    
    def validate_data(self, data: Dict[str, Any]) -> DataValidationResult:
        """Validate data using configured rules."""
        errors = []
        warnings = []
        validated_data = data.copy()
        
        # Apply field-specific validation rules
        for field_name, rules in self.validation_rules.items():
            if field_name in data:
                field_value = data[field_name]
                
                for validator, error_message in rules:
                    try:
                        if not validator(field_value):
                            errors.append(f"{field_name}: {error_message}")
                    except Exception as e:
                        errors.append(f"{field_name}: Validation error - {str(e)}")
        
        # Apply custom validators
        for validator_name, validator in self.custom_validators.items():
            try:
                result = validator(data)
                if not result.is_valid:
                    errors.extend(result.error_messages)
                    warnings.extend(result.warnings)
            except Exception as e:
                errors.append(f"Custom validator '{validator_name}' failed: {str(e)}")
        
        is_valid = len(errors) == 0
        
        return DataValidationResult(
            is_valid=is_valid,
            error_messages=errors,
            warnings=warnings,
            validated_data=validated_data if is_valid else None
        )
    
    @staticmethod
    def validate_numeric_range(min_val: float, max_val: float) -> Callable[[Any], bool]:
        """Create validator for numeric range."""
        def validator(value: Any) -> bool:
            try:
                numeric_value = float(value)
                return min_val <= numeric_value <= max_val
            except (ValueError, TypeError):
                return False
        return validator
    
    @staticmethod
    def validate_string_pattern(pattern: str) -> Callable[[Any], bool]:
        """Create validator for string pattern matching."""
        import re
        compiled_pattern = re.compile(pattern)
        
        def validator(value: Any) -> bool:
            if not isinstance(value, str):
                return False
            return compiled_pattern.match(value) is not None
        
        return validator
    
    @staticmethod
    def validate_array_dimensions(expected_shape: Tuple[int, ...]) -> Callable[[Any], bool]:
        """Create validator for array dimensions."""
        def validator(value: Any) -> bool:
            if not isinstance(value, np.ndarray):
                return False
            return value.shape == expected_shape
        return validator

class DataCompression:
    """Advanced data compression utilities."""
    
    @staticmethod
    def compress_sensor_data(sensor_readings: np.ndarray, 
                           compression_ratio: float = 0.1) -> np.ndarray:
        """Compress sensor data while preserving important features."""
        # Use discrete cosine transform for compression
        from scipy.fft import dct, idct
        
        # Apply DCT
        dct_coefficients = dct(sensor_readings, type=2, norm='ortho')
        
        # Keep only the most significant coefficients
        num_coefficients = int(len(dct_coefficients) * compression_ratio)
        compressed_coefficients = np.zeros_like(dct_coefficients)
        
        # Keep coefficients with largest absolute values
        indices = np.argsort(np.abs(dct_coefficients))[-num_coefficients:]
        compressed_coefficients[indices] = dct_coefficients[indices]
        
        return compressed_coefficients
    
    @staticmethod
    def decompress_sensor_data(compressed_coefficients: np.ndarray) -> np.ndarray:
        """Decompress sensor data from DCT coefficients."""
        from scipy.fft import idct
        return idct(compressed_coefficients, type=2, norm='ortho')
    
    @staticmethod
    def adaptive_compression(data: np.ndarray, target_size_ratio: float = 0.5) -> np.ndarray:
        """Apply adaptive compression based on data characteristics."""
        # Analyze data characteristics
        data_variance = np.var(data)
        data_entropy = DataCompression._calculate_entropy(data)
        
        # Choose compression strategy based on data characteristics
        if data_entropy < 2.0:  # Low entropy data
            return DataCompression._run_length_encoding(data)
        elif data_variance > 1.0:  # High variance data
            return DataCompression.compress_sensor_data(data, target_size_ratio)
        else:
            # Use simple quantization
            return DataCompression._quantize_data(data, target_size_ratio)
    
    @staticmethod
    def _calculate_entropy(data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        # Discretize continuous data
        hist, _ = np.histogram(data, bins=50)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Calculate probabilities
        probabilities = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    @staticmethod
    def _run_length_encoding(data: np.ndarray) -> np.ndarray:
        """Simple run-length encoding for repeated values."""
        if len(data) == 0:
            return data
        
        encoded = []
        current_value = data[0]
        count = 1
        
        for i in range(1, len(data)):
            if np.isclose(data[i], current_value, rtol=1e-5):
                count += 1
            else:
                encoded.extend([current_value, count])
                current_value = data[i]
                count = 1
        
        encoded.extend([current_value, count])
        return np.array(encoded)
    
    @staticmethod
    def _quantize_data(data: np.ndarray, reduction_factor: float) -> np.ndarray:
        """Quantize data to reduce precision and size."""
        data_range = np.max(data) - np.min(data)
        num_levels = int(256 * reduction_factor)
        
        quantization_step = data_range / num_levels
        quantized = np.round((data - np.min(data)) / quantization_step) * quantization_step
        quantized += np.min(data)
        
        return quantized

class DataBatcher:
    """Batch data processing utilities."""
    
    @staticmethod
    def create_batches(data: List[Any], batch_size: int, 
                      overlap: int = 0) -> Iterator[List[Any]]:
        """Create batches from data with optional overlap."""
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if overlap >= batch_size:
            raise ValueError("Overlap must be less than batch size")
        
        step_size = batch_size - overlap
        
        for i in range(0, len(data), step_size):
            if i + batch_size <= len(data):
                yield data[i:i + batch_size]
            elif i < len(data):  # Handle remaining data
                yield data[i:]
    
    @staticmethod
    def process_batches_parallel(data: List[Any], processor: Callable[[List[Any]], Any],
                               batch_size: int, max_workers: Optional[int] = None) -> List[Any]:
        """Process batches in parallel using ThreadPoolExecutor."""
        import concurrent.futures
        
        batches = list(DataBatcher.create_batches(data, batch_size))
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(processor, batch): batch 
                             for batch in batches}
            
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    raise DataProcessingError(f"Batch processing failed: {str(e)}")
        
        return results
    
    @staticmethod
    async def process_batches_async(data: List[Any], 
                                  processor: Callable[[List[Any]], Any],
                                  batch_size: int, 
                                  max_concurrent: int = 10) -> List[Any]:
        """Process batches asynchronously."""
        batches = list(DataBatcher.create_batches(data, batch_size))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch: List[Any]) -> Any:
            async with semaphore:
                return processor(batch)
        
        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        return results

class SecureDataHandler:
    """Secure data handling with encryption and access control."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self._access_log = []
    
    def _generate_key(self) -> bytes:
        """Generate encryption key for data security."""
        import os
        return os.urandom(32)  # 256-bit key
    
    def encrypt_data(self, data: Any, format_type: SerializationFormat) -> bytes:
        """Encrypt data using AES encryption."""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Create Fernet instance with key
            key = base64.urlsafe_b64encode(self.encryption_key)
            cipher_suite = Fernet(key)
            
            # Serialize data first
            serialized_data = DataSerializer.serialize(data, format_type)
            
            # Encrypt serialized data
            encrypted_data = cipher_suite.encrypt(serialized_data)
            
            self._log_access("encrypt", len(serialized_data))
            return encrypted_data
            
        except Exception as e:
            raise DataSecurityError(f"Encryption failed: {str(e)}")
    
    def decrypt_data(self, encrypted_data: bytes, 
                    format_type: SerializationFormat) -> Any:
        """Decrypt and deserialize data."""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Create Fernet instance with key
            key = base64.urlsafe_b64encode(self.encryption_key)
            cipher_suite = Fernet(key)
            
            # Decrypt data
            decrypted_data = cipher_suite.decrypt(encrypted_data)
            
            # Deserialize decrypted data
            result = DataSerializer.deserialize(decrypted_data, format_type)
            
            self._log_access("decrypt", len(decrypted_data))
            return result
            
        except Exception as e:
            raise DataSecurityError(f"Decryption failed: {str(e)}")
    
    def _log_access(self, operation: str, data_size: int):
        """Log data access for security auditing."""
        self._access_log.append({
            'timestamp': time.time(),
            'operation': operation,
            'data_size': data_size,
            'thread_id': threading.get_ident()
        })
        
        # Keep only recent access logs (last 1000 entries)
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get security access log."""
        return self._access_log.copy()

# Custom exceptions for data operations
class DataOperationError(Exception):
    """Base exception for data operation errors."""
    pass

class DataSerializationError(DataOperationError):
    """Exception for data serialization/deserialization errors."""
    pass

class DataIntegrityError(DataOperationError):
    """Exception for data integrity check failures."""
    pass

class DataTransformationError(DataOperationError):
    """Exception for data transformation errors."""
    pass

class DataProcessingError(DataOperationError):
    """Exception for data processing errors."""
    pass

class DataSecurityError(DataOperationError):
    """Exception for data security operation errors."""
    pass

# Utility functions for common data operations
def deep_copy_data(data: Any) -> Any:
    """Create deep copy of data structure."""
    return copy.deepcopy(data)

def merge_dictionaries(*dicts: Dict[str, Any], deep_merge: bool = True) -> Dict[str, Any]:
    """Merge multiple dictionaries with optional deep merging."""
    result = {}
    
    for dictionary in dicts:
        if deep_merge:
            for key, value in dictionary.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dictionaries(result[key], value, deep_merge=True)
                else:
                    result[key] = deep_copy_data(value)
        else:
            result.update(dictionary)
    
    return result

def filter_data_by_criteria(data: List[Dict[str, Any]], 
                          criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter data based on specified criteria."""
    def matches_criteria(item: Dict[str, Any]) -> bool:
        for key, expected_value in criteria.items():
            if key not in item:
                return False
            
            if callable(expected_value):
                if not expected_value(item[key]):
                    return False
            else:
                if item[key] != expected_value:
                    return False
        
        return True
    
    return [item for item in data if matches_criteria(item)]

def sort_data_by_multiple_keys(data: List[Dict[str, Any]], 
                              sort_keys: List[Tuple[str, bool]]) -> List[Dict[str, Any]]:
    """Sort data by multiple keys with ascending/descending specification."""
    for key, reverse in reversed(sort_keys):
        data.sort(key=lambda x: x.get(key, 0), reverse=reverse)
    
    return data

def calculate_data_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive statistics for numerical data."""
    return {
        'count': len(data),
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'var': float(np.var(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q1': float(np.percentile(data, 25)),
        'q3': float(np.percentile(data, 75)),
        'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
        'skewness': float(scipy.stats.skew(data)),
        'kurtosis': float(scipy.stats.kurtosis(data))
    }

# Global data utilities
global_cache = LRUCache(max_size=10000)
global_data_validator = DataValidator()

# Initialize common validation rules
global_data_validator.add_validation_rule(
    'timestamp', 
    lambda x: isinstance(x, (int, float)) and x > 0,
    'Timestamp must be a positive number'
)

global_data_validator.add_validation_rule(
    'sensor_reading',
    DataValidator.validate_numeric_range(-1000, 1000),
    'Sensor reading must be between -1000 and 1000'
)

if __name__ == "__main__":
    # Example usage and testing
    print("AI Holographic Wristwatch - Data Utilities Module")
    print("Testing data manipulation utilities...")
    
    # Test serialization
    test_data = {"sensor_data": [1, 2, 3, 4, 5], "timestamp": time.time()}
    serialized = DataSerializer.serialize(test_data, SerializationFormat.JSON)
    deserialized = DataSerializer.deserialize(serialized, SerializationFormat.JSON)
    print(f"Serialization test successful: {len(serialized)} bytes")
    
    # Test caching
    global_cache.put("test_key", test_data)
    cached_data = global_cache.get("test_key")
    print(f"Caching test successful: {cached_data is not None}")
    
    # Test data validation
    validation_result = global_data_validator.validate_data({
        'timestamp': time.time(),
        'sensor_reading': 50.0
    })
    print(f"Validation test successful: {validation_result.is_valid}")
    
    print("Data utilities module initialized successfully.")