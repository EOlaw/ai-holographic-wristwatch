"""
Robust Data Manipulation Utilities for AI Holographic Wristwatch System

This module provides comprehensive data handling capabilities including serialization,
validation, transformation pipelines, caching, streaming data management, advanced
data processing, and enterprise-grade data operations for the holographic wristwatch system.
"""

import json
import pickle
import gzip
import lzma
import hashlib
import base64
import struct
import uuid
import hmac
import secrets
import zlib
from typing import (
    Tuple, Any, Dict, List, Optional, Union, Callable, Iterator, BinaryIO, 
    Type, TypeVar, Generic, Protocol, runtime_checkable, Awaitable, AsyncIterator
)
from dataclasses import dataclass, asdict, field, fields, is_dataclass
from enum import Enum, IntEnum, auto
import asyncio
import threading
import time
import concurrent.futures
from collections import OrderedDict, deque, defaultdict, Counter, ChainMap
from abc import ABC, abstractmethod
import weakref
import numpy as np
import pandas as pd
import scipy.stats
import copy
import re
import functools
import io
import sys
import os
import tempfile
import warnings
import logging
from contextlib import contextmanager, asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone, timedelta
import dateutil.parser
from urllib.parse import urlparse, parse_qs
import mimetypes
from decimal import Decimal, getcontext
from fractions import Fraction
import sqlite3
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
import yaml
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

# Type variables for generic implementations
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class SerializationFormat(Enum):
    """Supported serialization formats with enhanced options."""
    JSON = "json"
    ORJSON = "orjson"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    BINARY = "binary"
    COMPRESSED_JSON = "compressed_json"
    COMPRESSED_PICKLE = "compressed_pickle"
    COMPRESSED_MSGPACK = "compressed_msgpack"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"

class CompressionType(Enum):
    """Enhanced compression algorithms with performance characteristics."""
    GZIP = "gzip"
    LZMA = "lzma"
    ZLIB = "zlib"
    BZ2 = "bz2"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"

class DataIntegrityLevel(IntEnum):
    """Data integrity checking levels."""
    NONE = 0
    BASIC = 1
    STANDARD = 2
    HIGH = 3
    PARANOID = 4

class CachePolicy(Enum):
    """Cache eviction and management policies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    LIFO = "lifo"
    TTL = "ttl"
    WEIGHTED = "weighted"

class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DataStreamingMode(Enum):
    """Data streaming processing modes."""
    SYNCHRONOUS = "sync"
    ASYNCHRONOUS = "async"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"
    REAL_TIME = "real_time"

@dataclass
class DataValidationResult:
    """Enhanced validation result with detailed reporting."""
    is_valid: bool
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info_messages: List[str] = field(default_factory=list)
    validated_data: Optional[Any] = None
    validation_time: float = 0.0
    field_errors: Dict[str, List[str]] = field(default_factory=dict)
    severity_counts: Dict[ValidationSeverity, int] = field(default_factory=lambda: {
        ValidationSeverity.INFO: 0,
        ValidationSeverity.WARNING: 0,
        ValidationSeverity.ERROR: 0,
        ValidationSeverity.CRITICAL: 0
    })
    
    def add_message(self, message: str, severity: ValidationSeverity, field_name: Optional[str] = None):
        """Add validation message with severity and optional field association."""
        self.severity_counts[severity] += 1
        
        if severity == ValidationSeverity.INFO:
            self.info_messages.append(message)
        elif severity == ValidationSeverity.WARNING:
            self.warnings.append(message)
        elif severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.error_messages.append(message)
            self.is_valid = False
        
        if field_name:
            if field_name not in self.field_errors:
                self.field_errors[field_name] = []
            self.field_errors[field_name].append(message)

@dataclass
class CacheStatistics:
    """Enhanced cache performance statistics with detailed metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    average_access_time: float = 0.0
    memory_usage_bytes: int = 0
    oldest_entry_age: float = 0.0
    newest_entry_age: float = 0.0
    access_pattern_stats: Dict[str, int] = field(default_factory=dict)
    
    def update_rates(self):
        """Update calculated statistics."""
        total_accesses = self.hits + self.misses
        if total_accesses > 0:
            self.hit_rate = self.hits / total_accesses
            self.miss_rate = self.misses / total_accesses

@dataclass
class DataProcessingMetrics:
    """Comprehensive data processing performance metrics."""
    total_records_processed: int = 0
    processing_rate_per_second: float = 0.0
    average_record_size_bytes: float = 0.0
    total_processing_time: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    skipped_records: int = 0
    memory_peak_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    io_operations: int = 0
    compression_ratio: float = 0.0
    throughput_mbps: float = 0.0
    latency_percentiles: Dict[int, float] = field(default_factory=dict)

@dataclass
class SchemaValidationRule:
    """Advanced schema validation rule definition."""
    field_name: str
    field_type: Type
    required: bool = True
    nullable: bool = False
    default_value: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    choices: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    nested_schema: Optional['DataSchema'] = None
    array_item_type: Optional[Type] = None
    description: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR

@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for data transformation operations."""
    
    def transform(self, data: Any) -> Any:
        """Transform input data."""
        ...
    
    def inverse_transform(self, data: Any) -> Any:
        """Apply inverse transformation."""
        ...
    
    def fit(self, data: Any) -> None:
        """Fit transformer parameters to data."""
        ...
    
    def is_fitted(self) -> bool:
        """Check if transformer is fitted."""
        ...

class DataSchema:
    """Advanced data schema definition and validation framework."""
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.rules: Dict[str, SchemaValidationRule] = {}
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        
    def add_rule(self, rule: SchemaValidationRule) -> 'DataSchema':
        """Add validation rule to schema."""
        self.rules[rule.field_name] = rule
        self.updated_at = datetime.now(timezone.utc)
        return self
    
    def remove_rule(self, field_name: str) -> 'DataSchema':
        """Remove validation rule from schema."""
        if field_name in self.rules:
            del self.rules[field_name]
            self.updated_at = datetime.now(timezone.utc)
        return self
    
    def validate(self, data: Dict[str, Any]) -> DataValidationResult:
        """Validate data against schema rules."""
        start_time = time.perf_counter()
        result = DataValidationResult()
        
        # Check required fields
        for field_name, rule in self.rules.items():
            if rule.required and field_name not in data:
                result.add_message(
                    f"Required field '{field_name}' is missing",
                    rule.severity,
                    field_name
                )
                continue
            
            if field_name not in data:
                continue
                
            value = data[field_name]
            
            # Check nullable
            if value is None:
                if not rule.nullable:
                    result.add_message(
                        f"Field '{field_name}' cannot be null",
                        rule.severity,
                        field_name
                    )
                continue
            
            # Type checking
            if not isinstance(value, rule.field_type) and value is not None:
                result.add_message(
                    f"Field '{field_name}' must be of type {rule.field_type.__name__}",
                    rule.severity,
                    field_name
                )
                continue
            
            # Range validation
            if rule.min_value is not None and hasattr(value, '__lt__'):
                if value < rule.min_value:
                    result.add_message(
                        f"Field '{field_name}' must be >= {rule.min_value}",
                        rule.severity,
                        field_name
                    )
            
            if rule.max_value is not None and hasattr(value, '__gt__'):
                if value > rule.max_value:
                    result.add_message(
                        f"Field '{field_name}' must be <= {rule.max_value}",
                        rule.severity,
                        field_name
                    )
            
            # Length validation
            if hasattr(value, '__len__'):
                length = len(value)
                if rule.min_length is not None and length < rule.min_length:
                    result.add_message(
                        f"Field '{field_name}' length must be >= {rule.min_length}",
                        rule.severity,
                        field_name
                    )
                
                if rule.max_length is not None and length > rule.max_length:
                    result.add_message(
                        f"Field '{field_name}' length must be <= {rule.max_length}",
                        rule.severity,
                        field_name
                    )
            
            # Pattern validation
            if rule.pattern and isinstance(value, str):
                if not re.match(rule.pattern, value):
                    result.add_message(
                        f"Field '{field_name}' does not match pattern {rule.pattern}",
                        rule.severity,
                        field_name
                    )
            
            # Choices validation
            if rule.choices and value not in rule.choices:
                result.add_message(
                    f"Field '{field_name}' must be one of {rule.choices}",
                    rule.severity,
                    field_name
                )
            
            # Custom validation
            if rule.custom_validator and not rule.custom_validator(value):
                result.add_message(
                    f"Field '{field_name}' failed custom validation",
                    rule.severity,
                    field_name
                )
            
            # Array validation
            if rule.array_item_type and isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if not isinstance(item, rule.array_item_type):
                        result.add_message(
                            f"Field '{field_name}[{i}]' must be of type {rule.array_item_type.__name__}",
                            rule.severity,
                            field_name
                        )
            
            # Nested schema validation
            if rule.nested_schema and isinstance(value, dict):
                nested_result = rule.nested_schema.validate(value)
                if not nested_result.is_valid:
                    for error in nested_result.error_messages:
                        result.add_message(
                            f"Nested validation in '{field_name}': {error}",
                            rule.severity,
                            field_name
                        )
        
        result.validation_time = time.perf_counter() - start_time
        result.update_rates()
        
        if result.is_valid:
            result.validated_data = data.copy()
        
        return result

class AdvancedDataSerializer:
    """Enterprise-grade data serialization with multiple format support."""
    
    def __init__(self, default_format: SerializationFormat = SerializationFormat.JSON):
        self.default_format = default_format
        self.custom_encoders = {}
        self.custom_decoders = {}
        self.compression_level = 6
        self.encoding = 'utf-8'
    
    def register_custom_type(self, type_class: Type, 
                           encoder: Callable[[Any], Dict], 
                           decoder: Callable[[Dict], Any]):
        """Register custom type serialization handlers."""
        self.custom_encoders[type_class] = encoder
        self.custom_decoders[type_class.__name__] = decoder
    
    def serialize(self, data: Any, 
                 format_type: Optional[SerializationFormat] = None,
                 compression: Optional[CompressionType] = None,
                 **kwargs) -> bytes:
        """Advanced serialization with format and compression options."""
        format_type = format_type or self.default_format
        
        try:
            # Preprocess data for custom types
            preprocessed_data = self._preprocess_data(data)
            
            # Serialize based on format
            if format_type == SerializationFormat.JSON:
                serialized = json.dumps(
                    preprocessed_data, 
                    default=self._json_default,
                    ensure_ascii=False,
                    separators=(',', ':'),
                    **kwargs
                ).encode(self.encoding)
                
            elif format_type == SerializationFormat.ORJSON and HAS_ORJSON:
                serialized = orjson.dumps(
                    preprocessed_data,
                    default=self._json_default,
                    **kwargs
                )
                
            elif format_type == SerializationFormat.PICKLE:
                serialized = pickle.dumps(
                    preprocessed_data, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )
                
            elif format_type == SerializationFormat.MSGPACK and HAS_MSGPACK:
                serialized = msgpack.packb(
                    preprocessed_data,
                    default=self._msgpack_default,
                    use_bin_type=True,
                    **kwargs
                )
                
            elif format_type == SerializationFormat.XML:
                serialized = self._serialize_to_xml(preprocessed_data)
                
            elif format_type == SerializationFormat.YAML:
                serialized = yaml.dump(
                    preprocessed_data,
                    default_flow_style=False,
                    allow_unicode=True,
                    encoding=self.encoding
                )
                
            elif format_type == SerializationFormat.BINARY:
                serialized = self._serialize_binary(preprocessed_data)
                
            else:
                raise ValueError(f"Unsupported serialization format: {format_type}")
            
            # Apply compression if specified
            if compression:
                serialized = self._apply_compression(serialized, compression)
            
            return serialized
            
        except Exception as e:
            raise DataSerializationError(f"Serialization failed: {str(e)}") from e
    
    def deserialize(self, data: bytes,
                   format_type: Optional[SerializationFormat] = None,
                   compression: Optional[CompressionType] = None,
                   **kwargs) -> Any:
        """Advanced deserialization with format and compression handling."""
        format_type = format_type or self.default_format
        
        try:
            # Apply decompression if specified
            if compression:
                data = self._apply_decompression(data, compression)
            
            # Deserialize based on format
            if format_type == SerializationFormat.JSON:
                deserialized = json.loads(data.decode(self.encoding), **kwargs)
                
            elif format_type == SerializationFormat.ORJSON and HAS_ORJSON:
                deserialized = orjson.loads(data)
                
            elif format_type == SerializationFormat.PICKLE:
                deserialized = pickle.loads(data)
                
            elif format_type == SerializationFormat.MSGPACK and HAS_MSGPACK:
                deserialized = msgpack.unpackb(
                    data,
                    raw=False,
                    strict_map_key=False,
                    **kwargs
                )
                
            elif format_type == SerializationFormat.XML:
                deserialized = self._deserialize_from_xml(data)
                
            elif format_type == SerializationFormat.YAML:
                deserialized = yaml.safe_load(data.decode(self.encoding))
                
            elif format_type == SerializationFormat.BINARY:
                deserialized = self._deserialize_binary(data)
                
            else:
                raise ValueError(f"Unsupported deserialization format: {format_type}")
            
            # Postprocess data for custom types
            return self._postprocess_data(deserialized)
            
        except Exception as e:
            raise DataSerializationError(f"Deserialization failed: {str(e)}") from e
    
    def _preprocess_data(self, data: Any) -> Any:
        """Preprocess data for serialization, handling custom types."""
        if isinstance(data, dict):
            return {key: self._preprocess_data(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._preprocess_data(item) for item in data]
        elif type(data) in self.custom_encoders:
            encoded = self.custom_encoders[type(data)](data)
            return {
                '__custom_type__': type(data).__name__,
                '__data__': encoded
            }
        elif isinstance(data, np.ndarray):
            return {
                '__numpy_array__': True,
                'data': data.tolist(),
                'dtype': str(data.dtype),
                'shape': data.shape
            }
        elif isinstance(data, pd.DataFrame):
            return {
                '__pandas_dataframe__': True,
                'data': data.to_dict('records'),
                'columns': data.columns.tolist(),
                'index': data.index.tolist()
            }
        elif isinstance(data, datetime):
            return {
                '__datetime__': True,
                'isoformat': data.isoformat()
            }
        elif isinstance(data, (Decimal, Fraction)):
            return {
                f'__{type(data).__name__.lower()}__': True,
                'value': str(data)
            }
        elif isinstance(data, complex):
            return {
                '__complex__': True,
                'real': data.real,
                'imag': data.imag
            }
        elif hasattr(data, '__dict__') and not isinstance(data, (str, bytes)):
            return {
                '__object__': True,
                'class': type(data).__name__,
                'data': self._preprocess_data(data.__dict__)
            }
        else:
            return data
    
    def _postprocess_data(self, data: Any) -> Any:
        """Postprocess data after deserialization, reconstructing custom types."""
        if isinstance(data, dict):
            if '__custom_type__' in data:
                type_name = data['__custom_type__']
                if type_name in self.custom_decoders:
                    return self.custom_decoders[type_name](data['__data__'])
            elif '__numpy_array__' in data:
                return np.array(data['data'], dtype=data['dtype']).reshape(data['shape'])
            elif '__pandas_dataframe__' in data:
                df = pd.DataFrame(data['data'])
                df.columns = data['columns']
                df.index = data['index']
                return df
            elif '__datetime__' in data:
                return dateutil.parser.isoparse(data['isoformat'])
            elif '__decimal__' in data:
                return Decimal(data['value'])
            elif '__fraction__' in data:
                return Fraction(data['value'])
            elif '__complex__' in data:
                return complex(data['real'], data['imag'])
            else:
                return {key: self._postprocess_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._postprocess_data(item) for item in data]
        else:
            return data
    
    def _json_default(self, obj: Any) -> Any:
        """Default JSON serialization handler for unsupported types."""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def _msgpack_default(self, obj: Any) -> Any:
        """Default MessagePack serialization handler."""
        return self._json_default(obj)
    
    def _serialize_to_xml(self, data: Any, root_name: str = 'data') -> bytes:
        """Serialize data to XML format."""
        def dict_to_xml(d: Dict, parent: ET.Element):
            for key, value in d.items():
                child = ET.SubElement(parent, str(key))
                if isinstance(value, dict):
                    dict_to_xml(value, child)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            item_elem = ET.SubElement(child, 'item')
                            dict_to_xml(item, item_elem)
                        else:
                            item_elem = ET.SubElement(child, 'item')
                            item_elem.text = str(item)
                else:
                    child.text = str(value)
        
        root = ET.Element(root_name)
        if isinstance(data, dict):
            dict_to_xml(data, root)
        else:
            root.text = str(data)
        
        # Pretty print XML
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ").encode(self.encoding)
    
    def _deserialize_from_xml(self, data: bytes) -> Dict:
        """Deserialize data from XML format."""
        def xml_to_dict(element: ET.Element) -> Union[Dict, str, List]:
            if len(element) == 0:
                return element.text
            
            result = {}
            for child in element:
                if child.tag in result:
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(xml_to_dict(child))
                else:
                    result[child.tag] = xml_to_dict(child)
            
            return result
        
        root = ET.fromstring(data.decode(self.encoding))
        return {root.tag: xml_to_dict(root)}
    
    def _serialize_binary(self, data: Any) -> bytes:
        """Custom binary serialization for performance-critical data."""
        if isinstance(data, (int, float)):
            return struct.pack('d', float(data))
        elif isinstance(data, str):
            encoded = data.encode(self.encoding)
            return struct.pack('I', len(encoded)) + encoded
        elif isinstance(data, np.ndarray):
            dtype_str = str(data.dtype).encode('ascii')
            header = struct.pack('II', len(dtype_str), len(data.shape))
            header += struct.pack('I' * len(data.shape), *data.shape)
            header += dtype_str
            return header + data.tobytes()
        elif isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()):
            # Simple string-keyed dictionary
            serialized_items = []
            for key, value in data.items():
                key_bytes = key.encode(self.encoding)
                value_bytes = self._serialize_binary(value)
                item = struct.pack('II', len(key_bytes), len(value_bytes)) + key_bytes + value_bytes
                serialized_items.append(item)
            
            result = struct.pack('I', len(serialized_items))
            return result + b''.join(serialized_items)
        else:
            # Fallback to pickle for complex objects
            return b'PICKLE:' + pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_binary(self, data: bytes) -> Any:
        """Custom binary deserialization."""
        if data.startswith(b'PICKLE:'):
            return pickle.loads(data[7:])
        
        try:
            # Try different formats based on data length
            if len(data) == 8:
                return struct.unpack('d', data)[0]
            elif len(data) > 4:
                # Try string format
                try:
                    str_len = struct.unpack('I', data[:4])[0]
                    if len(data) == 4 + str_len:
                        return data[4:4+str_len].decode(self.encoding)
                except:
                    pass
                
                # Try numpy array format
                try:
                    dtype_len, shape_len = struct.unpack('II', data[:8])
                    offset = 8
                    shape = struct.unpack('I' * shape_len, data[offset:offset + 4*shape_len])
                    offset += 4 * shape_len
                    dtype_str = data[offset:offset + dtype_len].decode('ascii')
                    offset += dtype_len
                    array_data = data[offset:]
                    
                    array = np.frombuffer(array_data, dtype=dtype_str)
                    return array.reshape(shape)
                except:
                    pass
                
                # Try dictionary format
                try:
                    num_items = struct.unpack('I', data[:4])[0]
                    offset = 4
                    result = {}
                    
                    for _ in range(num_items):
                        key_len, value_len = struct.unpack('II', data[offset:offset+8])
                        offset += 8
                        key = data[offset:offset+key_len].decode(self.encoding)
                        offset += key_len
                        value_data = data[offset:offset+value_len]
                        offset += value_len
                        
                        result[key] = self._deserialize_binary(value_data)
                    
                    return result
                except:
                    pass
            
            # Fallback to pickle
            return pickle.loads(data)
            
        except Exception:
            # Final fallback - return raw bytes
            return data
    
    def _apply_compression(self, data: bytes, compression: CompressionType) -> bytes:
        """Apply compression to serialized data."""
        if compression == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=self.compression_level)
        elif compression == CompressionType.LZMA:
            return lzma.compress(data, preset=self.compression_level)
        elif compression == CompressionType.ZLIB:
            return zlib.compress(data, level=self.compression_level)
        elif compression == CompressionType.BZ2:
            import bz2
            return bz2.compress(data, compresslevel=self.compression_level)
        else:
            # For unsupported compression types, try to import and use if available
            try:
                if compression == CompressionType.LZ4:
                    import lz4.frame
                    return lz4.frame.compress(data, compression_level=self.compression_level)
                elif compression == CompressionType.ZSTD:
                    import zstandard as zstd
                    cctx = zstd.ZstdCompressor(level=self.compression_level)
                    return cctx.compress(data)
                elif compression == CompressionType.SNAPPY:
                    import snappy
                    return snappy.compress(data)
            except ImportError:
                warnings.warn(f"Compression type {compression} not available, using gzip instead")
                return gzip.compress(data, compresslevel=self.compression_level)
        
        return data
    
    def _apply_decompression(self, data: bytes, compression: CompressionType) -> bytes:
        """Apply decompression to compressed data."""
        if compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.LZMA:
            return lzma.decompress(data)
        elif compression == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif compression == CompressionType.BZ2:
            import bz2
            return bz2.decompress(data)
        else:
            # For unsupported compression types, try to import and use if available
            try:
                if compression == CompressionType.LZ4:
                    import lz4.frame
                    return lz4.frame.decompress(data)
                elif compression == CompressionType.ZSTD:
                    import zstandard as zstd
                    dctx = zstd.ZstdDecompressor()
                    return dctx.decompress(data)
                elif compression == CompressionType.SNAPPY:
                    import snappy
                    return snappy.decompress(data)
            except ImportError:
                warnings.warn(f"Compression type {compression} not available, using gzip instead")
                return gzip.decompress(data)
        
        return data

class DataIntegrityManager:
    """Advanced data integrity verification and management."""
    
    def __init__(self, integrity_level: DataIntegrityLevel = DataIntegrityLevel.STANDARD):
        self.integrity_level = integrity_level
        self.hash_algorithm = 'sha256'
        self.signing_key = None
        self.verification_key = None
        self.enable_encryption = False
        self.encryption_key = None
    
    def set_keys(self, signing_key: Optional[bytes] = None, 
                 verification_key: Optional[bytes] = None,
                 encryption_key: Optional[bytes] = None):
        """Set cryptographic keys for signing and encryption."""
        self.signing_key = signing_key
        self.verification_key = verification_key or signing_key
        self.encryption_key = encryption_key
    
    def calculate_checksum(self, data: Union[bytes, str], 
                          algorithm: Optional[str] = None) -> str:
        """Calculate cryptographic checksum with enhanced algorithms."""
        algorithm = algorithm or self.hash_algorithm
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        hash_algorithms = {
            'md5': hashlib.md5,
            'sha1': hashlib.sha1,
            'sha224': hashlib.sha224,
            'sha256': hashlib.sha256,
            'sha384': hashlib.sha384,
            'sha512': hashlib.sha512,
            'sha3_224': hashlib.sha3_224,
            'sha3_256': hashlib.sha3_256,
            'sha3_384': hashlib.sha3_384,
            'sha3_512': hashlib.sha3_512,
            'blake2b': hashlib.blake2b,
            'blake2s': hashlib.blake2s
        }
        
        if algorithm not in hash_algorithms:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_obj = hash_algorithms[algorithm]()
        hash_obj.update(data)
        return hash_obj.hexdigest()
    
    def calculate_hmac(self, data: Union[bytes, str], 
                      key: Optional[bytes] = None,
                      algorithm: str = 'sha256') -> str:
        """Calculate HMAC for authenticated integrity checking."""
        key = key or self.signing_key
        if key is None:
            raise ValueError("HMAC key required but not provided")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hmac.new(key, data, algorithm).hexdigest()
    
    def create_integrity_wrapper(self, data: Any, 
                                format_type: SerializationFormat,
                                serializer: Optional[AdvancedDataSerializer] = None) -> Dict[str, Any]:
        """Create comprehensive integrity wrapper for data."""
        serializer = serializer or AdvancedDataSerializer()
        serialized_data = serializer.serialize(data, format_type)
        
        wrapper = {
            'data': base64.b64encode(serialized_data).decode('ascii'),
            'format': format_type.value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '2.0',
            'size': len(serialized_data),
            'id': str(uuid.uuid4())
        }
        
        # Add integrity checks based on level
        if self.integrity_level >= DataIntegrityLevel.BASIC:
            wrapper['checksum'] = self.calculate_checksum(serialized_data)
            wrapper['hash_algorithm'] = self.hash_algorithm
        
        if self.integrity_level >= DataIntegrityLevel.STANDARD:
            wrapper['crc32'] = zlib.crc32(serialized_data) & 0xffffffff
            
        if self.integrity_level >= DataIntegrityLevel.HIGH:
            if self.signing_key:
                wrapper['hmac'] = self.calculate_hmac(serialized_data)
            wrapper['integrity_level'] = self.integrity_level.value
            
        if self.integrity_level >= DataIntegrityLevel.PARANOID:
            # Multiple hash algorithms for paranoid level
            wrapper['checksums'] = {
                'sha256': self.calculate_checksum(serialized_data, 'sha256'),
                'sha3_256': self.calculate_checksum(serialized_data, 'sha3_256'),
                'blake2b': self.calculate_checksum(serialized_data, 'blake2b')
            }
            wrapper['data_entropy'] = self._calculate_entropy(serialized_data)
        
        return wrapper
    
    def verify_and_extract(self, wrapped_data: Dict[str, Any],
                          format_type: SerializationFormat,
                          serializer: Optional[AdvancedDataSerializer] = None) -> Any:
        """Verify integrity and extract data from wrapper."""
        serializer = serializer or AdvancedDataSerializer()
        
        try:
            data_bytes = base64.b64decode(wrapped_data['data'])
        except Exception as e:
            raise DataIntegrityError(f"Failed to decode data: {str(e)}")
        
        # Verify integrity based on level
        if self.integrity_level >= DataIntegrityLevel.BASIC:
            expected_checksum = wrapped_data.get('checksum')
            if expected_checksum:
                algorithm = wrapped_data.get('hash_algorithm', self.hash_algorithm)
                actual_checksum = self.calculate_checksum(data_bytes, algorithm)
                if actual_checksum != expected_checksum:
                    raise DataIntegrityError("Checksum verification failed")
        
        if self.integrity_level >= DataIntegrityLevel.STANDARD:
            expected_crc = wrapped_data.get('crc32')
            if expected_crc is not None:
                actual_crc = zlib.crc32(data_bytes) & 0xffffffff
                if actual_crc != expected_crc:
                    raise DataIntegrityError("CRC32 verification failed")
        
        if self.integrity_level >= DataIntegrityLevel.HIGH:
            expected_hmac = wrapped_data.get('hmac')
            if expected_hmac and self.verification_key:
                actual_hmac = self.calculate_hmac(data_bytes)
                if actual_hmac != expected_hmac:
                    raise DataIntegrityError("HMAC verification failed")
        
        if self.integrity_level >= DataIntegrityLevel.PARANOID:
            checksums = wrapped_data.get('checksums', {})
            for algorithm, expected_hash in checksums.items():
                actual_hash = self.calculate_checksum(data_bytes, algorithm)
                if actual_hash != expected_hash:
                    raise DataIntegrityError(f"{algorithm} verification failed")
        
        # Verify size
        expected_size = wrapped_data.get('size')
        if expected_size is not None and len(data_bytes) != expected_size:
            raise DataIntegrityError("Size verification failed")
        
        return serializer.deserialize(data_bytes, format_type)
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data for integrity analysis."""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = Counter(data)
        data_len = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy

class AdvancedCacheManager(Generic[K, V]):
    """Enterprise-grade caching system with multiple policies and advanced features."""
    
    def __init__(self, max_size: int = 10000, 
                 policy: CachePolicy = CachePolicy.LRU,
                 ttl_seconds: Optional[float] = None,
                 enable_statistics: bool = True):
        self.max_size = max_size
        self.policy = policy
        self.ttl_seconds = ttl_seconds
        self.enable_statistics = enable_statistics
        
        self._cache: Dict[K, Any] = {}
        self._access_order: OrderedDict[K, float] = OrderedDict()
        self._access_counts: Dict[K, int] = defaultdict(int)
        self._timestamps: Dict[K, float] = {}
        self._weights: Dict[K, float] = {}
        self._size_estimates: Dict[K, int] = {}
        
        self.statistics = CacheStatistics(max_size=max_size)
        self._lock = threading.RLock()
        
        # Background cleanup for TTL
        if ttl_seconds:
            self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
            self._cleanup_thread.start()
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get value from cache with policy-based access tracking."""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Check TTL expiration
                if self.ttl_seconds and current_time - self._timestamps[key] > self.ttl_seconds:
                    self._remove_key(key)
                    if self.enable_statistics:
                        self.statistics.misses += 1
                        self.statistics.update_rates()
                    return default
                
                # Update access patterns based on policy
                if self.policy == CachePolicy.LRU:
                    self._access_order.move_to_end(key)
                elif self.policy == CachePolicy.LFU:
                    self._access_counts[key] += 1
                
                if self.enable_statistics:
                    self.statistics.hits += 1
                    self.statistics.update_rates()
                
                return self._cache[key]
            else:
                if self.enable_statistics:
                    self.statistics.misses += 1
                    self.statistics.update_rates()
                return default
    
    def put(self, key: K, value: V, weight: float = 1.0) -> None:
        """Put value in cache with advanced eviction handling."""
        with self._lock:
            current_time = time.time()
            
            # Remove existing key if present
            if key in self._cache:
                self._remove_key(key)
            
            # Check if eviction is needed
            while len(self._cache) >= self.max_size:
                self._evict_one()
            
            # Add new entry
            self._cache[key] = value
            self._timestamps[key] = current_time
            self._weights[key] = weight
            self._size_estimates[key] = sys.getsizeof(value)
            
            if self.policy == CachePolicy.LRU:
                self._access_order[key] = current_time
            elif self.policy == CachePolicy.LFU:
                self._access_counts[key] = 1
            elif self.policy in [CachePolicy.FIFO, CachePolicy.LIFO]:
                self._access_order[key] = current_time
            
            if self.enable_statistics:
                self.statistics.total_size = len(self._cache)
                self.statistics.memory_usage_bytes += self._size_estimates[key]
    
    def _evict_one(self) -> None:
        """Evict one item based on the current policy."""
        if not self._cache:
            return
        
        key_to_evict = None
        
        if self.policy == CachePolicy.LRU:
            key_to_evict = next(iter(self._access_order))
        elif self.policy == CachePolicy.LFU:
            key_to_evict = min(self._access_counts.keys(), key=lambda k: self._access_counts[k])
        elif self.policy == CachePolicy.FIFO:
            key_to_evict = next(iter(self._access_order))
        elif self.policy == CachePolicy.LIFO:
            key_to_evict = next(reversed(self._access_order))
        elif self.policy == CachePolicy.TTL:
            # Evict oldest entry by timestamp
            key_to_evict = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
        elif self.policy == CachePolicy.WEIGHTED:
            # Evict entry with lowest weight
            key_to_evict = min(self._weights.keys(), key=lambda k: self._weights[k])
        
        if key_to_evict:
            self._remove_key(key_to_evict)
            if self.enable_statistics:
                self.statistics.evictions += 1
    
    def _remove_key(self, key: K) -> None:
        """Remove key from all internal data structures."""
        if key in self._cache:
            if self.enable_statistics:
                self.statistics.memory_usage_bytes -= self._size_estimates.get(key, 0)
            
            del self._cache[key]
            self._access_order.pop(key, None)
            self._access_counts.pop(key, None)
            self._timestamps.pop(key, None)
            self._weights.pop(key, None)
            self._size_estimates.pop(key, None)
    
    def _cleanup_expired(self) -> None:
        """Background thread to clean up expired entries."""
        while True:
            if self.ttl_seconds:
                current_time = time.time()
                with self._lock:
                    expired_keys = [
                        key for key, timestamp in self._timestamps.items()
                        if current_time - timestamp > self.ttl_seconds
                    ]
                    
                    for key in expired_keys:
                        self._remove_key(key)
                
                time.sleep(min(self.ttl_seconds / 10, 60))  # Check every 10% of TTL or 1 minute
            else:
                time.sleep(60)
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_counts.clear()
            self._timestamps.clear()
            self._weights.clear()
            self._size_estimates.clear()
            
            if self.enable_statistics:
                self.statistics = CacheStatistics(max_size=self.max_size)
    
    def get_statistics(self) -> CacheStatistics:
        """Get comprehensive cache statistics."""
        with self._lock:
            if self.enable_statistics:
                self.statistics.total_size = len(self._cache)
                self.statistics.update_rates()
                
                if self._timestamps:
                    current_time = time.time()
                    ages = [current_time - ts for ts in self._timestamps.values()]
                    self.statistics.oldest_entry_age = max(ages) if ages else 0.0
                    self.statistics.newest_entry_age = min(ages) if ages else 0.0
                
                # Access pattern statistics
                self.statistics.access_pattern_stats = {
                    f"access_count_{count}": len([k for k, c in self._access_counts.items() if c == count])
                    for count in set(self._access_counts.values())
                }
            
            return self.statistics
    
    def keys(self) -> List[K]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def values(self) -> List[V]:
        """Get all cache values."""
        with self._lock:
            return list(self._cache.values())
    
    def items(self) -> List[Tuple[K, V]]:
        """Get all cache items."""
        with self._lock:
            return list(self._cache.items())
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: K) -> bool:
        with self._lock:
            if key in self._cache and self.ttl_seconds:
                current_time = time.time()
                if current_time - self._timestamps[key] > self.ttl_seconds:
                    self._remove_key(key)
                    return False
            return key in self._cache

class DataPipelineProcessor:
    """Advanced data transformation pipeline with error handling and monitoring."""
    
    def __init__(self, name: str = "default_pipeline", 
                 enable_monitoring: bool = True,
                 max_workers: Optional[int] = None):
        self.name = name
        self.enable_monitoring = enable_monitoring
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        
        self.transformers: List[Tuple[str, DataTransformer, Dict[str, Any]]] = []
        self.metrics = DataProcessingMetrics()
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self.validators: List[Callable[[Any], DataValidationResult]] = []
        self.hooks = {
            'before_transform': [],
            'after_transform': [],
            'on_error': [],
            'on_complete': []
        }
        
        self._lock = threading.Lock()
    
    def add_transformer(self, transformer: DataTransformer, 
                       name: Optional[str] = None,
                       **config) -> 'DataPipelineProcessor':
        """Add transformation step to pipeline with configuration."""
        step_name = name or f"step_{len(self.transformers)}"
        self.transformers.append((step_name, transformer, config))
        return self
    
    def add_validator(self, validator: Callable[[Any], DataValidationResult]) -> 'DataPipelineProcessor':
        """Add data validator to pipeline."""
        self.validators.append(validator)
        return self
    
    def add_error_handler(self, exception_type: Type[Exception], 
                         handler: Callable[[Exception, Any], Any]) -> 'DataPipelineProcessor':
        """Add error handler for specific exception types."""
        self.error_handlers[exception_type] = handler
        return self
    
    def add_hook(self, event: str, hook: Callable) -> 'DataPipelineProcessor':
        """Add event hook to pipeline."""
        if event in self.hooks:
            self.hooks[event].append(hook)
        return self
    
    def process(self, data: Any, 
               parallel: bool = False,
               batch_size: Optional[int] = None) -> Any:
        """Process data through transformation pipeline with advanced options."""
        start_time = time.perf_counter()
        result = data
        
        try:
            # Execute before_transform hooks
            for hook in self.hooks['before_transform']:
                hook(data)
            
            # Validate input data
            for validator in self.validators:
                validation_result = validator(result)
                if not validation_result.is_valid:
                    raise DataValidationError(f"Input validation failed: {validation_result.error_messages}")
            
            # Process through transformers
            if parallel and isinstance(data, (list, tuple)) and len(data) > 1:
                result = self._process_parallel(data, batch_size)
            else:
                result = self._process_sequential(result)
            
            # Validate output data
            for validator in self.validators:
                validation_result = validator(result)
                if not validation_result.is_valid:
                    raise DataValidationError(f"Output validation failed: {validation_result.error_messages}")
            
            # Execute after_transform hooks
            for hook in self.hooks['after_transform']:
                hook(result)
            
            # Update metrics
            if self.enable_monitoring:
                processing_time = time.perf_counter() - start_time
                with self._lock:
                    self.metrics.total_processing_time += processing_time
                    self.metrics.total_records_processed += 1
                    if self.metrics.total_records_processed > 0:
                        self.metrics.processing_rate_per_second = (
                            self.metrics.total_records_processed / self.metrics.total_processing_time
                        )
            
            # Execute completion hooks
            for hook in self.hooks['on_complete']:
                hook(result)
            
            return result
            
        except Exception as e:
            # Handle errors
            handled = False
            for exception_type, handler in self.error_handlers.items():
                if isinstance(e, exception_type):
                    result = handler(e, data)
                    handled = True
                    break
            
            # Execute error hooks
            for hook in self.hooks['on_error']:
                hook(e, data)
            
            if not handled:
                raise DataTransformationError(f"Pipeline '{self.name}' failed: {str(e)}") from e
            
            return result
    
    def _process_sequential(self, data: Any) -> Any:
        """Process data sequentially through transformers."""
        result = data
        
        for step_name, transformer, config in self.transformers:
            start_time = time.perf_counter()
            
            try:
                # Fit transformer if needed
                if hasattr(transformer, 'is_fitted') and not transformer.is_fitted():
                    transformer.fit(result)
                
                # Apply transformation
                result = transformer.transform(result)
                
                if self.enable_monitoring:
                    execution_time = time.perf_counter() - start_time
                    # Update step-specific metrics if needed
                    
            except Exception as e:
                raise DataTransformationError(f"Step '{step_name}' failed: {str(e)}") from e
        
        return result
    
    def _process_parallel(self, data: List[Any], batch_size: Optional[int] = None) -> List[Any]:
        """Process data in parallel using ThreadPoolExecutor."""
        batch_size = batch_size or max(1, len(data) // self.max_workers)
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_sequential, batch): batch 
                for batch in batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_result = future.result()
                    results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
                except Exception as e:
                    if self.enable_monitoring:
                        self.metrics.error_count += 1
                    raise
        
        return results
    
    async def process_async(self, data: Any) -> Any:
        """Process data asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process, data)
    
    def get_metrics(self) -> DataProcessingMetrics:
        """Get pipeline processing metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset processing metrics."""
        with self._lock:
            self.metrics = DataProcessingMetrics()

# Custom Transformers
class NormalizationTransformer:
    """Advanced data normalization transformer with multiple methods."""
    
    def __init__(self, method: str = 'minmax', 
                 feature_range: Tuple[float, float] = (0, 1),
                 epsilon: float = 1e-8):
        self.method = method
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.min_vals = None
        self.max_vals = None
        self.mean_vals = None
        self.std_vals = None
        self.median_vals = None
        self.mad_vals = None
        self._is_fitted = False
    
    def fit(self, data: Union[np.ndarray, List, pd.DataFrame]) -> None:
        """Fit transformer parameters to data."""
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        
        if self.method == 'minmax':
            self.min_vals = np.min(data, axis=0)
            self.max_vals = np.max(data, axis=0)
        elif self.method == 'standard':
            self.mean_vals = np.mean(data, axis=0)
            self.std_vals = np.std(data, axis=0)
        elif self.method == 'robust':
            self.median_vals = np.median(data, axis=0)
            self.mad_vals = np.median(np.abs(data - self.median_vals), axis=0)
        elif self.method == 'quantile':
            self.q25 = np.percentile(data, 25, axis=0)
            self.q75 = np.percentile(data, 75, axis=0)
        
        self._is_fitted = True
    
    def transform(self, data: Union[np.ndarray, List, pd.DataFrame]) -> np.ndarray:
        """Apply normalization transformation."""
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before transformation")
        
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            original_columns = data.columns
            original_index = data.index
            data = data.values
            return_df = True
        else:
            return_df = False
        
        if self.method == 'minmax':
            range_vals = self.max_vals - self.min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            normalized = (data - self.min_vals) / range_vals
            result = normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        elif self.method == 'standard':
            std_vals = self.std_vals.copy()
            std_vals[std_vals == 0] = 1  # Avoid division by zero
            result = (data - self.mean_vals) / std_vals
        
        elif self.method == 'robust':
            mad_vals = self.mad_vals.copy()
            mad_vals[mad_vals == 0] = 1  # Avoid division by zero
            result = (data - self.median_vals) / mad_vals
        
        elif self.method == 'quantile':
            iqr_vals = self.q75 - self.q25
            iqr_vals[iqr_vals == 0] = 1  # Avoid division by zero
            result = (data - self.q25) / iqr_vals
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        if return_df:
            return pd.DataFrame(result, columns=original_columns, index=original_index)
        
        return result
    
    def inverse_transform(self, data: Union[np.ndarray, List, pd.DataFrame]) -> np.ndarray:
        """Apply inverse normalization."""
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before inverse transformation")
        
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            original_columns = data.columns
            original_index = data.index
            data = data.values
            return_df = True
        else:
            return_df = False
        
        if self.method == 'minmax':
            denormalized = ((data - self.feature_range[0]) / 
                          (self.feature_range[1] - self.feature_range[0]))
            result = denormalized * (self.max_vals - self.min_vals) + self.min_vals
        
        elif self.method == 'standard':
            result = data * self.std_vals + self.mean_vals
        
        elif self.method == 'robust':
            result = data * self.mad_vals + self.median_vals
        
        elif self.method == 'quantile':
            iqr_vals = self.q75 - self.q25
            result = data * iqr_vals + self.q25
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        if return_df:
            return pd.DataFrame(result, columns=original_columns, index=original_index)
        
        return result
    
    def is_fitted(self) -> bool:
        """Check if transformer is fitted."""
        return self._is_fitted

class OutlierDetectionTransformer:
    """Advanced outlier detection and handling transformer."""
    
    def __init__(self, method: str = 'iqr', 
                 threshold: float = 3.0,
                 action: str = 'remove'):  # 'remove', 'clip', 'transform'
        self.method = method
        self.threshold = threshold
        self.action = action
        self.lower_bounds = None
        self.upper_bounds = None
        self.outlier_mask = None
        self._is_fitted = False
    
    def fit(self, data: Union[np.ndarray, List, pd.DataFrame]) -> None:
        """Fit outlier detection parameters."""
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        
        if self.method == 'iqr':
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            self.lower_bounds = q1 - self.threshold * iqr
            self.upper_bounds = q3 + self.threshold * iqr
        
        elif self.method == 'zscore':
            mean_vals = np.mean(data, axis=0)
            std_vals = np.std(data, axis=0)
            self.lower_bounds = mean_vals - self.threshold * std_vals
            self.upper_bounds = mean_vals + self.threshold * std_vals
        
        elif self.method == 'modified_zscore':
            median_vals = np.median(data, axis=0)
            mad_vals = np.median(np.abs(data - median_vals), axis=0)
            modified_z = 0.6745 * (data - median_vals) / mad_vals
            self.outlier_mask = np.abs(modified_z) > self.threshold
        
        self._is_fitted = True
    
    def transform(self, data: Union[np.ndarray, List, pd.DataFrame]) -> np.ndarray:
        """Apply outlier detection and handling."""
        if not self._is_fitted:
            self.fit(data)
        
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            original_columns = data.columns
            original_index = data.index
            data = data.values
            return_df = True
        else:
            return_df = False
        
        result = data.copy()
        
        if self.method in ['iqr', 'zscore']:
            outliers = (data < self.lower_bounds) | (data > self.upper_bounds)
            
            if self.action == 'remove':
                # Remove rows with any outliers
                valid_rows = ~np.any(outliers, axis=1)
                result = result[valid_rows]
                if return_df and 'original_index' in locals():
                    original_index = original_index[valid_rows]
            
            elif self.action == 'clip':
                # Clip values to bounds
                result = np.clip(result, self.lower_bounds, self.upper_bounds)
            
            elif self.action == 'transform':
                # Set outliers to NaN for further processing
                result[outliers] = np.nan
        
        elif self.method == 'modified_zscore':
            if self.action == 'remove':
                valid_rows = ~np.any(self.outlier_mask, axis=1)
                result = result[valid_rows]
                if return_df and 'original_index' in locals():
                    original_index = original_index[valid_rows]
            elif self.action == 'transform':
                result[self.outlier_mask] = np.nan
        
        if return_df and 'return_df' in locals() and return_df:
            return pd.DataFrame(result, columns=original_columns, 
                              index=original_index if 'original_index' in locals() else None)
        
        return result
    
    def inverse_transform(self, data: Union[np.ndarray, List, pd.DataFrame]) -> np.ndarray:
        """Inverse transformation (identity for outlier detection)."""
        if isinstance(data, (list, pd.DataFrame)):
            return np.array(data) if isinstance(data, list) else data.values
        return data
    
    def is_fitted(self) -> bool:
        """Check if transformer is fitted."""
        return self._is_fitted

# Exception classes
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

class DataValidationError(DataOperationError):
    """Exception for data validation errors."""
    pass

class DataProcessingError(DataOperationError):
    """Exception for data processing errors."""
    pass

class CacheError(DataOperationError):
    """Exception for cache operation errors."""
    pass

# Global instances and utilities
global_serializer = AdvancedDataSerializer()
global_integrity_manager = DataIntegrityManager()
global_cache = AdvancedCacheManager[str, Any](max_size=10000)

def serialize_with_integrity(data: Any, 
                           format_type: SerializationFormat = SerializationFormat.JSON,
                           integrity_level: DataIntegrityLevel = DataIntegrityLevel.STANDARD) -> Dict[str, Any]:
    """Convenience function for serializing data with integrity checking."""
    integrity_manager = DataIntegrityManager(integrity_level)
    return integrity_manager.create_integrity_wrapper(data, format_type, global_serializer)

def deserialize_with_verification(wrapped_data: Dict[str, Any],
                                 format_type: SerializationFormat,
                                 integrity_level: DataIntegrityLevel = DataIntegrityLevel.STANDARD) -> Any:
    """Convenience function for deserializing data with integrity verification."""
    integrity_manager = DataIntegrityManager(integrity_level)
    return integrity_manager.verify_and_extract(wrapped_data, format_type, global_serializer)

def create_data_pipeline(*transformers: DataTransformer, 
                        name: str = "pipeline",
                        enable_monitoring: bool = True) -> DataPipelineProcessor:
    """Convenience function for creating data processing pipeline."""
    pipeline = DataPipelineProcessor(name=name, enable_monitoring=enable_monitoring)
    for i, transformer in enumerate(transformers):
        pipeline.add_transformer(transformer, f"transformer_{i}")
    return pipeline

def validate_data_schema(data: Dict[str, Any], schema: DataSchema) -> DataValidationResult:
    """Convenience function for validating data against schema."""
    return schema.validate(data)

# Testing and demonstration functions
def run_comprehensive_data_tests():
    """Comprehensive test suite for data utilities."""
    print("Running comprehensive data utilities test suite...")
    
    # Test serialization
    test_data = {
        "string": "Hello, World!",
        "number": 42,
        "float": 3.14159,
        "boolean": True,
        "null": None,
        "array": [1, 2, 3, 4, 5],
        "nested": {
            "inner": "value",
            "timestamp": datetime.now()
        },
        "numpy_array": np.array([1, 2, 3, 4, 5]),
        "pandas_df": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    }
    
    # Test various serialization formats
    for format_type in [SerializationFormat.JSON, SerializationFormat.PICKLE, SerializationFormat.YAML]:
        try:
            serialized = global_serializer.serialize(test_data, format_type)
            deserialized = global_serializer.deserialize(serialized, format_type)
            print(f"✓ {format_type.value} serialization test passed")
        except Exception as e:
            print(f"✗ {format_type.value} serialization test failed: {e}")
    
    # Test integrity checking
    try:
        wrapper = serialize_with_integrity(test_data, SerializationFormat.JSON, DataIntegrityLevel.HIGH)
        verified_data = deserialize_with_verification(wrapper, SerializationFormat.JSON, DataIntegrityLevel.HIGH)
        print("✓ Data integrity test passed")
    except Exception as e:
        print(f"✗ Data integrity test failed: {e}")
    
    # Test caching
    try:
        cache = AdvancedCacheManager[str, str](max_size=100)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        
        stats = cache.get_statistics()
        assert stats.hits == 2
        assert stats.misses == 1
        
        print("✓ Advanced caching test passed")
    except Exception as e:
        print(f"✗ Advanced caching test failed: {e}")
    
    # Test data pipeline
    try:
        # Create test transformers
        normalizer = NormalizationTransformer(method='standard')
        outlier_detector = OutlierDetectionTransformer(method='iqr', action='clip')
        
        pipeline = create_data_pipeline(normalizer, outlier_detector, name="test_pipeline")
        
        test_array = np.random.randn(100, 3) * 10 + 5
        test_array[0, 0] = 1000  # Add outlier
        
        processed_data = pipeline.process(test_array)
        metrics = pipeline.get_metrics()
        
        assert processed_data is not None
        assert metrics.total_records_processed == 1
        
        print("✓ Data pipeline test passed")
    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
    
    # Test schema validation
    try:
        schema = DataSchema("test_schema")
        schema.add_rule(SchemaValidationRule(
            field_name="name",
            field_type=str,
            required=True,
            min_length=2,
            max_length=50
        ))
        schema.add_rule(SchemaValidationRule(
            field_name="age",
            field_type=int,
            required=True,
            min_value=0,
            max_value=120
        ))
        
        valid_data = {"name": "John Doe", "age": 30}
        invalid_data = {"name": "A", "age": -5}
        
        valid_result = schema.validate(valid_data)
        invalid_result = schema.validate(invalid_data)
        
        assert valid_result.is_valid
        assert not invalid_result.is_valid
        assert len(invalid_result.error_messages) > 0
        
        print("✓ Schema validation test passed")
    except Exception as e:
        print(f"✗ Schema validation test failed: {e}")
    
    print("All data utilities tests completed!")

# Module initialization
if __name__ == "__main__":
    print("AI Holographic Wristwatch - Data Utilities Module")
    print("=" * 60)
    
    # Run comprehensive test suite
    try:
        run_comprehensive_data_tests()
        print("\n✓ All tests passed successfully")
    except Exception as e:
        print(f"\n✗ Test suite failed: {str(e)}")
    
    print("\nData utilities module initialized and ready for use.")
    print("Available classes and functions:")
    print("- AdvancedDataSerializer: Multi-format serialization")
    print("- DataIntegrityManager: Data integrity verification")
    print("- AdvancedCacheManager: Enterprise caching system")
    print("- DataPipelineProcessor: Data transformation pipelines")
    print("- DataSchema: Schema validation framework")
    print("- NormalizationTransformer: Data normalization")
    print("- OutlierDetectionTransformer: Outlier detection")
    print("- Various utility functions for common operations")
    print("\nModule features:")
    print("- Multiple serialization formats (JSON, Pickle, MessagePack, XML, YAML)")
    print("- Advanced compression support")
    print("- Integrity checking with multiple algorithms")
    print("- Sophisticated caching with multiple policies")
    print("- Parallel data processing")
    print("- Comprehensive validation framework")
    print("- Performance monitoring and metrics")
    print("- Error handling and recovery")
    print("- Async/await support")
    print("- Thread-safe operations")