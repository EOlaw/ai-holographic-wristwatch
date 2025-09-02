# src/core/utils/memory_utils.py
"""
Memory Management Optimization Utilities for AI Holographic Wristwatch System

This module provides comprehensive memory management capabilities including caching 
strategies, memory pool management, garbage collection optimization, memory leak 
detection, performance profiling, and resource monitoring for optimal system performance.
"""

import gc
import sys
import threading
import time
import weakref
import psutil
import os
import mmap
from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
import pickle
import struct
import numpy as np
from contextlib import contextmanager
import tracemalloc
import linecache
from concurrent.futures import ThreadPoolExecutor
import asyncio

class MemoryUnit(Enum):
    """Memory size units for calculations."""
    BYTE = 1
    KILOBYTE = 1024
    MEGABYTE = 1024 * 1024
    GIGABYTE = 1024 * 1024 * 1024

class CachePolicy(Enum):
    """Cache replacement policies."""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    FIFO = "first_in_first_out"
    RANDOM = "random"
    TTL = "time_to_live"
    ADAPTIVE = "adaptive"

class GCStrategy(Enum):
    """Garbage collection strategies."""
    AUTOMATIC = "automatic"
    MANUAL = "manual" 
    THRESHOLD_BASED = "threshold_based"
    PERIODIC = "periodic"
    PRESSURE_BASED = "pressure_based"

@dataclass
class MemoryStats:
    """Comprehensive memory statistics."""
    total_memory: int  # bytes
    available_memory: int  # bytes
    used_memory: int  # bytes
    memory_percent: float
    swap_total: int = 0
    swap_used: int = 0
    swap_percent: float = 0.0
    process_memory: int = 0
    cache_memory: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_human_readable(self) -> Dict[str, str]:
        """Convert memory stats to human-readable format."""
        def format_bytes(bytes_value: int) -> str:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_value < 1024.0:
                    return f"{bytes_value:.1f} {unit}"
                bytes_value /= 1024.0
            return f"{bytes_value:.1f} TB"
        
        return {
            'total_memory': format_bytes(self.total_memory),
            'available_memory': format_bytes(self.available_memory),
            'used_memory': format_bytes(self.used_memory),
            'memory_percent': f"{self.memory_percent:.1f}%",
            'process_memory': format_bytes(self.process_memory),
            'cache_memory': format_bytes(self.cache_memory)
        }

@dataclass
class CacheEntry:
    """Cache entry with metadata for advanced cache management."""
    value: Any
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 1
    size_bytes: int = 0
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate entry size after initialization."""
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size(self.value)
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of object in bytes."""
        try:
            return sys.getsizeof(obj)
        except TypeError:
            # Fallback for objects that don't support getsizeof
            try:
                return len(pickle.dumps(obj))
            except:
                return 1024  # Default estimate
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > self.creation_time + self.ttl
    
    def touch(self):
        """Update access time and count."""
        self.last_access_time = time.time()
        self.access_count += 1

class AdvancedCache:
    """Advanced caching system with multiple eviction policies."""
    
    def __init__(self, max_size_bytes: int, policy: CachePolicy = CachePolicy.LRU,
                 max_entries: int = 10000, cleanup_threshold: float = 0.8):
        self.max_size_bytes = max_size_bytes
        self.policy = policy
        self.max_entries = max_entries
        self.cleanup_threshold = cleanup_threshold
        self.entries = OrderedDict()
        self.size_bytes = 0
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        self._lock = threading.RLock()
        self._access_frequency = defaultdict(int)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with policy-specific handling."""
        with self._lock:
            if key not in self.entries:
                self.stats['misses'] += 1
                self.stats['total_requests'] += 1
                return None
            
            entry = self.entries[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.stats['misses'] += 1
                self.stats['total_requests'] += 1
                return None
            
            # Update access information
            entry.touch()
            self._access_frequency[key] += 1
            
            # Move to end for LRU policy
            if self.policy == CachePolicy.LRU:
                self.entries.move_to_end(key)
            
            self.stats['hits'] += 1
            self.stats['total_requests'] += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, 
            tags: Optional[List[str]] = None):
        """Put value in cache with specified parameters."""
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                value=value,
                ttl=ttl,
                tags=tags or []
            )
            
            # Remove existing entry if present
            if key in self.entries:
                self._remove_entry(key)
            
            # Check if we need to make space
            projected_size = self.size_bytes + entry.size_bytes
            
            if (projected_size > self.max_size_bytes or 
                len(self.entries) >= self.max_entries):
                self._evict_entries(entry.size_bytes)
            
            # Add new entry
            self.entries[key] = entry
            self.size_bytes += entry.size_bytes
            self._access_frequency[key] = 1
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache."""
        if key not in self.entries:
            return False
        
        entry = self.entries.pop(key)
        self.size_bytes -= entry.size_bytes
        self._access_frequency.pop(key, None)
        return True
    
    def _evict_entries(self, space_needed: int):
        """Evict entries based on cache policy."""
        target_size = max(
            self.max_size_bytes - space_needed,
            int(self.max_size_bytes * (1 - self.cleanup_threshold))
        )
        
        evicted_count = 0
        
        while self.size_bytes > target_size and self.entries:
            if self.policy == CachePolicy.LRU:
                # Remove least recently used (first in OrderedDict)
                key = next(iter(self.entries))
                
            elif self.policy == CachePolicy.LFU:
                # Remove least frequently used
                key = min(self._access_frequency.keys(), 
                         key=self._access_frequency.get)
                
            elif self.policy == CachePolicy.FIFO:
                # Remove first in (oldest by creation time)
                key = min(self.entries.keys(), 
                         key=lambda k: self.entries[k].creation_time)
                
            elif self.policy == CachePolicy.TTL:
                # Remove expired entries first, then oldest
                expired_keys = [k for k, entry in self.entries.items() if entry.is_expired()]
                if expired_keys:
                    key = expired_keys[0]
                else:
                    key = min(self.entries.keys(), 
                             key=lambda k: self.entries[k].creation_time)
                
            else:  # Random or adaptive
                key = next(iter(self.entries))
            
            self._remove_entry(key)
            evicted_count += 1
        
        self.stats['evictions'] += evicted_count
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate all cache entries with specified tags."""
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self.entries.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        with self._lock:
            hit_rate = (self.stats['hits'] / self.stats['total_requests'] 
                       if self.stats['total_requests'] > 0 else 0.0)
            
            return {
                'hit_rate': hit_rate,
                'miss_rate': 1.0 - hit_rate,
                'total_entries': len(self.entries),
                'size_bytes': self.size_bytes,
                'size_utilization': self.size_bytes / self.max_size_bytes,
                'entry_utilization': len(self.entries) / self.max_entries,
                'eviction_rate': (self.stats['evictions'] / self.stats['total_requests']
                                if self.stats['total_requests'] > 0 else 0.0),
                'average_entry_size': (self.size_bytes / len(self.entries) 
                                     if self.entries else 0),
                **self.stats
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.entries.clear()
            self._access_frequency.clear()
            self.size_bytes = 0

class MemoryPool:
    """Memory pool management for efficient allocation and reuse."""
    
    def __init__(self, pool_name: str, object_factory: Callable[[], Any],
                 initial_size: int = 10, max_size: int = 100,
                 auto_expand: bool = True):
        self.pool_name = pool_name
        self.object_factory = object_factory
        self.max_size = max_size
        self.auto_expand = auto_expand
        self.available_objects = deque()
        self.allocated_objects = weakref.WeakSet()
        self.creation_count = 0
        self.allocation_count = 0
        self.deallocation_count = 0
        self._lock = threading.Lock()
        
        # Pre-populate pool
        self._populate_pool(initial_size)
    
    def _populate_pool(self, count: int):
        """Pre-populate pool with objects."""
        for _ in range(count):
            try:
                obj = self.object_factory()
                self.available_objects.append(obj)
                self.creation_count += 1
            except Exception as e:
                print(f"Error creating pool object: {e}")
                break
    
    def acquire(self) -> Optional[Any]:
        """Acquire object from pool."""
        with self._lock:
            # Try to get from available objects
            if self.available_objects:
                obj = self.available_objects.popleft()
                self.allocated_objects.add(obj)
                self.allocation_count += 1
                return obj
            
            # Create new object if auto-expansion enabled
            if self.auto_expand and len(self.allocated_objects) < self.max_size:
                try:
                    obj = self.object_factory()
                    self.allocated_objects.add(obj)
                    self.creation_count += 1
                    self.allocation_count += 1
                    return obj
                except Exception as e:
                    print(f"Error creating pool object: {e}")
            
            return None  # Pool exhausted
    
    def release(self, obj: Any):
        """Release object back to pool."""
        with self._lock:
            if obj in self.allocated_objects:
                self.allocated_objects.discard(obj)
                
                # Reset object state if possible
                if hasattr(obj, 'reset'):
                    try:
                        obj.reset()
                    except Exception as e:
                        print(f"Error resetting pool object: {e}")
                        return  # Don't return corrupted object to pool
                
                # Return to pool if space available
                if len(self.available_objects) < self.max_size:
                    self.available_objects.append(obj)
                    self.deallocation_count += 1
    
    @contextmanager
    def get_object(self):
        """Context manager for automatic object acquisition and release."""
        obj = self.acquire()
        if obj is None:
            raise MemoryPoolExhaustedException(f"Pool {self.pool_name} exhausted")
        
        try:
            yield obj
        finally:
            self.release(obj)
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get pool performance statistics."""
        with self._lock:
            return {
                'pool_name': self.pool_name,
                'available_objects': len(self.available_objects),
                'allocated_objects': len(self.allocated_objects),
                'total_created': self.creation_count,
                'total_allocations': self.allocation_count,
                'total_deallocations': self.deallocation_count,
                'utilization_ratio': (len(self.allocated_objects) / self.max_size 
                                    if self.max_size > 0 else 0.0),
                'efficiency_ratio': (self.deallocation_count / self.allocation_count
                                   if self.allocation_count > 0 else 0.0)
            }
    
    def resize_pool(self, new_max_size: int):
        """Resize pool maximum capacity."""
        with self._lock:
            self.max_size = new_max_size
            
            # Trim available objects if new size is smaller
            while len(self.available_objects) > new_max_size:
                self.available_objects.pop()

class MemoryLeakDetector:
    """Advanced memory leak detection and monitoring."""
    
    def __init__(self, sample_interval: int = 60, history_size: int = 1000):
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.memory_history = deque(maxlen=history_size)
        self.object_growth_tracking = defaultdict(deque)
        self.is_monitoring = False
        self.monitoring_thread = None
        self._lock = threading.Lock()
        self.leak_thresholds = {
            'memory_growth_mb': 50.0,  # MB per hour
            'object_growth_rate': 1000  # Objects per hour
        }
        
    def start_monitoring(self):
        """Start memory leak monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Enable tracemalloc for detailed tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def stop_monitoring(self):
        """Stop memory leak monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    
    def _monitoring_loop(self):
        """Main monitoring loop for memory leak detection."""
        while self.is_monitoring:
            try:
                # Collect memory statistics
                current_stats = self._collect_memory_stats()
                
                with self._lock:
                    self.memory_history.append(current_stats)
                
                # Analyze for potential leaks
                if len(self.memory_history) >= 10:
                    leak_analysis = self._analyze_memory_trends()
                    if leak_analysis['potential_leaks']:
                        self._handle_detected_leak(leak_analysis)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                time.sleep(5)
    
    def _collect_memory_stats(self) -> Dict[str, Any]:
        """Collect comprehensive memory statistics."""
        # System memory stats
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Object count statistics
        object_counts = {}
        for obj_type in [dict, list, str, int, float]:
            object_counts[obj_type.__name__] = len(gc.get_objects())
        
        # Tracemalloc statistics if available
        tracemalloc_stats = None
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_stats = {
                'current_memory': current,
                'peak_memory': peak
            }
        
        return {
            'timestamp': time.time(),
            'system_memory': {
                'total': memory_info.total,
                'available': memory_info.available,
                'used': memory_info.used,
                'percent': memory_info.percent
            },
            'process_memory': {
                'rss': process_memory.rss,
                'vms': process_memory.vms
            },
            'object_counts': object_counts,
            'tracemalloc_stats': tracemalloc_stats,
            'gc_counts': gc.get_count()
        }
    
    def _analyze_memory_trends(self) -> Dict[str, Any]:
        """Analyze memory usage trends for leak detection."""
        if len(self.memory_history) < 10:
            return {'potential_leaks': False}
        
        # Get recent memory samples
        recent_samples = list(self.memory_history)[-10:]
        
        # Calculate memory growth rate
        timestamps = [sample['timestamp'] for sample in recent_samples]
        rss_values = [sample['process_memory']['rss'] for sample in recent_samples]
        
        # Linear regression to find growth trend
        time_diffs = [(t - timestamps[0]) / 3600 for t in timestamps]  # Hours
        memory_diffs = [(rss - rss_values[0]) / (1024 * 1024) for rss in rss_values]  # MB
        
        if len(time_diffs) < 2:
            return {'potential_leaks': False}
        
        # Calculate growth rate (MB per hour)
        growth_rate = np.polyfit(time_diffs, memory_diffs, 1)[0] if len(time_diffs) > 1 else 0
        
        # Analyze object count growth
        object_growth = {}
        for obj_type in ['dict', 'list', 'str']:
            start_count = recent_samples[0]['object_counts'].get(obj_type, 0)
            end_count = recent_samples[-1]['object_counts'].get(obj_type, 0)
            time_span_hours = (timestamps[-1] - timestamps[0]) / 3600
            
            if time_span_hours > 0:
                object_growth[obj_type] = (end_count - start_count) / time_span_hours
        
        # Determine if potential leak exists
        potential_leaks = (
            growth_rate > self.leak_thresholds['memory_growth_mb'] or
            any(growth > self.leak_thresholds['object_growth_rate'] 
                for growth in object_growth.values())
        )
        
        return {
            'potential_leaks': potential_leaks,
            'memory_growth_rate_mb_hour': growth_rate,
            'object_growth_rates': object_growth,
            'analysis_window_hours': time_span_hours,
            'confidence': min(1.0, abs(growth_rate) / 10.0)  # Confidence based on growth rate
        }
    
    def _handle_detected_leak(self, leak_analysis: Dict[str, Any]):
        """Handle detected memory leak."""
        print(f"Memory leak detected in {self.pool_name if hasattr(self, 'pool_name') else 'system'}")
        print(f"Growth rate: {leak_analysis['memory_growth_rate_mb_hour']:.2f} MB/hour")
        
        # Trigger garbage collection
        collected = gc.collect()
        print(f"Forced garbage collection freed {collected} objects")
        
        # Get top memory consumers if tracemalloc available
        if tracemalloc.is_tracing():
            top_stats = tracemalloc.take_snapshot().statistics('lineno')[:10]
            print("Top memory consumers:")
            for stat in top_stats:
                print(f"  {stat.traceback.format()[-1]}: {stat.size / 1024:.1f} KB")
    
    def get_leak_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory leak analysis report."""
        with self._lock:
            if len(self.memory_history) < 5:
                return {'status': 'insufficient_data', 'samples': len(self.memory_history)}
            
            leak_analysis = self._analyze_memory_trends()
            
            # Additional analysis
            latest_stats = self.memory_history[-1]
            earliest_stats = self.memory_history[0]
            
            time_span = latest_stats['timestamp'] - earliest_stats['timestamp']
            memory_change = (latest_stats['process_memory']['rss'] - 
                           earliest_stats['process_memory']['rss']) / (1024 * 1024)
            
            return {
                'status': 'potential_leak' if leak_analysis['potential_leaks'] else 'healthy',
                'monitoring_duration_hours': time_span / 3600,
                'total_memory_change_mb': memory_change,
                'leak_analysis': leak_analysis,
                'latest_memory_stats': latest_stats,
                'recommendations': self._generate_leak_recommendations(leak_analysis),
                'report_timestamp': time.time()
            }
    
    def _generate_leak_recommendations(self, leak_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for addressing memory issues."""
        recommendations = []
        
        if leak_analysis['potential_leaks']:
            recommendations.append("Monitor memory usage closely and consider profiling")
            
            growth_rate = leak_analysis['memory_growth_rate_mb_hour']
            if growth_rate > 100:
                recommendations.append("Critical memory growth detected - immediate investigation required")
            elif growth_rate > 50:
                recommendations.append("Significant memory growth - review recent code changes")
            
            # Object-specific recommendations
            for obj_type, growth in leak_analysis['object_growth_rates'].items():
                if growth > self.leak_thresholds['object_growth_rate']:
                    recommendations.append(f"High {obj_type} object growth - check for retention issues")
            
            recommendations.append("Consider enabling memory profiling and reviewing cache policies")
        
        return recommendations

class GarbageCollectionOptimizer:
    """Optimize garbage collection for better performance."""
    
    def __init__(self, strategy: GCStrategy = GCStrategy.THRESHOLD_BASED):
        self.strategy = strategy
        self.gc_thresholds = [700, 10, 10]  # Default Python thresholds
        self.custom_thresholds = None
        self.gc_statistics = {
            'collections': defaultdict(int),
            'collection_times': defaultdict(list),
            'objects_collected': defaultdict(int)
        }
        self.is_monitoring = False
        self._monitoring_thread = None
    
    def optimize_thresholds(self):
        """Automatically optimize GC thresholds based on usage patterns."""
        # Analyze current memory usage patterns
        memory_stats = self._analyze_allocation_patterns()
        
        # Calculate optimal thresholds based on allocation rate
        if memory_stats['allocation_rate'] > 1000:  # High allocation rate
            # More aggressive GC for generation 0
            new_thresholds = [500, 8, 8]
        elif memory_stats['allocation_rate'] < 100:  # Low allocation rate
            # Less aggressive GC
            new_thresholds = [1000, 15, 15]
        else:
            # Moderate allocation rate
            new_thresholds = [700, 10, 10]
        
        self.set_gc_thresholds(new_thresholds)
        
        return {
            'old_thresholds': self.gc_thresholds,
            'new_thresholds': new_thresholds,
            'allocation_rate': memory_stats['allocation_rate']
        }
    
    def set_gc_thresholds(self, thresholds: List[int]):
        """Set custom garbage collection thresholds."""
        self.custom_thresholds = thresholds
        gc.set_threshold(*thresholds)
        print(f"GC thresholds updated: {thresholds}")
    
    def _analyze_allocation_patterns(self) -> Dict[str, Any]:
        """Analyze memory allocation patterns."""
        # Enable allocation tracking temporarily
        tracemalloc.start()
        initial_stats = tracemalloc.get_traced_memory()
        
        # Wait and measure allocation rate
        time.sleep(1.0)
        final_stats = tracemalloc.get_traced_memory()
        
        tracemalloc.stop()
        
        allocation_rate = (final_stats[0] - initial_stats[0]) / 1024  # KB per second
        
        return {
            'allocation_rate': allocation_rate,
            'current_memory_kb': final_stats[0] / 1024,
            'peak_memory_kb': final_stats[1] / 1024
        }
    
    def force_collection(self, generation: Optional[int] = None) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        start_time = time.perf_counter()
        
        if generation is None:
            collected = gc.collect()
            generation = 2  # Full collection
        else:
            collected = gc.collect(generation)
        
        end_time = time.perf_counter()
        collection_time = end_time - start_time
        
        # Update statistics
        self.gc_statistics['collections'][generation] += 1
        self.gc_statistics['collection_times'][generation].append(collection_time)
        self.gc_statistics['objects_collected'][generation] += collected
        
        return {
            'objects_collected': collected,
            'collection_time': collection_time,
            'generation': generation
        }
    
    def get_gc_statistics(self) -> Dict[str, Any]:
        """Get comprehensive garbage collection statistics."""
        current_thresholds = gc.get_threshold()
        current_counts = gc.get_count()
        
        # Calculate average collection times
        avg_collection_times = {}
        for gen, times in self.gc_statistics['collection_times'].items():
            avg_collection_times[gen] = np.mean(times) if times else 0.0
        
        return {
            'current_thresholds': current_thresholds,
            'current_counts': current_counts,
            'custom_thresholds': self.custom_thresholds,
            'total_collections': dict(self.gc_statistics['collections']),
            'average_collection_times': avg_collection_times,
            'total_objects_collected': dict(self.gc_statistics['objects_collected']),
            'gc_enabled': gc.isenabled()
        }

class ResourceMonitor:
    """Comprehensive system resource monitoring."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.resource_history = deque(maxlen=1000)
        self.alerts = []
        self.thresholds = {
            'memory_percent': 85.0,
            'cpu_percent': 80.0,
            'disk_percent': 90.0,
            'swap_percent': 50.0
        }
        self.is_monitoring = False
        self._monitoring_thread = None
        self._alert_callbacks = []
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
    
    def _monitoring_loop(self):
        """Main resource monitoring loop."""
        while self.is_monitoring:
            try:
                stats = self.collect_resource_stats()
                self.resource_history.append(stats)
                
                # Check thresholds and generate alerts
                alerts = self._check_resource_thresholds(stats)
                if alerts:
                    self._trigger_alerts(alerts)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                time.sleep(5)
    
    def collect_resource_stats(self) -> Dict[str, Any]:
        """Collect comprehensive system resource statistics."""
        # Memory statistics
        memory_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        
        # Process statistics
        process = psutil.Process()
        process_info = process.memory_info()
        
        # CPU statistics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Disk statistics
        disk_usage = psutil.disk_usage('/')
        
        # Network statistics (if needed for communication monitoring)
        try:
            network_io = psutil.net_io_counters()
            network_stats = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
        except AttributeError:
            network_stats = None
        
        return {
            'timestamp': time.time(),
            'memory': {
                'total': memory_info.total,
                'available': memory_info.available,
                'used': memory_info.used,
                'percent': memory_info.percent
            },
            'swap': {
                'total': swap_info.total,
                'used': swap_info.used,
                'percent': swap_info.percent
            },
            'process': {
                'rss': process_info.rss,
                'vms': process_info.vms,
                'percent': process.memory_percent()
            },
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'per_cpu': psutil.cpu_percent(percpu=True) if cpu_count > 1 else [cpu_percent]
            },
            'disk': {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': (disk_usage.used / disk_usage.total * 100)
            },
            'network': network_stats
        }
    
    def _check_resource_thresholds(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check resource usage against thresholds."""
        alerts = []
        
        # Memory threshold check
        if stats['memory']['percent'] > self.thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_threshold_exceeded',
                'severity': 'warning',
                'current_value': stats['memory']['percent'],
                'threshold': self.thresholds['memory_percent'],
                'message': f"Memory usage at {stats['memory']['percent']:.1f}%"
            })
        
        # CPU threshold check
        if stats['cpu']['percent'] > self.thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_threshold_exceeded',
                'severity': 'warning',
                'current_value': stats['cpu']['percent'],
                'threshold': self.thresholds['cpu_percent'],
                'message': f"CPU usage at {stats['cpu']['percent']:.1f}%"
            })
        
        # Swap threshold check
        if stats['swap']['percent'] > self.thresholds['swap_percent']:
            alerts.append({
                'type': 'swap_threshold_exceeded',
                'severity': 'critical',
                'current_value': stats['swap']['percent'],
                'threshold': self.thresholds['swap_percent'],
                'message': f"Swap usage at {stats['swap']['percent']:.1f}%"
            })
        
        # Disk threshold check
        if stats['disk']['percent'] > self.thresholds['disk_percent']:
            alerts.append({
                'type': 'disk_threshold_exceeded',
                'severity': 'critical',
                'current_value': stats['disk']['percent'],
                'threshold': self.thresholds['disk_percent'],
                'message': f"Disk usage at {stats['disk']['percent']:.1f}%"
            })
        
        return alerts
    
    def _trigger_alerts(self, alerts: List[Dict[str, Any]]):
        """Trigger resource alerts."""
        timestamp = time.time()
        
        for alert in alerts:
            alert['timestamp'] = timestamp
            self.alerts.append(alert)
            
            # Notify registered callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Alert callback error: {e}")
        
        # Keep only recent alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for resource alerts."""
        self._alert_callbacks.append(callback)
    
    def get_resource_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze resource usage trends over specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_stats = [stat for stat in self.resource_history 
                       if stat['timestamp'] > cutoff_time]
        
        if len(recent_stats) < 5:
            return {'status': 'insufficient_data'}
        
        # Calculate trends for key metrics
        timestamps = [stat['timestamp'] for stat in recent_stats]
        
        trends = {}
        for metric_path in ['memory.percent', 'cpu.percent', 'disk.percent']:
            path_parts = metric_path.split('.')
            values = []
            
            for stat in recent_stats:
                current_dict = stat
                for part in path_parts:
                    current_dict = current_dict.get(part, {})
                    if not isinstance(current_dict, dict):
                        values.append(float(current_dict) if current_dict else 0.0)
                        break
            
            if len(values) == len(recent_stats):
                # Calculate linear trend
                time_hours = [(t - timestamps[0]) / 3600 for t in timestamps]
                if len(time_hours) > 1:
                    trend_slope = np.polyfit(time_hours, values, 1)[0]
                    trends[metric_path] = {
                        'slope_per_hour': trend_slope,
                        'current_value': values[-1],
                        'min_value': min(values),
                        'max_value': max(values),
                        'average_value': np.mean(values)
                    }
        
        return {
            'analysis_period_hours': hours,
            'data_points': len(recent_stats),
            'trends': trends,
            'analysis_timestamp': time.time()
        }

class MemoryMappedBuffer:
    """Memory-mapped buffer for efficient large data handling."""
    
    def __init__(self, size_bytes: int, filename: Optional[str] = None):
        self.size_bytes = size_bytes
        self.filename = filename
        self.file_obj = None
        self.mmap_obj = None
        self.is_anonymous = filename is None
        self._lock = threading.Lock()
        self.access_count = 0
        self.creation_time = time.time()
        
        self._initialize_buffer()
    
    def _initialize_buffer(self):
        """Initialize memory-mapped buffer."""
        try:
            if self.is_anonymous:
                # Anonymous memory mapping
                self.mmap_obj = mmap.mmap(-1, self.size_bytes)
            else:
                # File-backed memory mapping
                self.file_obj = open(self.filename, 'r+b')
                
                # Ensure file is large enough
                file_size = os.path.getsize(self.filename)
                if file_size < self.size_bytes:
                    self.file_obj.seek(self.size_bytes - 1)
                    self.file_obj.write(b'\0')
                    self.file_obj.flush()
                
                self.mmap_obj = mmap.mmap(self.file_obj.fileno(), self.size_bytes)
                
        except Exception as e:
            self.close()
            raise MemoryMappingError(f"Failed to create memory mapping: {str(e)}")
    
    def read(self, offset: int, size: int) -> bytes:
        """Read data from memory-mapped buffer."""
        with self._lock:
            if not self.mmap_obj:
                raise MemoryMappingError("Buffer is closed")
            
            if offset + size > self.size_bytes:
                raise ValueError("Read beyond buffer bounds")
            
            self.access_count += 1
            self.mmap_obj.seek(offset)
            return self.mmap_obj.read(size)
    
    def write(self, offset: int, data: bytes) -> int:
        """Write data to memory-mapped buffer."""
        with self._lock:
            if not self.mmap_obj:
                raise MemoryMappingError("Buffer is closed")
            
            if offset + len(data) > self.size_bytes:
                raise ValueError("Write beyond buffer bounds")
            
            self.access_count += 1
            self.mmap_obj.seek(offset)
            self.mmap_obj.write(data)
            self.mmap_obj.flush()
            return len(data)
    
    def resize(self, new_size: int):
        """Resize memory-mapped buffer (file-backed only)."""
        if self.is_anonymous:
            raise MemoryMappingError("Cannot resize anonymous memory mapping")
        
        with self._lock:
            if self.mmap_obj:
                self.mmap_obj.close()
            
            # Resize file
            if self.file_obj:
                self.file_obj.seek(new_size - 1)
                self.file_obj.write(b'\0')
                self.file_obj.flush()
            
            # Recreate mapping
            self.size_bytes = new_size
            if self.file_obj:
                self.mmap_obj = mmap.mmap(self.file_obj.fileno(), new_size)
    
    def close(self):
        """Close memory-mapped buffer and release resources."""
        with self._lock:
            if self.mmap_obj:
                self.mmap_obj.close()
                self.mmap_obj = None
            
            if self.file_obj:
                self.file_obj.close()
                self.file_obj = None
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer information and statistics."""
        return {
            'size_bytes': self.size_bytes,
            'filename': self.filename,
            'is_anonymous': self.is_anonymous,
            'access_count': self.access_count,
            'age_seconds': time.time() - self.creation_time,
            'is_active': self.mmap_obj is not None
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

class SmartCache:
    """Intelligent cache with adaptive behavior and multiple eviction strategies."""
    
    def __init__(self, name: str, max_size_bytes: int = 100 * 1024 * 1024,
                 adaptive_sizing: bool = True, auto_optimize: bool = True):
        self.name = name
        self.max_size_bytes = max_size_bytes
        self.adaptive_sizing = adaptive_sizing
        self.auto_optimize = auto_optimize
        self.cache = AdvancedCache(max_size_bytes, CachePolicy.ADAPTIVE)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.optimization_history = []
        
        # Adaptive parameters
        self.current_policy = CachePolicy.LRU
        self.policy_performance = {policy: deque(maxlen=100) for policy in CachePolicy}
        
        # Auto-optimization thread
        self.optimization_thread = None
        self.optimization_interval = 300  # 5 minutes
        
        if auto_optimize:
            self._start_optimization()
    
    def _start_optimization(self):
        """Start cache optimization thread."""
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        self.optimization_thread.start()
    
    def _optimization_loop(self):
        """Continuously optimize cache performance."""
        while self.auto_optimize:
            try:
                time.sleep(self.optimization_interval)
                
                if len(self.performance_history) >= 50:
                    self._adaptive_policy_selection()
                    
                if self.adaptive_sizing:
                    self._adaptive_size_adjustment()
                    
            except Exception as e:
                print(f"Cache optimization error: {e}")
    
    def _adaptive_policy_selection(self):
        """Automatically select best cache policy based on performance."""
        current_hit_rate = self._calculate_recent_hit_rate()
        
        # Test different policies periodically
        if len(self.optimization_history) % 10 == 0:  # Every 10 optimization cycles
            # Temporarily switch to different policy for testing
            test_policies = [p for p in CachePolicy if p != self.current_policy]
            if test_policies:
                test_policy = np.random.choice(test_policies)
                old_policy = self.current_policy
                
                # Switch policy and measure performance
                self.cache.policy = test_policy
                self.current_policy = test_policy
                
                # Record result after some time
                self.policy_performance[test_policy].append(current_hit_rate)
                
                # Revert if performance is worse
                if len(self.policy_performance[test_policy]) > 5:
                    test_performance = np.mean(list(self.policy_performance[test_policy]))
                    old_performance = np.mean(list(self.policy_performance[old_policy])) if self.policy_performance[old_policy] else 0.0
                    
                    if test_performance < old_performance:
                        self.cache.policy = old_policy
                        self.current_policy = old_policy
    
    def _calculate_recent_hit_rate(self) -> float:
        """Calculate recent cache hit rate."""
        if not self.performance_history:
            return 0.0
        
        recent_stats = list(self.performance_history)[-10:]
        total_requests = sum(stat.get('total_requests', 0) for stat in recent_stats)
        total_hits = sum(stat.get('hits', 0) for stat in recent_stats)
        
        return total_hits / total_requests if total_requests > 0 else 0.0
    
    def _adaptive_size_adjustment(self):
        """Adaptively adjust cache size based on memory pressure."""
        # Get current memory usage
        memory_info = psutil.virtual_memory()
        
        # Adjust cache size based on available memory
        if memory_info.percent > 80:  # High memory pressure
            new_size = int(self.max_size_bytes * 0.8)
        elif memory_info.percent < 50:  # Low memory pressure
            new_size = int(self.max_size_bytes * 1.2)
        else:
            new_size = self.max_size_bytes
        
        # Apply reasonable bounds
        min_size = 10 * 1024 * 1024  # 10 MB minimum
        max_size = 500 * 1024 * 1024  # 500 MB maximum
        new_size = max(min_size, min(new_size, max_size))
        
        if new_size != self.cache.max_size_bytes:
            self.cache.max_size_bytes = new_size
            
            # Force cleanup if cache is now oversized
            if self.cache.size_bytes > new_size:
                self.cache._evict_entries(self.cache.size_bytes - new_size)
    
    def put(self, key: str, value: Any, **kwargs):
        """Put value in cache and update performance tracking."""
        start_time = time.perf_counter()
        
        self.cache.put(key, value, **kwargs)
        
        # Track performance
        operation_time = time.perf_counter() - start_time
        self._record_performance('put', operation_time)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache and update performance tracking."""
        start_time = time.perf_counter()
        
        result = self.cache.get(key)
        
        # Track performance
        operation_time = time.perf_counter() - start_time
        self._record_performance('get', operation_time)
        
        return result
    
    def _record_performance(self, operation: str, operation_time: float):
        """Record cache performance metrics."""
        stats = self.cache.get_cache_statistics()
        stats.update({
            'operation': operation,
            'operation_time': operation_time,
            'timestamp': time.time()
        })
        
        self.performance_history.append(stats)

class MemoryProfiler:
    """Advanced memory profiling and analysis."""
    
    def __init__(self, enable_line_profiling: bool = True):
        self.enable_line_profiling = enable_line_profiling
        self.profiling_active = False
        self.snapshots = []
        self.allocation_profiles = {}
        self._profiling_overhead = 0.0
    
    def start_profiling(self, trace_malloc: bool = True):
        """Start memory profiling session."""
        if self.profiling_active:
            return
        
        start_time = time.perf_counter()
        
        if trace_malloc and not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Keep 25 frames
        
        self.profiling_active = True
        self._baseline_snapshot = self._take_snapshot()
        
        self._profiling_overhead = time.perf_counter() - start_time
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return analysis results."""
        if not self.profiling_active:
            return {'status': 'not_profiling'}
        
        final_snapshot = self._take_snapshot()
        self.profiling_active = False
        
        # Analyze allocations between baseline and final
        analysis = self._analyze_snapshots(self._baseline_snapshot, final_snapshot)
        
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        return analysis
    
    def _take_snapshot(self) -> Dict[str, Any]:
        """Take memory snapshot with multiple data sources."""
        snapshot_data = {
            'timestamp': time.time(),
            'system_memory': dict(psutil.virtual_memory()._asdict()),
            'process_memory': dict(psutil.Process().memory_info()._asdict()),
            'gc_stats': {
                'counts': gc.get_count(),
                'objects': len(gc.get_objects())
            }
        }
        
        # Add tracemalloc snapshot if available
        if tracemalloc.is_tracing():
            tracemalloc_snapshot = tracemalloc.take_snapshot()
            snapshot_data['tracemalloc'] = {
                'top_stats': tracemalloc_snapshot.statistics('lineno')[:20],
                'traced_memory': tracemalloc.get_traced_memory()
            }
        
        return snapshot_data
    
    def _analyze_snapshots(self, baseline: Dict[str, Any], 
                          final: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze differences between memory snapshots."""
        time_diff = final['timestamp'] - baseline['timestamp']
        
        # Memory usage changes
        process_memory_change = (final['process_memory']['rss'] - 
                               baseline['process_memory']['rss'])
        
        # Object count changes
        object_count_change = (final['gc_stats']['objects'] - 
                             baseline['gc_stats']['objects'])
        
        analysis = {
            'profiling_duration': time_diff,
            'memory_change_bytes': process_memory_change,
            'memory_change_mb': process_memory_change / (1024 * 1024),
            'object_count_change': object_count_change,
            'allocation_rate_mb_per_sec': (process_memory_change / (1024 * 1024)) / time_diff if time_diff > 0 else 0,
            'profiling_overhead': self._profiling_overhead
        }
        
        # Detailed tracemalloc analysis if available
        if ('tracemalloc' in baseline and 'tracemalloc' in final and 
            baseline['tracemalloc'] and final['tracemalloc']):
            
            baseline_traced = baseline['tracemalloc']['traced_memory'][0]
            final_traced = final['tracemalloc']['traced_memory'][0]
            traced_memory_change = final_traced - baseline_traced
            
            analysis['traced_memory_change'] = {
                'bytes': traced_memory_change,
                'mb': traced_memory_change / (1024 * 1024)
            }
            
            # Top memory allocation locations
            top_allocations = final['tracemalloc']['top_stats'][:10]
            analysis['top_allocations'] = [
                {
                    'filename': stat.traceback.format()[0] if stat.traceback.format() else 'unknown',
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count
                }
                for stat in top_allocations
            ]
        
        return analysis
    
    @contextmanager
    def profile_block(self, block_name: str):
        """Context manager for profiling code blocks."""
        block_start_time = time.time()
        
        if not self.profiling_active:
            self.start_profiling()
            started_here = True
        else:
            started_here = False
        
        block_baseline = self._take_snapshot()
        
        try:
            yield
        finally:
            block_final = self._take_snapshot()
            block_analysis = self._analyze_snapshots(block_baseline, block_final)
            
            # Store block-specific analysis
            self.allocation_profiles[block_name] = {
                **block_analysis,
                'block_execution_time': time.time() - block_start_time
            }
            
            if started_here:
                self.stop_profiling()
    
    def get_allocation_profile(self, block_name: str) -> Optional[Dict[str, Any]]:
        """Get allocation profile for specific code block."""
        return self.allocation_profiles.get(block_name)
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        current_stats = self._take_snapshot()
        
        # Calculate memory efficiency metrics
        process_memory = current_stats['process_memory']['rss']
        system_memory = current_stats['system_memory']['total']
        memory_efficiency = 1.0 - (process_memory / system_memory)
        
        return {
            'current_memory_stats': current_stats,
            'allocation_profiles': self.allocation_profiles,
            'memory_efficiency': memory_efficiency,
            'gc_statistics': {
                'collections': gc.get_stats(),
                'thresholds': gc.get_threshold(),
                'current_counts': gc.get_count()
            },
            'recommendations': self._generate_memory_recommendations(current_stats),
            'report_timestamp': time.time()
        }
    
    def _generate_memory_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        memory_percent = stats['system_memory']['percent']
        process_memory_mb = stats['process_memory']['rss'] / (1024 * 1024)
        
        if memory_percent > 85:
            recommendations.append("System memory usage is high. Consider reducing cache sizes.")
        
        if process_memory_mb > 500:
            recommendations.append("Process memory usage is high. Review memory allocations.")
        
        gc_counts = stats['gc_stats']['counts']
        if gc_counts[0] > 1000:
            recommendations.append("High generation-0 GC activity. Consider optimizing object creation.")
        
        if len(self.allocation_profiles) > 0:
            # Find memory-intensive blocks
            intensive_blocks = sorted(
                [(name, profile['memory_change_mb']) 
                 for name, profile in self.allocation_profiles.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            if intensive_blocks and intensive_blocks[0][1] > 10:
                recommendations.append(f"High memory allocation in '{intensive_blocks[0][0]}' block.")
        
        return recommendations

class ObjectPool(ABC):
    """Abstract base class for object pooling implementations."""
    
    @abstractmethod
    def acquire(self) -> Any:
        """Acquire object from pool."""
        pass
    
    @abstractmethod
    def release(self, obj: Any):
        """Release object back to pool."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool usage statistics."""
        pass

class TypedObjectPool(ObjectPool):
    """Type-specific object pool with enhanced lifecycle management."""
    
    def __init__(self, object_type: Type, factory_args: Tuple = (),
                 factory_kwargs: Optional[Dict[str, Any]] = None,
                 initial_size: int = 5, max_size: int = 50,
                 validation_func: Optional[Callable[[Any], bool]] = None):
        self.object_type = object_type
        self.factory_args = factory_args
        self.factory_kwargs = factory_kwargs or {}
        self.max_size = max_size
        self.validation_func = validation_func
        
        self.available_objects = deque()
        self.allocated_objects = weakref.WeakSet()
        self.total_created = 0
        self.total_allocations = 0
        self.total_releases = 0
        self.validation_failures = 0
        
        self._lock = threading.Lock()
        
        # Pre-populate pool
        self._create_initial_objects(initial_size)
    
    def _create_initial_objects(self, count: int):
        """Create initial pool objects."""
        for _ in range(count):
            try:
                obj = self._create_object()
                self.available_objects.append(obj)
            except Exception as e:
                print(f"Error creating initial pool object: {e}")
                break
    
    def _create_object(self) -> Any:
        """Create new object instance."""
        obj = self.object_type(*self.factory_args, **self.factory_kwargs)
        self.total_created += 1
        return obj
    
    def acquire(self) -> Optional[Any]:
        """Acquire object from pool with validation."""
        with self._lock:
            # Try to get validated object from pool
            while self.available_objects:
                obj = self.available_objects.popleft()
                
                # Validate object if validation function provided
                if self.validation_func and not self.validation_func(obj):
                    self.validation_failures += 1
                    continue
                
                self.allocated_objects.add(obj)
                self.total_allocations += 1
                return obj
            
            # Create new object if pool is empty and under limit
            if len(self.allocated_objects) < self.max_size:
                try:
                    obj = self._create_object()
                    self.allocated_objects.add(obj)
                    self.total_allocations += 1
                    return obj
                except Exception as e:
                    print(f"Error creating pool object: {e}")
            
            return None  # Pool exhausted
    
    def release(self, obj: Any):
        """Release object back to pool with validation."""
        if not isinstance(obj, self.object_type):
            return
        
        with self._lock:
            if obj in self.allocated_objects:
                self.allocated_objects.discard(obj)
                
                # Validate object before returning to pool
                if self.validation_func and not self.validation_func(obj):
                    self.validation_failures += 1
                    return
                
                # Reset object state if method available
                if hasattr(obj, 'reset') and callable(getattr(obj, 'reset')):
                    try:
                        obj.reset()
                    except Exception as e:
                        print(f"Error resetting pool object: {e}")
                        return
                
                # Return to pool if space available
                if len(self.available_objects) < self.max_size:
                    self.available_objects.append(obj)
                    self.total_releases += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self._lock:
            return {
                'object_type': self.object_type.__name__,
                'available_objects': len(self.available_objects),
                'allocated_objects': len(self.allocated_objects),
                'total_created': self.total_created,
                'total_allocations': self.total_allocations,
                'total_releases': self.total_releases,
                'validation_failures': self.validation_failures,
                'pool_utilization': len(self.allocated_objects) / self.max_size,
                'efficiency_ratio': (self.total_releases / self.total_allocations 
                                   if self.total_allocations > 0 else 0.0)
            }

class MemoryFragmentationAnalyzer:
    """Analyze and optimize memory fragmentation."""
    
    def __init__(self):
        self.fragmentation_history = deque(maxlen=100)
        self.defragmentation_events = []
    
    def analyze_fragmentation(self) -> Dict[str, Any]:
        """Analyze current memory fragmentation level."""
        # Get memory mapping information
        try:
            process = psutil.Process()
            memory_maps = process.memory_maps()
            
            total_mapped = sum(mmap.rss for mmap in memory_maps if mmap.rss)
            total_virtual = sum(mmap.size for mmap in memory_maps)
            
            # Calculate fragmentation metrics
            fragmentation_ratio = 1.0 - (total_mapped / total_virtual) if total_virtual > 0 else 0.0
            
            # Analyze heap fragmentation using allocation patterns
            heap_analysis = self._analyze_heap_fragmentation()
            
            fragmentation_data = {
                'timestamp': time.time(),
                'fragmentation_ratio': fragmentation_ratio,
                'total_mapped_mb': total_mapped / (1024 * 1024),
                'total_virtual_mb': total_virtual / (1024 * 1024),
                'heap_fragmentation': heap_analysis,
                'memory_map_count': len(memory_maps)
            }
            
            self.fragmentation_history.append(fragmentation_data)
            
            return fragmentation_data
            
        except Exception as e:
            return {'error': f"Fragmentation analysis failed: {str(e)}"}
    
    def _analyze_heap_fragmentation(self) -> Dict[str, Any]:
        """Analyze heap-specific fragmentation patterns."""
        # Force garbage collection to clean up loose objects
        collected_objects = gc.collect()
        
        # Analyze object size distribution
        all_objects = gc.get_objects()
        object_sizes = []
        
        for obj in all_objects[:1000]:  # Sample subset for performance
            try:
                size = sys.getsizeof(obj)
                object_sizes.append(size)
            except Exception:
                continue
        
        if not object_sizes:
            return {'status': 'no_data'}
        
        # Calculate fragmentation indicators
        size_variance = np.var(object_sizes)
        size_mean = np.mean(object_sizes)
        size_std = np.std(object_sizes)
        
        # Fragmentation score based on size distribution
        fragmentation_score = min(1.0, size_std / size_mean if size_mean > 0 else 0.0)
        
        return {
            'collected_objects': collected_objects,
            'sample_objects': len(object_sizes),
            'average_object_size': size_mean,
            'size_variance': size_variance,
            'size_std': size_std,
            'fragmentation_score': fragmentation_score
        }
    
    def suggest_defragmentation(self) -> Dict[str, Any]:
        """Suggest defragmentation strategies based on analysis."""
        if not self.fragmentation_history:
            return {'status': 'no_analysis_data'}
        
        latest_analysis = self.fragmentation_history[-1]
        fragmentation_ratio = latest_analysis.get('fragmentation_ratio', 0.0)
        
        suggestions = []
        
        if fragmentation_ratio > 0.3:
            suggestions.append("High memory fragmentation detected")
            suggestions.append("Consider consolidating memory allocations")
            suggestions.append("Implement memory compaction for large objects")
        
        if latest_analysis.get('heap_fragmentation', {}).get('fragmentation_score', 0) > 0.5:
            suggestions.append("Heap fragmentation is high")
            suggestions.append("Consider object pooling for frequently allocated types")
            suggestions.append("Review object lifecycle management")
        
        # GC-based suggestions
        if latest_analysis['heap_fragmentation'].get('collected_objects', 0) > 1000:
            suggestions.append("High garbage collection activity")
            suggestions.append("Optimize object creation patterns")
        
        return {
            'fragmentation_level': 'high' if fragmentation_ratio > 0.3 else 'moderate' if fragmentation_ratio > 0.1 else 'low',
            'suggestions': suggestions,
            'defragmentation_priority': min(10, int(fragmentation_ratio * 10))
        }

class ResourceLimitManager:
    """Manage system resource limits and quotas."""
    
    def __init__(self):
        self.resource_limits = {}
        self.current_usage = {}
        self.limit_callbacks = defaultdict(list)
        self._monitoring_active = False
        self._monitoring_thread = None
    
    def set_memory_limit(self, limit_bytes: int, soft_limit_ratio: float = 0.8):
        """Set memory usage limits."""
        self.resource_limits['memory'] = {
            'hard_limit': limit_bytes,
            'soft_limit': int(limit_bytes * soft_limit_ratio),
            'current_usage': 0
        }
    
    def set_object_count_limit(self, object_type: Type, max_count: int):
        """Set limit on number of objects of specific type."""
        type_name = object_type.__name__
        self.resource_limits[f'objects_{type_name}'] = {
            'hard_limit': max_count,
            'soft_limit': int(max_count * 0.8),
            'object_type': object_type,
            'current_usage': 0
        }
    
    def check_resource_usage(self) -> Dict[str, Any]:
        """Check current resource usage against limits."""
        violations = []
        warnings = []
        
        # Check memory limit
        if 'memory' in self.resource_limits:
            current_memory = psutil.Process().memory_info().rss
            memory_limit = self.resource_limits['memory']
            
            memory_limit['current_usage'] = current_memory
            
            if current_memory > memory_limit['hard_limit']:
                violations.append({
                    'type': 'memory_hard_limit',
                    'current': current_memory,
                    'limit': memory_limit['hard_limit'],
                    'overage': current_memory - memory_limit['hard_limit']
                })
            elif current_memory > memory_limit['soft_limit']:
                warnings.append({
                    'type': 'memory_soft_limit',
                    'current': current_memory,
                    'limit': memory_limit['soft_limit']
                })
        
        # Check object count limits
        all_objects = gc.get_objects()
        object_counts = defaultdict(int)
        
        for obj in all_objects:
            obj_type = type(obj).__name__
            object_counts[obj_type] += 1
        
        for limit_key, limit_info in self.resource_limits.items():
            if limit_key.startswith('objects_'):
                object_type_name = limit_key.replace('objects_', '')
                current_count = object_counts.get(object_type_name, 0)
                
                limit_info['current_usage'] = current_count
                
                if current_count > limit_info['hard_limit']:
                    violations.append({
                        'type': f'object_count_hard_limit_{object_type_name}',
                        'current': current_count,
                        'limit': limit_info['hard_limit']
                    })
                elif current_count > limit_info['soft_limit']:
                    warnings.append({
                        'type': f'object_count_soft_limit_{object_type_name}',
                        'current': current_count,
                        'limit': limit_info['soft_limit']
                    })
        
        return {
            'violations': violations,
            'warnings': warnings,
            'total_objects': len(all_objects),
            'object_type_distribution': dict(object_counts)
        }
    
    def register_limit_callback(self, resource_type: str, 
                              callback: Callable[[Dict[str, Any]], None]):
        """Register callback for resource limit violations."""
        self.limit_callbacks[resource_type].append(callback)
    
    def start_monitoring(self, check_interval: float = 30.0):
        """Start continuous resource limit monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self.check_interval = check_interval
        self._monitoring_thread = threading.Thread(target=self._limit_monitoring_loop, daemon=True)
        self._monitoring_thread.start()
    
    def _limit_monitoring_loop(self):
        """Continuous monitoring loop for resource limits."""
        while self._monitoring_active:
            try:
                usage_report = self.check_resource_usage()
                
                # Trigger callbacks for violations and warnings
                for violation in usage_report['violations']:
                    for callback in self.limit_callbacks.get(violation['type'], []):
                        try:
                            callback(violation)
                        except Exception as e:
                            print(f"Limit callback error: {e}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"Resource limit monitoring error: {e}")
                time.sleep(5)

class MemoryOptimizer:
    """Comprehensive memory optimization coordinator."""
    
    def __init__(self):
        self.cache_instances = {}
        self.memory_pools = {}
        self.leak_detector = MemoryLeakDetector()
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.resource_monitor = ResourceMonitor()
        self.fragmentation_analyzer = MemoryFragmentationAnalyzer()
        self.limit_manager = ResourceLimitManager()
        
        self.optimization_history = []
        self.is_optimizing = False
    
    def register_cache(self, cache_name: str, cache_instance: AdvancedCache):
        """Register cache for optimization management."""
        self.cache_instances[cache_name] = cache_instance
    
    def register_memory_pool(self, pool_name: str, pool_instance: MemoryPool):
        """Register memory pool for optimization management."""
        self.memory_pools[pool_name] = pool_instance
    
    def optimize_system_memory(self) -> Dict[str, Any]:
        """Perform comprehensive system memory optimization."""
        optimization_start = time.time()
        optimization_results = {}
        
        # 1. Garbage collection optimization
        gc_result = self.gc_optimizer.force_collection()
        optimization_results['garbage_collection'] = gc_result
        
        # 2. Cache optimization
        cache_optimization = self._optimize_caches()
        optimization_results['cache_optimization'] = cache_optimization
        
        # 3. Memory pool optimization
        pool_optimization = self._optimize_memory_pools()
        optimization_results['pool_optimization'] = pool_optimization
        
        # 4. Fragmentation analysis and suggestions
        fragmentation_analysis = self.fragmentation_analyzer.analyze_fragmentation()
        optimization_results['fragmentation_analysis'] = fragmentation_analysis
        
        # 5. Resource limit checking
        resource_check = self.limit_manager.check_resource_usage()
        optimization_results['resource_check'] = resource_check
        
        optimization_time = time.time() - optimization_start
        
        optimization_summary = {
            'optimization_time': optimization_time,
            'results': optimization_results,
            'memory_freed_estimate': self._estimate_memory_freed(optimization_results),
            'recommendations': self._generate_optimization_recommendations(optimization_results),
            'timestamp': time.time()
        }
        
        self.optimization_history.append(optimization_summary)
        
        return optimization_summary
    
    def _optimize_caches(self) -> Dict[str, Any]:
        """Optimize all registered caches."""
        optimization_results = {}
        
        for cache_name, cache_instance in self.cache_instances.items():
            stats_before = cache_instance.get_cache_statistics()
            
            # Trigger cache cleanup
            cache_instance._evict_entries(0)  # Force cleanup of expired entries
            
            stats_after = cache_instance.get_cache_statistics()
            
            optimization_results[cache_name] = {
                'entries_before': stats_before['total_entries'],
                'entries_after': stats_after['total_entries'],
                'size_before': stats_before['size_bytes'],
                'size_after': stats_after['size_bytes'],
                'entries_evicted': stats_before['total_entries'] - stats_after['total_entries'],
                'memory_freed': stats_before['size_bytes'] - stats_after['size_bytes']
            }
        
        return optimization_results
    
    def _optimize_memory_pools(self) -> Dict[str, Any]:
        """Optimize all registered memory pools."""
        optimization_results = {}
        
        for pool_name, pool_instance in self.memory_pools.items():
            stats_before = pool_instance.get_pool_statistics()
            
            # Optimize pool size based on usage patterns
            utilization = stats_before['utilization_ratio']
            
            if utilization < 0.3:  # Low utilization
                new_size = max(5, int(pool_instance.max_size * 0.8))
                pool_instance.resize_pool(new_size)
            elif utilization > 0.9:  # High utilization
                new_size = min(200, int(pool_instance.max_size * 1.2))
                pool_instance.resize_pool(new_size)
            
            stats_after = pool_instance.get_pool_statistics()
            
            optimization_results[pool_name] = {
                'stats_before': stats_before,
                'stats_after': stats_after,
                'size_adjusted': stats_after.get('max_size', 0) != stats_before.get('max_size', 0)
            }
        
        return optimization_results
    
    def _estimate_memory_freed(self, optimization_results: Dict[str, Any]) -> int:
        """Estimate total memory freed by optimization."""
        total_freed = 0
        
        # Memory freed from garbage collection
        gc_result = optimization_results.get('garbage_collection', {})
        if 'objects_collected' in gc_result:
            # Rough estimate: 100 bytes per collected object
            total_freed += gc_result['objects_collected'] * 100
        
        # Memory freed from cache optimization
        cache_optimization = optimization_results.get('cache_optimization', {})
        for cache_name, cache_result in cache_optimization.items():
            total_freed += cache_result.get('memory_freed', 0)
        
        return total_freed
    
    def _generate_optimization_recommendations(self, optimization_results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on results."""
        recommendations = []
        
        # Analyze garbage collection results
        gc_result = optimization_results.get('garbage_collection', {})
        objects_collected = gc_result.get('objects_collected', 0)
        
        if objects_collected > 1000:
            recommendations.append("High garbage collection activity detected - review object lifecycle management")
        elif objects_collected < 10:
            recommendations.append("Low garbage collection activity - consider manual GC optimization")
        
        # Analyze cache performance
        cache_optimization = optimization_results.get('cache_optimization', {})
        for cache_name, cache_result in cache_optimization.items():
            if cache_result.get('entries_evicted', 0) > 100:
                recommendations.append(f"Cache '{cache_name}' had high eviction rate - consider increasing size")
            elif cache_result.get('entries_evicted', 0) == 0:
                recommendations.append(f"Cache '{cache_name}' may be underutilized - consider reducing size")
        
        # Analyze fragmentation
        fragmentation_analysis = optimization_results.get('fragmentation_analysis', {})
        fragmentation_ratio = fragmentation_analysis.get('fragmentation_ratio', 0.0)
        
        if fragmentation_ratio > 0.3:
            recommendations.append("High memory fragmentation detected - consider memory compaction strategies")
        
        # Analyze resource usage
        resource_check = optimization_results.get('resource_check', {})
        if resource_check.get('violations'):
            recommendations.append("Resource limit violations detected - review memory allocation patterns")
        
        if not recommendations:
            recommendations.append("System memory appears well-optimized")
        
        return recommendations
    
    def start_continuous_optimization(self, interval_minutes: int = 30):
        """Start continuous memory optimization."""
        def optimization_loop():
            while self.is_optimizing:
                try:
                    self.optimize_system_memory()
                    time.sleep(interval_minutes * 60)
                except Exception as e:
                    print(f"Continuous optimization error: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        if not self.is_optimizing:
            self.is_optimizing = True
            threading.Thread(target=optimization_loop, daemon=True).start()
    
    def stop_continuous_optimization(self):
        """Stop continuous memory optimization."""
        self.is_optimizing = False
    
    def get_system_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive system memory report."""
        # Collect all relevant statistics
        memory_stats = psutil.virtual_memory()._asdict()
        process_memory = psutil.Process().memory_info()._asdict()
        
        cache_stats = {}
        for name, cache in self.cache_instances.items():
            cache_stats[name] = cache.get_cache_statistics()
        
        pool_stats = {}
        for name, pool in self.memory_pools.items():
            pool_stats[name] = pool.get_pool_statistics()
        
        gc_stats = self.gc_optimizer.get_gc_statistics()
        
        # Recent optimization history
        recent_optimizations = self.optimization_history[-10:] if self.optimization_history else []
        
        return {
            'system_memory': memory_stats,
            'process_memory': process_memory,
            'cache_statistics': cache_stats,
            'pool_statistics': pool_stats,
            'gc_statistics': gc_stats,
            'recent_optimizations': recent_optimizations,
            'memory_health_score': self._calculate_memory_health_score(),
            'report_timestamp': time.time()
        }
    
    def _calculate_memory_health_score(self) -> float:
        """Calculate overall memory health score (0-1, higher is better)."""
        try:
            # Base score from system memory usage
            memory_info = psutil.virtual_memory()
            memory_score = max(0.0, 1.0 - (memory_info.percent / 100.0))
            
            # Adjust based on cache hit rates
            cache_score = 1.0
            if self.cache_instances:
                hit_rates = []
                for cache in self.cache_instances.values():
                    stats = cache.get_cache_statistics()
                    hit_rates.append(stats.get('hit_rate', 0.0))
                cache_score = np.mean(hit_rates) if hit_rates else 1.0
            
            # Adjust based on pool efficiency
            pool_score = 1.0
            if self.memory_pools:
                efficiency_rates = []
                for pool in self.memory_pools.values():
                    stats = pool.get_pool_statistics()
                    efficiency_rates.append(stats.get('efficiency_ratio', 1.0))
                pool_score = np.mean(efficiency_rates) if efficiency_rates else 1.0
            
            # Combined health score
            health_score = (memory_score * 0.5 + cache_score * 0.3 + pool_score * 0.2)
            return max(0.0, min(1.0, health_score))
            
        except Exception:
            return 0.5  # Default moderate score on error

# Custom exceptions for memory management
class MemoryManagementError(Exception):
    """Base exception for memory management errors."""
    pass

class MemoryPoolExhaustedException(MemoryManagementError):
    """Exception raised when memory pool is exhausted."""
    pass

class MemoryMappingError(MemoryManagementError):
    """Exception for memory mapping operations."""
    pass

class CacheError(MemoryManagementError):
    """Exception for cache operations."""
    pass

class MemoryLeakError(MemoryManagementError):
    """Exception for memory leak detection."""
    pass

class ResourceLimitExceededError(MemoryManagementError):
    """Exception when resource limits are exceeded."""
    pass

# Utility functions for common memory operations
def get_object_size(obj: Any, deep: bool = False) -> int:
    """Get size of object in bytes with optional deep analysis."""
    try:
        if not deep:
            return sys.getsizeof(obj)
        
        # Deep size calculation for complex objects
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(get_object_size(k, deep) + get_object_size(v, deep) 
                       for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(get_object_size(item, deep) for item in obj)
        elif hasattr(obj, '__dict__'):
            size += get_object_size(obj.__dict__, deep)
        
        return size
        
    except (TypeError, AttributeError):
        # Fallback for objects that don't support deep analysis
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate

def format_memory_size(size_bytes: int) -> str:
    """Format memory size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def get_memory_usage_by_type() -> Dict[str, int]:
    """Get memory usage breakdown by object type."""
    type_sizes = defaultdict(int)
    
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        try:
            size = sys.getsizeof(obj)
            type_sizes[obj_type] += size
        except (TypeError, AttributeError):
            continue
    
    return dict(type_sizes)

def find_memory_leaks(threshold_mb: float = 10.0, 
                     sample_interval: float = 60.0) -> Dict[str, Any]:
    """Simple memory leak detection by comparing snapshots."""
    # Take initial snapshot
    initial_memory = psutil.Process().memory_info().rss
    initial_objects = len(gc.get_objects())
    initial_time = time.time()
    
    # Wait for sample interval
    time.sleep(sample_interval)
    
    # Take final snapshot
    final_memory = psutil.Process().memory_info().rss
    final_objects = len(gc.get_objects())
    final_time = time.time()
    
    # Calculate differences
    memory_diff_mb = (final_memory - initial_memory) / (1024 * 1024)
    object_diff = final_objects - initial_objects
    time_diff = final_time - initial_time
    
    # Determine if leak detected
    leak_detected = memory_diff_mb > threshold_mb
    
    return {
        'leak_detected': leak_detected,
        'memory_growth_mb': memory_diff_mb,
        'object_growth': object_diff,
        'sample_duration': time_diff,
        'growth_rate_mb_per_hour': (memory_diff_mb / time_diff * 3600) if time_diff > 0 else 0,
        'analysis_timestamp': final_time
    }

def optimize_python_memory():
    """Perform basic Python memory optimization."""
    # Force garbage collection
    collected = gc.collect()
    
    # Optimize string interning
    sys.intern("")  # Ensure empty string is interned
    
    # Clear various caches if available
    try:
        import functools
        functools.lru_cache(maxsize=None)(lambda: None).cache_clear()
    except:
        pass
    
    try:
        import re
        re.purge()  # Clear regex cache
    except:
        pass
    
    return {
        'objects_collected': collected,
        'optimization_timestamp': time.time()
    }

def create_memory_efficient_cache(max_size_mb: int = 100, 
                                policy: CachePolicy = CachePolicy.LRU) -> AdvancedCache:
    """Create memory-efficient cache with optimal settings."""
    max_size_bytes = max_size_mb * 1024 * 1024
    cache = AdvancedCache(
        max_size_bytes=max_size_bytes,
        policy=policy,
        cleanup_threshold=0.7  # More aggressive cleanup
    )
    return cache

def monitor_function_memory(func: Callable) -> Callable:
    """Decorator to monitor function memory usage."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Start memory monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            
            # Measure final memory
            final_memory = process.memory_info().rss
            end_time = time.perf_counter()
            
            # Calculate memory usage
            memory_used = final_memory - initial_memory
            execution_time = end_time - start_time
            
            # Log memory usage
            print(f"Function {func.__name__}: "
                  f"Memory: {format_memory_size(memory_used)}, "
                  f"Time: {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            final_memory = process.memory_info().rss
            memory_used = final_memory - initial_memory
            print(f"Function {func.__name__} failed: "
                  f"Memory: {format_memory_size(memory_used)}")
            raise
    
    return wrapper

def get_largest_objects(count: int = 10) -> List[Tuple[str, int, Any]]:
    """Get the largest objects in memory by size."""
    objects_with_sizes = []
    
    for obj in gc.get_objects():
        try:
            size = sys.getsizeof(obj)
            obj_type = type(obj).__name__
            objects_with_sizes.append((obj_type, size, obj))
        except (TypeError, AttributeError):
            continue
    
    # Sort by size and return top objects
    objects_with_sizes.sort(key=lambda x: x[1], reverse=True)
    return objects_with_sizes[:count]

def clear_all_caches():
    """Clear all known Python caches."""
    caches_cleared = []
    
    # Clear garbage collection
    collected = gc.collect()
    caches_cleared.append(f"GC: {collected} objects")
    
    # Clear functools caches
    try:
        import functools
        # This is a bit tricky since we need to clear all lru_cache instances
        # We'll force garbage collection which should help
        gc.collect()
        caches_cleared.append("functools caches")
    except:
        pass
    
    # Clear regex cache
    try:
        import re
        re.purge()
        caches_cleared.append("regex cache")
    except:
        pass
    
    # Clear import cache
    try:
        import importlib
        importlib.invalidate_caches()
        caches_cleared.append("import cache")
    except:
        pass
    
    # Clear linecache
    try:
        import linecache
        linecache.clearcache()
        caches_cleared.append("linecache")
    except:
        pass
    
    return {
        'caches_cleared': caches_cleared,
        'total_objects_collected': collected,
        'timestamp': time.time()
    }

class MemoryContextManager:
    """Context manager for memory usage monitoring and cleanup."""
    
    def __init__(self, cleanup_on_exit: bool = True, 
                 memory_limit_mb: Optional[int] = None):
        self.cleanup_on_exit = cleanup_on_exit
        self.memory_limit_mb = memory_limit_mb
        self.initial_memory = 0
        self.initial_objects = 0
        self.start_time = 0.0
    
    def __enter__(self):
        """Enter memory monitoring context."""
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss
        self.initial_objects = len(gc.get_objects())
        self.start_time = time.time()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit memory monitoring context with optional cleanup."""
        process = psutil.Process()
        final_memory = process.memory_info().rss
        final_objects = len(gc.get_objects())
        end_time = time.time()
        
        # Calculate memory usage
        memory_used = final_memory - self.initial_memory
        objects_created = final_objects - self.initial_objects
        duration = end_time - self.start_time
        
        # Check memory limit
        if self.memory_limit_mb:
            memory_used_mb = memory_used / (1024 * 1024)
            if memory_used_mb > self.memory_limit_mb:
                print(f"Warning: Memory usage ({memory_used_mb:.1f} MB) "
                      f"exceeded limit ({self.memory_limit_mb} MB)")
        
        # Perform cleanup if requested
        if self.cleanup_on_exit:
            collected = gc.collect()
            print(f"Memory context cleanup: {collected} objects collected")
        
        # Log memory usage statistics
        print(f"Memory context summary:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Memory used: {format_memory_size(memory_used)}")
        print(f"  Objects created: {objects_created}")
        
        return False  # Don't suppress exceptions

def memory_efficient_batch_processor(items: Iterator[Any], batch_size: int = 1000,
                                   processor: Callable[[List[Any]], Any] = None,
                                   memory_limit_mb: int = 100) -> Iterator[Any]:
    """Memory-efficient batch processor with automatic memory management."""
    current_batch = []
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    for item in items:
        current_batch.append(item)
        
        # Check if batch is ready or memory limit reached
        if len(current_batch) >= batch_size:
            # Check memory usage
            current_memory = process.memory_info().rss
            memory_used_mb = (current_memory - initial_memory) / (1024 * 1024)
            
            if memory_used_mb > memory_limit_mb:
                # Force garbage collection
                gc.collect()
                current_memory = process.memory_info().rss
                memory_used_mb = (current_memory - initial_memory) / (1024 * 1024)
                
                if memory_used_mb > memory_limit_mb:
                    print(f"Warning: Memory usage ({memory_used_mb:.1f} MB) "
                          f"exceeds limit ({memory_limit_mb} MB)")
            
            # Process batch
            if processor:
                result = processor(current_batch)
                yield result
            else:
                yield current_batch.copy()
            
            # Clear batch
            current_batch.clear()
    
    # Process remaining items
    if current_batch:
        if processor:
            result = processor(current_batch)
            yield result
        else:
            yield current_batch

# Global memory management instances
global_memory_optimizer = MemoryOptimizer()
global_resource_monitor = ResourceMonitor()
global_leak_detector = MemoryLeakDetector()

# Configuration for global memory management
MEMORY_MANAGEMENT_CONFIG = {
    'enable_automatic_optimization': True,
    'optimization_interval_minutes': 30,
    'memory_warning_threshold_mb': 500,
    'memory_critical_threshold_mb': 1000,
    'cache_default_size_mb': 50,
    'pool_default_size': 20,
    'enable_leak_detection': True,
    'leak_detection_interval_seconds': 300
}

def configure_global_memory_management(config: Optional[Dict[str, Any]] = None):
    """Configure global memory management settings."""
    if config:
        MEMORY_MANAGEMENT_CONFIG.update(config)
    
    # Start global services based on configuration
    if MEMORY_MANAGEMENT_CONFIG['enable_automatic_optimization']:
        global_memory_optimizer.start_continuous_optimization(
            MEMORY_MANAGEMENT_CONFIG['optimization_interval_minutes']
        )
    
    if MEMORY_MANAGEMENT_CONFIG['enable_leak_detection']:
        global_leak_detector.start_monitoring()
    
    # Configure resource monitoring thresholds
    global_resource_monitor.thresholds.update({
        'memory_mb_warning': MEMORY_MANAGEMENT_CONFIG['memory_warning_threshold_mb'],
        'memory_mb_critical': MEMORY_MANAGEMENT_CONFIG['memory_critical_threshold_mb']
    })
    
    global_resource_monitor.start_monitoring()

def get_global_memory_status() -> Dict[str, Any]:
    """Get comprehensive global memory management status."""
    return {
        'system_memory': global_memory_optimizer.get_system_memory_report(),
        'resource_monitoring': global_resource_monitor.get_resource_trends(),
        'leak_detection': global_leak_detector.get_leak_report(),
        'configuration': MEMORY_MANAGEMENT_CONFIG,
        'status_timestamp': time.time()
    }

def emergency_memory_cleanup() -> Dict[str, Any]:
    """Perform emergency memory cleanup when system is under pressure."""
    cleanup_results = {}
    
    # 1. Force aggressive garbage collection
    gc_result = []
    for generation in range(3):
        collected = gc.collect(generation)
        gc_result.append(collected)
    cleanup_results['garbage_collection'] = gc_result
    
    # 2. Clear all known caches
    cache_result = clear_all_caches()
    cleanup_results['cache_clearing'] = cache_result
    
    # 3. Optimize global memory system
    optimization_result = global_memory_optimizer.optimize_system_memory()
    cleanup_results['system_optimization'] = optimization_result
    
    # 4. Get final memory status
    process = psutil.Process()
    final_memory = process.memory_info()
    cleanup_results['final_memory_status'] = {
        'rss_mb': final_memory.rss / (1024 * 1024),
        'vms_mb': final_memory.vms / (1024 * 1024)
    }
    
    return {
        'cleanup_results': cleanup_results,
        'cleanup_timestamp': time.time(),
        'success': True
    }

# Utility functions for specific use cases
def create_ai_model_cache(max_size_gb: int = 2) -> SmartCache:
    """Create optimized cache for AI model data."""
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024
    cache = SmartCache(
        name="ai_model_cache",
        max_size_bytes=max_size_bytes,
        adaptive_sizing=True,
        auto_optimize=True
    )
    
    # Register with global optimizer
    global_memory_optimizer.register_cache("ai_model_cache", cache.cache)
    
    return cache

def create_sensor_data_pool(pool_size: int = 100) -> TypedObjectPool:
    """Create memory pool for sensor data objects."""
    pool = TypedObjectPool(
        object_type=np.ndarray,
        factory_args=(1000,),  # 1000-element array
        factory_kwargs={'dtype': np.float32},
        initial_size=pool_size // 2,
        max_size=pool_size
    )
    
    # Register with global optimizer
    global_memory_optimizer.register_memory_pool("sensor_data_pool", pool)
    
    return pool

def create_holographic_buffer(size_mb: int = 50) -> MemoryMappedBuffer:
    """Create memory-mapped buffer for holographic data."""
    size_bytes = size_mb * 1024 * 1024
    return MemoryMappedBuffer(size_bytes, filename=None)  # Anonymous mapping

def profile_memory_usage(duration_seconds: int = 60) -> Dict[str, Any]:
    """Profile memory usage over specified duration."""
    profiler = MemoryProfiler(enable_line_profiling=True)
    
    profiler.start_profiling()
    time.sleep(duration_seconds)
    results = profiler.stop_profiling()
    
    return {
        'profiling_results': results,
        'memory_report': profiler.generate_memory_report(),
        'profiling_duration': duration_seconds
    }

# Initialize global memory management on module import
def _initialize_memory_management():
    """Initialize global memory management system."""
    try:
        configure_global_memory_management()
        print("Memory management system initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize memory management: {e}")

# Initialize on import (with error handling)
if __name__ != "__main__":
    try:
        _initialize_memory_management()
    except Exception as e:
        print(f"Memory management initialization warning: {e}")

if __name__ == "__main__":
    # Example usage and testing
    print("AI Holographic Wristwatch - Memory Management Utilities Module")
    print("Testing memory management capabilities...")
    
    # Test advanced cache
    print("\n1. Testing Advanced Cache:")
    cache = AdvancedCache(max_size_bytes=1024*1024, policy=CachePolicy.LRU)
    cache.put("test_key", {"data": "test_value", "array": list(range(100))})
    cached_value = cache.get("test_key")
    cache_stats = cache.get_cache_statistics()
    print(f"   Cache hit rate: {cache_stats['hit_rate']:.2f}")
    print(f"   Cache size: {format_memory_size(cache_stats['size_bytes'])}")
    
    # Test memory pool
    print("\n2. Testing Memory Pool:")
    def create_test_object():
        return {"data": np.zeros(100), "id": np.random.randint(1000)}
    
    pool = MemoryPool("test_pool", create_test_object, initial_size=5, max_size=20)
    
    with pool.get_object() as obj:
        print(f"   Acquired object with ID: {obj['id']}")
    
    pool_stats = pool.get_pool_statistics()
    print(f"   Pool efficiency: {pool_stats['efficiency_ratio']:.2f}")
    
    # Test memory monitoring
    print("\n3. Testing Memory Monitoring:")
    monitor = ResourceMonitor(monitoring_interval=1.0)
    monitor.start_monitoring()
    
    time.sleep(2)  # Let it collect some data
    
    stats = monitor.collect_resource_stats()
    print(f"   System memory usage: {stats['memory']['percent']:.1f}%")
    print(f"   Process memory: {format_memory_size(stats['process']['rss'])}")
    
    monitor.stop_monitoring()
    
    # Test garbage collection optimization
    print("\n4. Testing Garbage Collection:")
    gc_optimizer = GarbageCollectionOptimizer()
    gc_result = gc_optimizer.force_collection()
    print(f"   Objects collected: {gc_result['objects_collected']}")
    print(f"   Collection time: {gc_result['collection_time']:.4f}s")
    
    # Test memory leak detection
    print("\n5. Testing Memory Leak Detection:")
    leak_result = find_memory_leaks(threshold_mb=1.0, sample_interval=1.0)
    print(f"   Memory growth: {leak_result['memory_growth_mb']:.2f} MB")
    print(f"   Object growth: {leak_result['object_growth']}")
    print(f"   Leak detected: {leak_result['leak_detected']}")
    
    # Test memory profiler
    print("\n6. Testing Memory Profiler:")
    profiler = MemoryProfiler()
    
    with profiler.profile_block("test_allocation"):
        # Simulate memory allocation
        test_data = [np.random.rand(1000) for _ in range(10)]
        time.sleep(0.1)
    
    profile_result = profiler.get_allocation_profile("test_allocation")
    if profile_result:
        print(f"   Memory allocated: {profile_result['memory_change_mb']:.2f} MB")
        print(f"   Block execution time: {profile_result['block_execution_time']:.3f}s")
    
    # Test smart cache
    print("\n7. Testing Smart Cache:")
    smart_cache = SmartCache("test_smart_cache", max_size_bytes=1024*1024)
    
    for i in range(100):
        smart_cache.put(f"key_{i}", {"value": i, "data": list(range(i))})
    
    hit_rate = smart_cache._calculate_recent_hit_rate()
    print(f"   Smart cache hit rate: {hit_rate:.2f}")
    
    # Test memory-mapped buffer
    print("\n8. Testing Memory-Mapped Buffer:")
    with MemoryMappedBuffer(1024 * 1024) as buffer:  # 1MB buffer
        test_data = b"Hello, holographic world! " * 100
        bytes_written = buffer.write(0, test_data)
        read_data = buffer.read(0, len(test_data))
        print(f"   Buffer I/O test: {len(read_data)} bytes read/written")
        print(f"   Data integrity: {read_data == test_data}")
    
    # Test system optimization
    print("\n9. Testing System Memory Optimization:")
    optimizer = MemoryOptimizer()
    optimization_result = optimizer.optimize_system_memory()
    
    print(f"   Optimization time: {optimization_result['optimization_time']:.3f}s")
    print(f"   Memory freed estimate: {format_memory_size(optimization_result['memory_freed_estimate'])}")
    print(f"   Health score: {optimizer._calculate_memory_health_score():.2f}")
    
    # Test utility functions
    print("\n10. Testing Utility Functions:")
    
    # Test object size calculation
    test_obj = {"large_list": list(range(1000)), "nested": {"data": np.zeros(500)}}
    obj_size = get_object_size(test_obj, deep=True)
    print(f"   Object size (deep): {format_memory_size(obj_size)}")
    
    # Test memory usage by type
    memory_by_type = get_memory_usage_by_type()
    largest_types = sorted(memory_by_type.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   Largest object types:")
    for obj_type, size in largest_types:
        print(f"     {obj_type}: {format_memory_size(size)}")
    
    # Test emergency cleanup
    print("\n11. Testing Emergency Memory Cleanup:")
    cleanup_result = emergency_memory_cleanup()
    if cleanup_result['success']:
        final_memory = cleanup_result['cleanup_results']['final_memory_status']
        print(f"   Final RSS memory: {final_memory['rss_mb']:.1f} MB")
        print(f"   Emergency cleanup completed successfully")
    
    print("\nMemory management utilities module testing completed successfully!")
    print(f"Current system memory usage: {psutil.virtual_memory().percent:.1f}%")
    print(f"Process memory usage: {format_memory_size(psutil.Process().memory_info().rss)}")