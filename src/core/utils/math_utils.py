"""
Advanced Mathematical Operations for AI Holographic Wristwatch System

This module provides comprehensive mathematical utilities for holographic calculations,
sensor data processing, 3D transformations, AI system computations, advanced optimization,
statistical analysis, signal processing, numerical methods, and performance profiling.
"""

import numpy as np
import math
import cmath
from typing import Tuple, List, Optional, Union, Any, Callable, Dict, Iterator
from dataclasses import dataclass, field
from enum import Enum
import scipy.fft
import scipy.stats
import scipy.optimize
import scipy.linalg
import scipy.integrate
import scipy.interpolate
import scipy.signal
import scipy.sparse
import scipy.spatial
from scipy.special import *
from collections import defaultdict, deque
import warnings
import functools
import time
import threading
import concurrent.futures
import asyncio
from abc import ABC, abstractmethod
import weakref
import pickle
import json
from contextlib import contextmanager
import tracemalloc
import psutil
import os

class TransformationType(Enum):
    """Types of mathematical transformations supported."""
    ROTATION = "rotation"
    TRANSLATION = "translation"
    SCALING = "scaling"
    PERSPECTIVE = "perspective"
    HOLOGRAPHIC = "holographic"
    AFFINE = "affine"
    PROJECTIVE = "projective"
    NONLINEAR = "nonlinear"

class OptimizationMethod(Enum):
    """Optimization algorithms available."""
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    LBFGS = "lbfgs"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"

class DistributionType(Enum):
    """Probability distribution types."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    BINOMIAL = "binomial"
    POISSON = "poisson"
    MULTIVARIATE_NORMAL = "multivariate_normal"

class FilterType(Enum):
    """Digital filter types."""
    LOW_PASS = "lowpass"
    HIGH_PASS = "highpass"
    BAND_PASS = "bandpass"
    BAND_STOP = "bandstop"
    NOTCH = "notch"
    ALL_PASS = "allpass"

class WindowType(Enum):
    """Window function types for signal processing."""
    HANNING = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    GAUSSIAN = "gaussian"
    KAISER = "kaiser"
    BARTLETT = "bartlett"

@dataclass
class MathPerformanceStats:
    """Performance statistics for mathematical operations."""
    operation_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    memory_usage_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update(self, execution_time: float, memory_usage: int = 0):
        """Update performance statistics with new measurement."""
        self.operation_count += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.operation_count
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.memory_usage_bytes += memory_usage

@dataclass
class OptimizationResult:
    """Result of optimization procedures."""
    success: bool
    optimal_parameters: np.ndarray
    optimal_value: float
    iterations: int
    function_evaluations: int
    convergence_message: str
    execution_time: float
    gradient_norm: Optional[float] = None
    hessian_condition_number: Optional[float] = None

@dataclass
class Vector3D:
    """3D vector representation for spatial calculations."""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector3D':
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3D':
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self) -> 'Vector3D':
        return Vector3D(-self.x, -self.y, -self.z)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def magnitude_squared(self) -> float:
        return self.x**2 + self.y**2 + self.z**2
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def distance_to(self, other: 'Vector3D') -> float:
        return (self - other).magnitude()
    
    def angle_to(self, other: 'Vector3D') -> float:
        """Calculate angle between vectors in radians."""
        dot_product = self.normalize().dot(other.normalize())
        dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to valid range
        return math.acos(dot_product)
    
    def project_onto(self, other: 'Vector3D') -> 'Vector3D':
        """Project this vector onto another vector."""
        other_normalized = other.normalize()
        projection_length = self.dot(other_normalized)
        return other_normalized * projection_length
    
    def reflect(self, normal: 'Vector3D') -> 'Vector3D':
        """Reflect this vector across a surface with given normal."""
        normal_normalized = normal.normalize()
        return self - 2 * self.dot(normal_normalized) * normal_normalized
    
    def lerp(self, other: 'Vector3D', t: float) -> 'Vector3D':
        """Linear interpolation between this vector and another."""
        t = max(0.0, min(1.0, t))
        return self + t * (other - self)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'Vector3D':
        if len(array) != 3:
            raise ValueError("Array must have exactly 3 elements")
        return cls(float(array[0]), float(array[1]), float(array[2]))
    
    @classmethod
    def zero(cls) -> 'Vector3D':
        return cls(0, 0, 0)
    
    @classmethod
    def one(cls) -> 'Vector3D':
        return cls(1, 1, 1)
    
    @classmethod
    def unit_x(cls) -> 'Vector3D':
        return cls(1, 0, 0)
    
    @classmethod
    def unit_y(cls) -> 'Vector3D':
        return cls(0, 1, 0)
    
    @classmethod
    def unit_z(cls) -> 'Vector3D':
        return cls(0, 0, 1)

@dataclass
class Quaternion:
    """Quaternion representation for 3D rotations with comprehensive operations."""
    w: float
    x: float
    y: float
    z: float
    
    def __mul__(self, other: Union['Quaternion', float]) -> 'Quaternion':
        if isinstance(other, Quaternion):
            return self.multiply(other)
        else:
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
    
    def __rmul__(self, scalar: float) -> 'Quaternion':
        return Quaternion(self.w * scalar, self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> 'Quaternion':
        """Create quaternion from Euler angles (in radians)."""
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return cls(w, x, y, z)
    
    @classmethod
    def from_axis_angle(cls, axis: Vector3D, angle: float) -> 'Quaternion':
        """Create quaternion from axis-angle representation."""
        norm_axis = axis.normalize()
        half_angle = angle * 0.5
        sin_half = math.sin(half_angle)
        
        return cls(
            math.cos(half_angle),
            norm_axis.x * sin_half,
            norm_axis.y * sin_half,
            norm_axis.z * sin_half
        )
    
    @classmethod
    def from_rotation_matrix(cls, matrix: np.ndarray) -> 'Quaternion':
        """Create quaternion from 3x3 rotation matrix."""
        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (matrix[2, 1] - matrix[1, 2]) / s
            y = (matrix[0, 2] - matrix[2, 0]) / s
            z = (matrix[1, 0] - matrix[0, 1]) / s
        else:
            if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
                s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
                w = (matrix[2, 1] - matrix[1, 2]) / s
                x = 0.25 * s
                y = (matrix[0, 1] + matrix[1, 0]) / s
                z = (matrix[0, 2] + matrix[2, 0]) / s
            elif matrix[1, 1] > matrix[2, 2]:
                s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
                w = (matrix[0, 2] - matrix[2, 0]) / s
                x = (matrix[0, 1] + matrix[1, 0]) / s
                y = 0.25 * s
                z = (matrix[1, 2] + matrix[2, 1]) / s
            else:
                s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
                w = (matrix[1, 0] - matrix[0, 1]) / s
                x = (matrix[0, 2] + matrix[2, 0]) / s
                y = (matrix[1, 2] + matrix[2, 1]) / s
                z = 0.25 * s
        
        return cls(w, x, y, z).normalize()
    
    def magnitude(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Quaternion':
        """Normalize the quaternion."""
        norm = self.magnitude()
        if norm == 0:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)
    
    def conjugate(self) -> 'Quaternion':
        """Return the conjugate of the quaternion."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self) -> 'Quaternion':
        """Return the inverse quaternion."""
        conj = self.conjugate()
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq == 0:
            raise ZeroDivisionError("Cannot invert zero quaternion")
        return Quaternion(conj.w / norm_sq, conj.x / norm_sq, conj.y / norm_sq, conj.z / norm_sq)
    
    def multiply(self, other: 'Quaternion') -> 'Quaternion':
        """Multiply two quaternions."""
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)
    
    def rotate_vector(self, vector: Vector3D) -> Vector3D:
        """Rotate a vector using this quaternion."""
        q_norm = self.normalize()
        v_quat = Quaternion(0, vector.x, vector.y, vector.z)
        result = q_norm.multiply(v_quat).multiply(q_norm.conjugate())
        return Vector3D(result.x, result.y, result.z)
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    def to_euler_angles(self) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        q = self.normalize()
        
        # Roll (rotation around x-axis)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (rotation around y-axis)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (rotation around z-axis)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def to_axis_angle(self) -> Tuple[Vector3D, float]:
        """Convert quaternion to axis-angle representation."""
        q = self.normalize()
        
        angle = 2 * math.acos(abs(q.w))
        
        if angle < 1e-6:
            # Near identity rotation
            axis = Vector3D(1, 0, 0)
        else:
            s = math.sqrt(1 - q.w * q.w)
            if s < 1e-6:
                axis = Vector3D(1, 0, 0)
            else:
                axis = Vector3D(q.x / s, q.y / s, q.z / s)
        
        return axis, angle
    
    def dot(self, other: 'Quaternion') -> float:
        """Calculate dot product with another quaternion."""
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    
    @classmethod
    def identity(cls) -> 'Quaternion':
        return cls(1, 0, 0, 0)

class AdvancedLinearAlgebra:
    """Advanced linear algebra operations for matrix computations."""
    
    @staticmethod
    def svd_decomposition(matrix: np.ndarray, full_matrices: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform Singular Value Decomposition."""
        U, s, Vt = scipy.linalg.svd(matrix, full_matrices=full_matrices)
        return U, s, Vt
    
    @staticmethod
    def qr_decomposition(matrix: np.ndarray, mode: str = 'full') -> Tuple[np.ndarray, np.ndarray]:
        """Perform QR decomposition."""
        return scipy.linalg.qr(matrix, mode=mode)
    
    @staticmethod
    def eigendecomposition(matrix: np.ndarray, 
                          compute_left: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors."""
        if compute_left:
            eigenvalues, eigenvectors = scipy.linalg.eig(matrix)
        else:
            eigenvalues = scipy.linalg.eigvals(matrix)
            eigenvectors = None
        
        return eigenvalues, eigenvectors
    
    @staticmethod
    def schur_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Schur decomposition."""
        T, Z = scipy.linalg.schur(matrix)
        return T, Z
    
    @staticmethod
    def cholesky_decomposition(matrix: np.ndarray, lower: bool = True) -> np.ndarray:
        """Compute Cholesky decomposition for positive definite matrices."""
        return scipy.linalg.cholesky(matrix, lower=lower)
    
    @staticmethod
    def lu_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform LU decomposition with partial pivoting."""
        P, L, U = scipy.linalg.lu(matrix)
        return P, L, U
    
    @staticmethod
    def matrix_exponential(matrix: np.ndarray) -> np.ndarray:
        """Compute matrix exponential."""
        return scipy.linalg.expm(matrix)
    
    @staticmethod
    def matrix_logarithm(matrix: np.ndarray) -> np.ndarray:
        """Compute matrix logarithm."""
        return scipy.linalg.logm(matrix)
    
    @staticmethod
    def matrix_square_root(matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root."""
        return scipy.linalg.sqrtm(matrix)
    
    @staticmethod
    def condition_number(matrix: np.ndarray, p: Optional[Union[None, int, str]] = None) -> float:
        """Compute condition number of matrix."""
        return scipy.linalg.norm(matrix, ord=p) * scipy.linalg.norm(scipy.linalg.pinv(matrix), ord=p)
    
    @staticmethod
    def matrix_rank(matrix: np.ndarray, tol: Optional[float] = None) -> int:
        """Compute rank of matrix."""
        return np.linalg.matrix_rank(matrix, tol=tol)
    
    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray, 
                           method: str = 'auto') -> np.ndarray:
        """Solve linear system Ax = b using specified method."""
        if method == 'auto':
            return scipy.linalg.solve(A, b)
        elif method == 'lu':
            return scipy.linalg.solve(A, b, assume_a='gen')
        elif method == 'cholesky':
            return scipy.linalg.solve(A, b, assume_a='pos')
        elif method == 'lstsq':
            return scipy.linalg.lstsq(A, b)[0]
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def generalized_eigenvalue_problem(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve generalized eigenvalue problem Av = λBv."""
        return scipy.linalg.eig(A, B)
    
    @staticmethod
    def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute Kronecker product of two matrices."""
        return np.kron(A, B)
    
    @staticmethod
    def hadamard_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute element-wise (Hadamard) product."""
        return np.multiply(A, B)
    
    @staticmethod
    def tensor_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute tensor product of two arrays."""
        return np.tensordot(A, B, axes=0)
    
    @staticmethod
    def moore_penrose_pseudoinverse(matrix: np.ndarray, rcond: Optional[float] = None) -> np.ndarray:
        """Compute Moore-Penrose pseudoinverse."""
        return scipy.linalg.pinv(matrix, rcond=rcond)

class HolographicCalculations:
    """Advanced calculations for holographic projection systems."""
    
    @staticmethod
    def calculate_projection_matrix(viewer_position: Vector3D, target_position: Vector3D, 
                                  field_of_view: float, aspect_ratio: float,
                                  near_plane: float, far_plane: float) -> np.ndarray:
        """Calculate projection matrix for holographic display."""
        fov_rad = math.radians(field_of_view)
        f = 1.0 / math.tan(fov_rad / 2.0)
        
        projection_matrix = np.array([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far_plane + near_plane) / (near_plane - far_plane), 
             (2 * far_plane * near_plane) / (near_plane - far_plane)],
            [0, 0, -1, 0]
        ])
        
        return projection_matrix
    
    @staticmethod
    def calculate_hologram_size(distance: float, viewing_angle: float, 
                              base_size: float) -> float:
        """Calculate optimal hologram size based on viewing conditions."""
        size_factor = math.tan(math.radians(viewing_angle / 2)) * distance * 2
        return min(base_size * size_factor, base_size * 3.0)  # Cap maximum size
    
    @staticmethod
    def calculate_interference_pattern(wave1: np.ndarray, wave2: np.ndarray,
                                     wavelength: float) -> np.ndarray:
        """Calculate interference pattern for holographic recording."""
        phase_diff = 2 * np.pi * np.abs(wave1 - wave2) / wavelength
        intensity = np.square(np.cos(phase_diff / 2))
        return intensity
    
    @staticmethod
    def optimize_viewing_angle(user_position: Vector3D, hologram_center: Vector3D,
                             hologram_normal: Vector3D) -> float:
        """Calculate optimal viewing angle for hologram visibility."""
        view_vector = (user_position - hologram_center).normalize()
        normal_vector = hologram_normal.normalize()
        
        dot_product = view_vector.dot(normal_vector)
        angle = math.acos(max(-1.0, min(1.0, dot_product)))
        
        # Optimize for angles between 30-60 degrees for best visibility
        optimal_angle = np.clip(math.degrees(angle), 30, 60)
        return optimal_angle
    
    @staticmethod
    def calculate_fresnel_diffraction(aperture_size: float, distance: float, 
                                    wavelength: float) -> np.ndarray:
        """Calculate Fresnel diffraction pattern."""
        # Create coordinate grid
        x = np.linspace(-aperture_size, aperture_size, 500)
        y = np.linspace(-aperture_size, aperture_size, 500)
        X, Y = np.meshgrid(x, y)
        
        # Calculate Fresnel numbers
        fresnel_x = X * np.sqrt(2 / (wavelength * distance))
        fresnel_y = Y * np.sqrt(2 / (wavelength * distance))
        
        # Use Fresnel integrals (approximated)
        pattern = np.exp(1j * np.pi * (fresnel_x**2 + fresnel_y**2) / 2)
        intensity = np.abs(pattern)**2
        
        return intensity
    
    @staticmethod
    def calculate_holographic_reconstruction(hologram_data: np.ndarray,
                                          reconstruction_distance: float,
                                          wavelength: float) -> np.ndarray:
        """Reconstruct holographic image from hologram data."""
        # Apply Fresnel transform for reconstruction
        fft_data = scipy.fft.fft2(hologram_data)
        
        # Apply phase factor for reconstruction distance
        ny, nx = hologram_data.shape
        kx = np.fft.fftfreq(nx)
        ky = np.fft.fftfreq(ny)
        KX, KY = np.meshgrid(kx, ky)
        
        phase_factor = np.exp(-1j * np.pi * wavelength * reconstruction_distance * 
                             (KX**2 + KY**2))
        
        reconstructed_fft = fft_data * phase_factor
        reconstructed = scipy.fft.ifft2(reconstructed_fft)
        
        return np.abs(reconstructed)**2
    
    @staticmethod
    def calculate_speckle_reduction(hologram: np.ndarray, 
                                  num_angles: int = 8) -> np.ndarray:
        """Reduce speckle noise in holographic reconstruction."""
        reduced_speckle = np.zeros_like(hologram)
        
        for i in range(num_angles):
            angle = i * 2 * np.pi / num_angles
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                      [np.sin(angle), np.cos(angle)]])
            
            # Apply small rotation and average
            rotated = scipy.ndimage.rotate(hologram, np.degrees(angle), reshape=False)
            reduced_speckle += rotated
        
        return reduced_speckle / num_angles
    
    @staticmethod
    def calculate_holographic_efficiency(input_power: float, output_power: float,
                                       diffraction_orders: List[float]) -> Dict[str, float]:
        """Calculate holographic diffraction efficiency."""
        total_diffracted = sum(diffraction_orders)
        
        return {
            'total_efficiency': output_power / input_power if input_power > 0 else 0,
            'first_order_efficiency': diffraction_orders[1] / input_power if len(diffraction_orders) > 1 and input_power > 0 else 0,
            'zero_order_efficiency': diffraction_orders[0] / input_power if len(diffraction_orders) > 0 and input_power > 0 else 0,
            'higher_order_efficiency': (total_diffracted - sum(diffraction_orders[:2])) / input_power if len(diffraction_orders) > 2 and input_power > 0 else 0
        }

class AdvancedSignalProcessing:
    """Comprehensive signal processing utilities with advanced filtering and analysis."""
    
    @staticmethod
    def design_filter(filter_type: FilterType, cutoff: Union[float, List[float]],
                     sampling_rate: float, order: int = 4,
                     method: str = 'butter') -> Tuple[np.ndarray, np.ndarray]:
        """Design digital filter with specified parameters."""
        nyquist = sampling_rate / 2
        
        if isinstance(cutoff, (int, float)):
            normalized_cutoff = cutoff / nyquist
        else:
            normalized_cutoff = [f / nyquist for f in cutoff]
        
        if method == 'butter':
            if filter_type == FilterType.LOW_PASS:
                b, a = scipy.signal.butter(order, normalized_cutoff, btype='low')
            elif filter_type == FilterType.HIGH_PASS:
                b, a = scipy.signal.butter(order, normalized_cutoff, btype='high')
            elif filter_type == FilterType.BAND_PASS:
                b, a = scipy.signal.butter(order, normalized_cutoff, btype='band')
            elif filter_type == FilterType.BAND_STOP:
                b, a = scipy.signal.butter(order, normalized_cutoff, btype='bandstop')
            else:
                raise ValueError(f"Unsupported filter type: {filter_type}")
        elif method == 'cheby1':
            rp = 1  # Passband ripple
            b, a = scipy.signal.cheby1(order, rp, normalized_cutoff, btype=filter_type.value)
        elif method == 'ellip':
            rp, rs = 1, 60  # Passband and stopband ripple
            b, a = scipy.signal.ellip(order, rp, rs, normalized_cutoff, btype=filter_type.value)
        else:
            raise ValueError(f"Unknown filter design method: {method}")
        
        return b, a
    
    @staticmethod
    def apply_filter(signal: np.ndarray, b: np.ndarray, a: np.ndarray,
                    method: str = 'filtfilt') -> np.ndarray:
        """Apply digital filter to signal."""
        if method == 'filtfilt':
            return scipy.signal.filtfilt(b, a, signal)
        elif method == 'lfilter':
            return scipy.signal.lfilter(b, a, signal)
        else:
            raise ValueError(f"Unknown filtering method: {method}")
    
    @staticmethod
    def spectral_analysis(signal: np.ndarray, sampling_rate: float,
                         window: WindowType = WindowType.HANNING,
                         nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform spectral analysis using Welch's method."""
        f, t, Sxx = scipy.signal.spectrogram(signal, fs=sampling_rate,
                                           window=window.value, nperseg=nperseg)
        return f, t, Sxx
    
    @staticmethod
    def power_spectral_density(signal: np.ndarray, sampling_rate: float,
                              window: WindowType = WindowType.HANNING,
                              nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectral density."""
        f, Pxx = scipy.signal.welch(signal, fs=sampling_rate,
                                  window=window.value, nperseg=nperseg)
        return f, Pxx
    
    @staticmethod
    def cross_correlation(signal1: np.ndarray, signal2: np.ndarray,
                         mode: str = 'full') -> np.ndarray:
        """Calculate cross-correlation between two signals."""
        return scipy.signal.correlate(signal1, signal2, mode=mode)
    
    @staticmethod
    def autocorrelation(signal: np.ndarray, maxlags: Optional[int] = None) -> np.ndarray:
        """Calculate autocorrelation of signal."""
        if maxlags is None:
            maxlags = len(signal) - 1
        
        correlation = scipy.signal.correlate(signal, signal, mode='full')
        mid = len(correlation) // 2
        return correlation[mid-maxlags:mid+maxlags+1]
    
    @staticmethod
    def hilbert_transform(signal: np.ndarray) -> np.ndarray:
        """Apply Hilbert transform to get analytic signal."""
        return scipy.signal.hilbert(signal)
    
    @staticmethod
    def instantaneous_phase(signal: np.ndarray) -> np.ndarray:
        """Calculate instantaneous phase of signal."""
        analytic_signal = scipy.signal.hilbert(signal)
        return np.angle(analytic_signal)
    
    @staticmethod
    def instantaneous_frequency(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Calculate instantaneous frequency of signal."""
        phase = AdvancedSignalProcessing.instantaneous_phase(signal)
        return sampling_rate / (2 * np.pi) * np.gradient(phase)
    
    @staticmethod
    def envelope_detection(signal: np.ndarray) -> np.ndarray:
        """Extract envelope of signal using Hilbert transform."""
        analytic_signal = scipy.signal.hilbert(signal)
        return np.abs(analytic_signal)
    
    @staticmethod
    def zero_crossing_rate(signal: np.ndarray) -> float:
        """Calculate zero crossing rate of signal."""
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        return len(zero_crossings) / len(signal)
    
    @staticmethod
    def spectral_centroid(signal: np.ndarray, sampling_rate: float) -> float:
        """Calculate spectral centroid (brightness) of signal."""
        fft = np.abs(scipy.fft.fft(signal))
        freqs = scipy.fft.fftfreq(len(signal), 1/sampling_rate)
        
        # Only consider positive frequencies
        half_len = len(freqs) // 2
        fft = fft[:half_len]
        freqs = freqs[:half_len]
        
        return np.sum(freqs * fft) / np.sum(fft) if np.sum(fft) > 0 else 0
    
    @staticmethod
    def spectral_rolloff(signal: np.ndarray, sampling_rate: float,
                        rolloff_percent: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        fft = np.abs(scipy.fft.fft(signal))**2
        freqs = scipy.fft.fftfreq(len(signal), 1/sampling_rate)
        
        # Only consider positive frequencies
        half_len = len(freqs) // 2
        fft = fft[:half_len]
        freqs = freqs[:half_len]
        
        cumulative_sum = np.cumsum(fft)
        rolloff_threshold = rolloff_percent * cumulative_sum[-1]
        
        rolloff_index = np.where(cumulative_sum >= rolloff_threshold)[0]
        return freqs[rolloff_index[0]] if len(rolloff_index) > 0 else freqs[-1]
    
    @staticmethod
    def mfcc_features(signal: np.ndarray, sampling_rate: float,
                     n_mfcc: int = 13, n_fft: int = 2048) -> np.ndarray:
        """Extract Mel-frequency cepstral coefficients."""
        # This is a simplified version - would normally use librosa
        # Apply windowing
        windowed = signal * np.hanning(len(signal))
        
        # FFT
        fft = scipy.fft.fft(windowed, n_fft)
        magnitude = np.abs(fft[:n_fft//2])
        
        # Mel filter bank (simplified)
        mel_filters = AdvancedSignalProcessing._create_mel_filterbank(n_fft//2, sampling_rate)
        mel_spectrum = np.dot(magnitude, mel_filters.T)
        
        # Log and DCT
        log_mel = np.log(mel_spectrum + 1e-10)
        mfcc = scipy.fft.dct(log_mel)[:n_mfcc]
        
        return mfcc
    
    @staticmethod
    def _create_mel_filterbank(n_freq: int, sampling_rate: float, n_mels: int = 26) -> np.ndarray:
        """Create Mel filter bank (simplified implementation)."""
        def mel_scale(freq):
            return 2595 * np.log10(1 + freq / 700)
        
        def inverse_mel_scale(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create mel scale points
        mel_min = mel_scale(0)
        mel_max = mel_scale(sampling_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        freq_points = inverse_mel_scale(mel_points)
        
        # Convert to frequency bin indices
        freq_bins = np.floor((n_freq + 1) * freq_points / (sampling_rate / 2))
        
        # Create filter bank
        filters = np.zeros((n_mels, n_freq))
        for m in range(1, n_mels + 1):
            for k in range(int(freq_bins[m-1]), int(freq_bins[m+1])):
                if k < int(freq_bins[m]):
                    filters[m-1, k] = (k - freq_bins[m-1]) / (freq_bins[m] - freq_bins[m-1])
                else:
                    filters[m-1, k] = (freq_bins[m+1] - k) / (freq_bins[m+1] - freq_bins[m])
        
        return filters
    
    @staticmethod
    def adaptive_filter_lms(input_signal: np.ndarray, desired_signal: np.ndarray,
                           filter_order: int, mu: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Least Mean Squares adaptive filter."""
        n_samples = len(input_signal)
        weights = np.zeros(filter_order)
        output = np.zeros(n_samples)
        error = np.zeros(n_samples)
        
        for i in range(filter_order, n_samples):
            x = input_signal[i-filter_order:i][::-1]  # Reverse for convolution
            output[i] = np.dot(weights, x)
            error[i] = desired_signal[i] - output[i]
            weights += mu * error[i] * x
        
        return output, error
    
    @staticmethod
    def wavelet_transform(signal: np.ndarray, wavelet: str = 'db4',
                         levels: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Discrete wavelet transform (simplified implementation)."""
        import pywt
        coeffs = pywt.wavedec(signal, wavelet, level=levels)
        return coeffs[0], coeffs[1:]  # Approximation and detail coefficients
    
    @staticmethod
    def chirp_z_transform(signal: np.ndarray, m: int, w: complex, a: complex) -> np.ndarray:
        """Chirp Z-transform for arbitrary frequency resolution."""
        n = len(signal)
        
        # Create chirp sequences
        n_seq = np.arange(n)
        m_seq = np.arange(m)
        
        chirp1 = w ** (-n_seq**2 / 2)
        chirp2 = w ** (m_seq**2 / 2) * a ** (-m_seq)
        
        # Apply first chirp
        x_chirped = signal * chirp1
        
        # Convolution using FFT
        next_pow2 = 2 ** int(np.ceil(np.log2(n + m - 1)))
        x_fft = scipy.fft.fft(x_chirped, next_pow2)
        
        # Create convolution kernel
        h = w ** (np.arange(-(n-1), m)**2 / 2)
        h_fft = scipy.fft.fft(h, next_pow2)
        
        # Convolution and extract result
        result_fft = x_fft * h_fft
        result = scipy.fft.ifft(result_fft)
        result = result[n-1:n-1+m]
        
        # Apply second chirp
        return result * chirp2

class AdvancedOptimization:
    """Comprehensive optimization algorithms with advanced methods."""
    
    @staticmethod
    def gradient_descent_with_momentum(objective_function: Callable[[np.ndarray], float],
                                     gradient_function: Callable[[np.ndarray], np.ndarray],
                                     initial_params: np.ndarray,
                                     learning_rate: float = 0.01,
                                     momentum: float = 0.9,
                                     max_iterations: int = 1000,
                                     tolerance: float = 1e-6) -> OptimizationResult:
        """Gradient descent with momentum optimization."""
        params = initial_params.copy()
        velocity = np.zeros_like(params)
        start_time = time.time()
        
        for iteration in range(max_iterations):
            gradient = gradient_function(params)
            velocity = momentum * velocity - learning_rate * gradient
            new_params = params + velocity
            
            if np.linalg.norm(new_params - params) < tolerance:
                break
            
            params = new_params
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            success=iteration < max_iterations,
            optimal_parameters=params,
            optimal_value=objective_function(params),
            iterations=iteration + 1,
            function_evaluations=iteration + 1,
            convergence_message="Converged" if iteration < max_iterations else "Max iterations reached",
            execution_time=execution_time,
            gradient_norm=np.linalg.norm(gradient_function(params))
        )
    
    @staticmethod
    def adam_optimizer(objective_function: Callable[[np.ndarray], float],
                      gradient_function: Callable[[np.ndarray], np.ndarray],
                      initial_params: np.ndarray,
                      learning_rate: float = 0.001,
                      beta1: float = 0.9,
                      beta2: float = 0.999,
                      epsilon: float = 1e-8,
                      max_iterations: int = 1000,
                      tolerance: float = 1e-6) -> OptimizationResult:
        """Adam optimization algorithm."""
        params = initial_params.copy()
        m = np.zeros_like(params)  # First moment estimate
        v = np.zeros_like(params)  # Second moment estimate
        start_time = time.time()
        
        for iteration in range(1, max_iterations + 1):
            gradient = gradient_function(params)
            
            # Update biased first and second moment estimates
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            
            # Compute bias-corrected estimates
            m_hat = m / (1 - beta1**iteration)
            v_hat = v / (1 - beta2**iteration)
            
            # Update parameters
            params_update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            new_params = params - params_update
            
            if np.linalg.norm(params_update) < tolerance:
                break
            
            params = new_params
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            success=iteration < max_iterations,
            optimal_parameters=params,
            optimal_value=objective_function(params),
            iterations=iteration,
            function_evaluations=iteration,
            convergence_message="Converged" if iteration < max_iterations else "Max iterations reached",
            execution_time=execution_time,
            gradient_norm=np.linalg.norm(gradient_function(params))
        )
    
    @staticmethod
    def differential_evolution(objective_function: Callable[[np.ndarray], float],
                             bounds: List[Tuple[float, float]],
                             population_size: Optional[int] = None,
                             mutation_factor: float = 0.8,
                             crossover_rate: float = 0.9,
                             max_iterations: int = 1000,
                             tolerance: float = 1e-6,
                             seed: Optional[int] = None) -> OptimizationResult:
        """Differential Evolution optimization algorithm."""
        if seed is not None:
            np.random.seed(seed)
        
        dimensions = len(bounds)
        if population_size is None:
            population_size = max(10, 10 * dimensions)
        
        # Initialize population
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        population = np.random.uniform(lower_bounds, upper_bounds, 
                                     (population_size, dimensions))
        
        fitness = np.array([objective_function(individual) for individual in population])
        best_index = np.argmin(fitness)
        best_params = population[best_index].copy()
        best_fitness = fitness[best_index]
        
        start_time = time.time()
        function_evaluations = population_size
        
        for iteration in range(max_iterations):
            for i in range(population_size):
                # Select three random individuals (different from current)
                candidates = [j for j in range(population_size) if j != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Mutation
                mutant = population[a] + mutation_factor * (population[b] - population[c])
                
                # Ensure bounds
                mutant = np.clip(mutant, lower_bounds, upper_bounds)
                
                # Crossover
                crossover_mask = np.random.rand(dimensions) < crossover_rate
                # Ensure at least one dimension is crossed over
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(dimensions)] = True
                
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                trial_fitness = objective_function(trial)
                function_evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_params = trial.copy()
                        best_fitness = trial_fitness
            
            # Check convergence
            if np.std(fitness) < tolerance:
                break
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            success=iteration < max_iterations - 1,
            optimal_parameters=best_params,
            optimal_value=best_fitness,
            iterations=iteration + 1,
            function_evaluations=function_evaluations,
            convergence_message="Converged" if iteration < max_iterations - 1 else "Max iterations reached",
            execution_time=execution_time
        )
    
    @staticmethod
    def bayesian_optimization(objective_function: Callable[[np.ndarray], float],
                            bounds: List[Tuple[float, float]],
                            n_initial_points: int = 5,
                            n_iterations: int = 25,
                            acquisition_function: str = 'ei',
                            exploration_weight: float = 0.01) -> OptimizationResult:
        """Bayesian optimization using Gaussian Process surrogate."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        
        dimensions = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        # Generate initial points
        X_sample = np.random.uniform(lower_bounds, upper_bounds, 
                                   (n_initial_points, dimensions))
        y_sample = np.array([objective_function(x) for x in X_sample])
        
        # Initialize Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        start_time = time.time()
        function_evaluations = n_initial_points
        best_index = np.argmin(y_sample)
        best_params = X_sample[best_index]
        best_value = y_sample[best_index]
        
        for iteration in range(n_iterations):
            # Fit GP to observed data
            gp.fit(X_sample, y_sample)
            
            # Optimize acquisition function
            def acquisition(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                
                if acquisition_function == 'ei':  # Expected Improvement
                    improvement = best_value - mu
                    z = improvement / (sigma + 1e-9)
                    ei = improvement * scipy.stats.norm.cdf(z) + sigma * scipy.stats.norm.pdf(z)
                    return -ei[0]  # Minimize (negative EI)
                elif acquisition_function == 'ucb':  # Upper Confidence Bound
                    return -(mu - exploration_weight * sigma)[0]
                else:
                    raise ValueError(f"Unknown acquisition function: {acquisition_function}")
            
            # Find next point to evaluate
            result = scipy.optimize.differential_evolution(acquisition, bounds,
                                                         maxiter=100, seed=42)
            next_point = result.x
            
            # Evaluate objective at next point
            next_value = objective_function(next_point)
            function_evaluations += 1
            
            # Update dataset
            X_sample = np.vstack([X_sample, next_point])
            y_sample = np.append(y_sample, next_value)
            
            # Update best point
            if next_value < best_value:
                best_params = next_point.copy()
                best_value = next_value
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            success=True,
            optimal_parameters=best_params,
            optimal_value=best_value,
            iterations=n_iterations,
            function_evaluations=function_evaluations,
            convergence_message="Bayesian optimization completed",
            execution_time=execution_time
        )
    
    @staticmethod
    def multi_objective_optimization(objective_functions: List[Callable[[np.ndarray], float]],
                                   bounds: List[Tuple[float, float]],
                                   population_size: int = 100,
                                   max_generations: int = 250) -> Dict[str, Any]:
        """Multi-objective optimization using NSGA-II inspired algorithm."""
        dimensions = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        # Initialize population
        population = np.random.uniform(lower_bounds, upper_bounds,
                                     (population_size, dimensions))
        
        start_time = time.time()
        
        def evaluate_population(pop):
            """Evaluate population on all objectives."""
            return np.array([[obj_func(individual) for obj_func in objective_functions]
                           for individual in pop])
        
        def fast_non_dominated_sort(fitness_values):
            """Fast non-dominated sorting."""
            num_objectives = fitness_values.shape[1]
            num_individuals = fitness_values.shape[0]
            
            # Initialize arrays
            domination_count = np.zeros(num_individuals, dtype=int)
            dominated_solutions = [[] for _ in range(num_individuals)]
            fronts = [[]]
            
            # Calculate domination
            for i in range(num_individuals):
                for j in range(num_individuals):
                    if i != j:
                        if np.all(fitness_values[i] <= fitness_values[j]) and \
                           np.any(fitness_values[i] < fitness_values[j]):
                            # i dominates j
                            dominated_solutions[i].append(j)
                        elif np.all(fitness_values[j] <= fitness_values[i]) and \
                             np.any(fitness_values[j] < fitness_values[i]):
                            # j dominates i
                            domination_count[i] += 1
                
                if domination_count[i] == 0:
                    fronts[0].append(i)
            
            # Build subsequent fronts
            front_index = 0
            while len(fronts[front_index]) > 0:
                next_front = []
                for i in fronts[front_index]:
                    for j in dominated_solutions[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                front_index += 1
                fronts.append(next_front)
            
            return fronts[:-1]  # Remove empty last front
        
        def calculate_crowding_distance(fitness_values, front):
            """Calculate crowding distance for diversity preservation."""
            if len(front) <= 2:
                return {i: float('inf') for i in front}
            
            distances = {i: 0 for i in front}
            num_objectives = fitness_values.shape[1]
            
            for obj in range(num_objectives):
                front_sorted = sorted(front, key=lambda x: fitness_values[x, obj])
                
                # Boundary points get infinite distance
                distances[front_sorted[0]] = float('inf')
                distances[front_sorted[-1]] = float('inf')
                
                obj_range = fitness_values[front_sorted[-1], obj] - fitness_values[front_sorted[0], obj]
                
                if obj_range > 0:
                    for i in range(1, len(front_sorted) - 1):
                        distances[front_sorted[i]] += \
                            (fitness_values[front_sorted[i+1], obj] - 
                             fitness_values[front_sorted[i-1], obj]) / obj_range
            
            return distances
        
        # Main evolution loop
        for generation in range(max_generations):
            fitness = evaluate_population(population)
            fronts = fast_non_dominated_sort(fitness)
            
            # Create new population
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= population_size:
                    new_population.extend(front)
                else:
                    # Need to select subset based on crowding distance
                    remaining_slots = population_size - len(new_population)
                    distances = calculate_crowding_distance(fitness, front)
                    selected = sorted(front, key=lambda x: distances[x], reverse=True)
                    new_population.extend(selected[:remaining_slots])
                    break
            
            # Generate offspring (simplified)
            selected_indices = new_population[:population_size]
            new_pop = population[selected_indices]
            
            # Simple mutation
            for i in range(population_size):
                if np.random.random() < 0.1:  # Mutation probability
                    mutation = np.random.normal(0, 0.1, dimensions)
                    new_pop[i] = np.clip(new_pop[i] + mutation, lower_bounds, upper_bounds)
            
            population = new_pop
        
        # Final evaluation and extract Pareto front
        final_fitness = evaluate_population(population)
        final_fronts = fast_non_dominated_sort(final_fitness)
        pareto_indices = final_fronts[0] if final_fronts else []
        
        execution_time = time.time() - start_time
        
        return {
            'pareto_front_solutions': population[pareto_indices] if pareto_indices else np.array([]),
            'pareto_front_objectives': final_fitness[pareto_indices] if pareto_indices else np.array([]),
            'all_solutions': population,
            'all_objectives': final_fitness,
            'execution_time': execution_time,
            'generations': max_generations
        }

class ProbabilityDistributions:
    """Advanced probability distribution utilities."""
    
    @staticmethod
    def sample_distribution(dist_type: DistributionType, parameters: Dict[str, float],
                           size: int = 1000, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from specified probability distribution."""
        if random_state is not None:
            np.random.seed(random_state)
        
        if dist_type == DistributionType.NORMAL:
            return np.random.normal(parameters['mean'], parameters['std'], size)
        elif dist_type == DistributionType.UNIFORM:
            return np.random.uniform(parameters['low'], parameters['high'], size)
        elif dist_type == DistributionType.EXPONENTIAL:
            return np.random.exponential(parameters['scale'], size)
        elif dist_type == DistributionType.GAMMA:
            return np.random.gamma(parameters['shape'], parameters['scale'], size)
        elif dist_type == DistributionType.BETA:
            return np.random.beta(parameters['alpha'], parameters['beta'], size)
        elif dist_type == DistributionType.BINOMIAL:
            return np.random.binomial(parameters['n'], parameters['p'], size)
        elif dist_type == DistributionType.POISSON:
            return np.random.poisson(parameters['lam'], size)
        elif dist_type == DistributionType.MULTIVARIATE_NORMAL:
            return np.random.multivariate_normal(parameters['mean'], parameters['cov'], size)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    
    @staticmethod
    def fit_distribution(data: np.ndarray, dist_type: DistributionType) -> Dict[str, float]:
        """Fit distribution parameters to data using maximum likelihood estimation."""
        if dist_type == DistributionType.NORMAL:
            return {'mean': np.mean(data), 'std': np.std(data, ddof=1)}
        elif dist_type == DistributionType.UNIFORM:
            return {'low': np.min(data), 'high': np.max(data)}
        elif dist_type == DistributionType.EXPONENTIAL:
            return {'scale': np.mean(data)}
        elif dist_type == DistributionType.GAMMA:
            # Method of moments estimation
            mean_data = np.mean(data)
            var_data = np.var(data, ddof=1)
            scale = var_data / mean_data
            shape = mean_data / scale
            return {'shape': shape, 'scale': scale}
        elif dist_type == DistributionType.BETA:
            mean_data = np.mean(data)
            var_data = np.var(data, ddof=1)
            
            # Method of moments
            alpha = mean_data * (mean_data * (1 - mean_data) / var_data - 1)
            beta = (1 - mean_data) * (mean_data * (1 - mean_data) / var_data - 1)
            return {'alpha': max(0.1, alpha), 'beta': max(0.1, beta)}
        else:
            raise ValueError(f"Parameter fitting not implemented for {dist_type}")
    
    @staticmethod
    def calculate_pdf(x: np.ndarray, dist_type: DistributionType,
                     parameters: Dict[str, float]) -> np.ndarray:
        """Calculate probability density function."""
        if dist_type == DistributionType.NORMAL:
            return scipy.stats.norm.pdf(x, parameters['mean'], parameters['std'])
        elif dist_type == DistributionType.UNIFORM:
            return scipy.stats.uniform.pdf(x, parameters['low'], 
                                         parameters['high'] - parameters['low'])
        elif dist_type == DistributionType.EXPONENTIAL:
            return scipy.stats.expon.pdf(x, scale=parameters['scale'])
        elif dist_type == DistributionType.GAMMA:
            return scipy.stats.gamma.pdf(x, parameters['shape'], scale=parameters['scale'])
        elif dist_type == DistributionType.BETA:
            return scipy.stats.beta.pdf(x, parameters['alpha'], parameters['beta'])
        else:
            raise ValueError(f"PDF calculation not implemented for {dist_type}")
    
    @staticmethod
    def calculate_cdf(x: np.ndarray, dist_type: DistributionType,
                     parameters: Dict[str, float]) -> np.ndarray:
        """Calculate cumulative distribution function."""
        if dist_type == DistributionType.NORMAL:
            return scipy.stats.norm.cdf(x, parameters['mean'], parameters['std'])
        elif dist_type == DistributionType.UNIFORM:
            return scipy.stats.uniform.cdf(x, parameters['low'], 
                                         parameters['high'] - parameters['low'])
        elif dist_type == DistributionType.EXPONENTIAL:
            return scipy.stats.expon.cdf(x, scale=parameters['scale'])
        elif dist_type == DistributionType.GAMMA:
            return scipy.stats.gamma.cdf(x, parameters['shape'], scale=parameters['scale'])
        elif dist_type == DistributionType.BETA:
            return scipy.stats.beta.cdf(x, parameters['alpha'], parameters['beta'])
        else:
            raise ValueError(f"CDF calculation not implemented for {dist_type}")
    
    @staticmethod
    def monte_carlo_integration(func: Callable[[np.ndarray], float],
                              bounds: List[Tuple[float, float]],
                              n_samples: int = 100000,
                              confidence_level: float = 0.95) -> Dict[str, float]:
        """Monte Carlo integration with confidence intervals."""
        dimensions = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        # Generate random samples
        samples = np.random.uniform(lower_bounds, upper_bounds, (n_samples, dimensions))
        
        # Evaluate function at samples
        function_values = np.array([func(sample) for sample in samples])
        
        # Calculate volume of integration region
        volume = np.prod(upper_bounds - lower_bounds)
        
        # Estimate integral
        integral_estimate = volume * np.mean(function_values)
        
        # Calculate confidence interval
        std_error = volume * np.std(function_values) / np.sqrt(n_samples)
        z_score = scipy.stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * std_error
        
        return {
            'integral_estimate': integral_estimate,
            'standard_error': std_error,
            'confidence_interval_lower': integral_estimate - margin_of_error,
            'confidence_interval_upper': integral_estimate + margin_of_error,
            'confidence_level': confidence_level
        }
    
    @staticmethod
    def bootstrap_statistics(data: np.ndarray, statistic: Callable[[np.ndarray], float],
                           n_bootstrap: int = 1000,
                           confidence_level: float = 0.95) -> Dict[str, float]:
        """Bootstrap estimation of statistic with confidence intervals."""
        n_samples = len(data)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'original_statistic': statistic(data),
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'confidence_interval_lower': np.percentile(bootstrap_stats, lower_percentile),
            'confidence_interval_upper': np.percentile(bootstrap_stats, upper_percentile),
            'confidence_level': confidence_level
        }

class AdvancedStatistics:
    """Advanced statistical analysis and hypothesis testing."""
    
    @staticmethod
    def hypothesis_test_ttest(sample1: np.ndarray, sample2: Optional[np.ndarray] = None,
                             null_hypothesis_mean: float = 0.0,
                             alternative: str = 'two-sided',
                             alpha: float = 0.05) -> Dict[str, Any]:
        """Perform t-test with comprehensive results."""
        if sample2 is None:
            # One-sample t-test
            statistic, p_value = scipy.stats.ttest_1samp(sample1, null_hypothesis_mean)
            test_type = 'one_sample'
            df = len(sample1) - 1
        else:
            # Two-sample t-test
            statistic, p_value = scipy.stats.ttest_ind(sample1, sample2)
            test_type = 'two_sample'
            df = len(sample1) + len(sample2) - 2
        
        # Adjust p-value for one-sided tests
        if alternative == 'greater':
            p_value = p_value / 2 if statistic > 0 else 1 - p_value / 2
        elif alternative == 'less':
            p_value = p_value / 2 if statistic < 0 else 1 - p_value / 2
        
        # Calculate effect size (Cohen's d)
        if sample2 is None:
            effect_size = (np.mean(sample1) - null_hypothesis_mean) / np.std(sample1, ddof=1)
        else:
            pooled_std = np.sqrt(((len(sample1) - 1) * np.var(sample1, ddof=1) +
                                (len(sample2) - 1) * np.var(sample2, ddof=1)) / df)
            effect_size = (np.mean(sample1) - np.mean(sample2)) / pooled_std
        
        return {
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'effect_size_cohens_d': effect_size,
            'significant': p_value < alpha,
            'alpha': alpha,
            'alternative': alternative
        }
    
    @staticmethod
    def hypothesis_test_chi_square(observed: np.ndarray, 
                                  expected: Optional[np.ndarray] = None,
                                  alpha: float = 0.05) -> Dict[str, Any]:
        """Perform chi-square goodness of fit test."""
        if expected is None:
            # Uniform distribution expected
            expected = np.full_like(observed, np.mean(observed))
        
        statistic, p_value = scipy.stats.chisquare(observed, expected)
        df = len(observed) - 1
        
        # Calculate Cramér's V (effect size)
        n = np.sum(observed)
        cramers_v = np.sqrt(statistic / (n * (min(len(observed), len(expected)) - 1)))
        
        return {
            'test_type': 'chi_square_goodness_of_fit',
            'statistic': statistic,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'effect_size_cramers_v': cramers_v,
            'significant': p_value < alpha,
            'alpha': alpha
        }
    
    @staticmethod
    def anova_one_way(groups: List[np.ndarray], alpha: float = 0.05) -> Dict[str, Any]:
        """Perform one-way ANOVA."""
        statistic, p_value = scipy.stats.f_oneway(*groups)
        
        # Calculate degrees of freedom
        k = len(groups)  # Number of groups
        n = sum(len(group) for group in groups)  # Total sample size
        df_between = k - 1
        df_within = n - k
        
        # Calculate effect size (eta-squared)
        ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(groups)))**2 
                        for group in groups)
        ss_total = sum(np.sum((group - np.mean(np.concatenate(groups)))**2) for group in groups)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'test_type': 'one_way_anova',
            'f_statistic': statistic,
            'p_value': p_value,
            'df_between': df_between,
            'df_within': df_within,
            'effect_size_eta_squared': eta_squared,
            'significant': p_value < alpha,
            'alpha': alpha
        }
    
    @staticmethod
    def correlation_analysis(x: np.ndarray, y: np.ndarray,
                           method: str = 'pearson') -> Dict[str, Any]:
        """Comprehensive correlation analysis."""
        if method == 'pearson':
            correlation, p_value = scipy.stats.pearsonr(x, y)
        elif method == 'spearman':
            correlation, p_value = scipy.stats.spearmanr(x, y)
        elif method == 'kendall':
            correlation, p_value = scipy.stats.kendalltau(x, y)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Calculate confidence interval for Pearson correlation
        if method == 'pearson':
            n = len(x)
            z = np.arctanh(correlation)
            se = 1 / np.sqrt(n - 3)
            z_critical = scipy.stats.norm.ppf(0.975)  # 95% confidence
            
            z_lower = z - z_critical * se
            z_upper = z + z_critical * se
            
            ci_lower = np.tanh(z_lower)
            ci_upper = np.tanh(z_upper)
        else:
            ci_lower = ci_upper = None
        
        return {
            'method': method,
            'correlation': correlation,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'sample_size': len(x)
        }
    
    @staticmethod
    def linear_regression_analysis(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Comprehensive linear regression analysis."""
        # Add intercept term
        X = np.column_stack([np.ones(len(x)), x])
        
        # Calculate coefficients using normal equation
        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        intercept, slope = coefficients
        
        # Calculate predictions and residuals
        y_pred = X @ coefficients
        residuals = y - y_pred
        
        # Calculate R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate standard errors
        n = len(x)
        mse = ss_res / (n - 2)
        var_coeff = mse * np.linalg.inv(X.T @ X)
        se_intercept = np.sqrt(var_coeff[0, 0])
        se_slope = np.sqrt(var_coeff[1, 1])
        
        # Calculate t-statistics and p-values
        t_intercept = intercept / se_intercept if se_intercept > 0 else 0
        t_slope = slope / se_slope if se_slope > 0 else 0
        
        p_intercept = 2 * (1 - scipy.stats.t.cdf(abs(t_intercept), n - 2))
        p_slope = 2 * (1 - scipy.stats.t.cdf(abs(t_slope), n - 2))
        
        # Calculate F-statistic for overall model
        ss_reg = np.sum((y_pred - np.mean(y))**2)
        f_statistic = (ss_reg / 1) / (ss_res / (n - 2)) if ss_res > 0 else 0
        p_f = 1 - scipy.stats.f.cdf(f_statistic, 1, n - 2)
        
        return {
            'coefficients': {
                'intercept': intercept,
                'slope': slope
            },
            'standard_errors': {
                'intercept': se_intercept,
                'slope': se_slope
            },
            'p_values': {
                'intercept': p_intercept,
                'slope': p_slope
            },
            'r_squared': r_squared,
            'adjusted_r_squared': 1 - (1 - r_squared) * (n - 1) / (n - 2),
            'f_statistic': f_statistic,
            'f_p_value': p_f,
            'residuals': residuals,
            'predictions': y_pred,
            'mse': mse
        }

class GeometricCalculations:
    """Advanced geometric calculations for 3D space and holographic positioning."""
    
    @staticmethod
    def calculate_distance_3d(point1: Vector3D, point2: Vector3D) -> float:
        """Calculate Euclidean distance between two 3D points."""
        return (point1 - point2).magnitude()
    
    @staticmethod
    def calculate_triangle_area(p1: Vector3D, p2: Vector3D, p3: Vector3D) -> float:
        """Calculate area of triangle formed by three points."""
        edge1 = p2 - p1
        edge2 = p3 - p1
        cross = edge1.cross(edge2)
        return cross.magnitude() / 2.0
    
    @staticmethod
    def calculate_tetrahedron_volume(p1: Vector3D, p2: Vector3D, 
                                   p3: Vector3D, p4: Vector3D) -> float:
        """Calculate volume of tetrahedron formed by four points."""
        edge1 = p2 - p1
        edge2 = p3 - p1
        edge3 = p4 - p1
        
        # Volume = |det(edge1, edge2, edge3)| / 6
        matrix = np.array([edge1.to_array(), edge2.to_array(), edge3.to_array()])
        return abs(np.linalg.det(matrix)) / 6.0
    
    @staticmethod
    def point_in_polygon_2d(point: Tuple[float, float], 
                           polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    @staticmethod
    def convex_hull_2d(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Calculate convex hull using Graham's scan algorithm."""
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        points = sorted(set(points))
        if len(points) <= 1:
            return points
        
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        return lower[:-1] + upper[:-1]
    
    @staticmethod
    def closest_point_on_line(point: Vector3D, line_start: Vector3D, 
                             line_end: Vector3D) -> Vector3D:
        """Find closest point on line segment to given point."""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_length_sq = line_vec.magnitude_squared()
        if line_length_sq == 0:
            return line_start
        
        t = max(0, min(1, point_vec.dot(line_vec) / line_length_sq))
        return line_start + line_vec * t
    
    @staticmethod
    def plane_from_points(p1: Vector3D, p2: Vector3D, p3: Vector3D) -> Tuple[Vector3D, float]:
        """Calculate plane equation from three points."""
        v1 = p2 - p1
        v2 = p3 - p1
        normal = v1.cross(v2).normalize()
        d = -normal.dot(p1)
        return normal, d
    
    @staticmethod
    def line_plane_intersection(line_start: Vector3D, line_direction: Vector3D,
                               plane_normal: Vector3D, plane_d: float) -> Optional[Vector3D]:
        """Calculate intersection of line with plane."""
        denominator = line_direction.dot(plane_normal)
        
        if abs(denominator) < 1e-10:  # Line is parallel to plane
            return None
        
        t = -(line_start.dot(plane_normal) + plane_d) / denominator
        return line_start + line_direction * t
    
    @staticmethod
    def sphere_ray_intersection(ray_origin: Vector3D, ray_direction: Vector3D,
                               sphere_center: Vector3D, sphere_radius: float) -> List[float]:
        """Calculate intersection points of ray with sphere."""
        oc = ray_origin - sphere_center
        
        a = ray_direction.dot(ray_direction)
        b = 2.0 * oc.dot(ray_direction)
        c = oc.dot(oc) - sphere_radius * sphere_radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return []  # No intersection
        elif discriminant == 0:
            return [-b / (2 * a)]  # One intersection
        else:
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2 * a)
            t2 = (-b + sqrt_discriminant) / (2 * a)
            return [t1, t2]
    
    @staticmethod
    def calculate_bounding_box(points: List[Vector3D]) -> Tuple[Vector3D, Vector3D]:
        """Calculate axis-aligned bounding box for set of points."""
        if not points:
            return Vector3D.zero(), Vector3D.zero()
        
        min_point = Vector3D(
            min(p.x for p in points),
            min(p.y for p in points),
            min(p.z for p in points)
        )
        
        max_point = Vector3D(
            max(p.x for p in points),
            max(p.y for p in points),
            max(p.z for p in points)
        )
        
        return min_point, max_point

class MathPerformanceProfiler:
    """Performance profiling and optimization for mathematical operations."""
    
    def __init__(self):
        self.operation_stats = defaultdict(lambda: MathPerformanceStats())
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = threading.Lock()
        self.memory_tracker = None
        
    def profile_operation(self, operation_name: str):
        """Decorator for profiling mathematical operations."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                
                # Enable memory tracking if available
                if self.memory_tracker is None and tracemalloc.is_tracing():
                    start_memory = tracemalloc.get_traced_memory()[0]
                else:
                    start_memory = 0
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Calculate execution time
                    execution_time = time.perf_counter() - start_time
                    
                    # Calculate memory usage
                    if start_memory > 0 and tracemalloc.is_tracing():
                        end_memory = tracemalloc.get_traced_memory()[0]
                        memory_usage = end_memory - start_memory
                    else:
                        memory_usage = 0
                    
                    # Update statistics
                    with self._lock:
                        self.operation_stats[operation_name].update(execution_time, memory_usage)
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.perf_counter() - start_time
                    with self._lock:
                        self.operation_stats[operation_name].update(execution_time, 0)
                    raise
            
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Dict[str, Any]]:
        """Generate comprehensive performance report."""
        report = {}
        
        with self._lock:
            for operation_name, stats in self.operation_stats.items():
                report[operation_name] = {
                    'operation_count': stats.operation_count,
                    'total_time': stats.total_execution_time,
                    'average_time': stats.average_execution_time,
                    'min_time': stats.min_execution_time if stats.min_execution_time != float('inf') else 0,
                    'max_time': stats.max_execution_time,
                    'total_memory': stats.memory_usage_bytes,
                    'cache_hit_rate': (self.cache_hits / (self.cache_hits + self.cache_misses)
                                     if (self.cache_hits + self.cache_misses) > 0 else 0.0)
                }
        
        return report
    
    def clear_statistics(self):
        """Clear all performance statistics."""
        with self._lock:
            self.operation_stats.clear()
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
    
    def memoize(self, func):
        """Memoization decorator for expensive mathematical functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            with self._lock:
                if key in self.cache:
                    self.cache_hits += 1
                    return self.cache[key]
                
                self.cache_misses += 1
                result = func(*args, **kwargs)
                
                # Limit cache size
                if len(self.cache) > 10000:
                    # Remove oldest entries (simple FIFO)
                    keys_to_remove = list(self.cache.keys())[:1000]
                    for k in keys_to_remove:
                        del self.cache[k]
                
                self.cache[key] = result
                return result
        
        return wrapper

# Global instances for module-level functionality
global_math_profiler = MathPerformanceProfiler()
global_performance_stats = defaultdict(lambda: MathPerformanceStats())

# Utility functions and decorators
def profile_math_operation(operation_name: str):
    """Decorator to profile mathematical operations."""
    return global_math_profiler.profile_operation(operation_name)

def memoize_math_function(func):
    """Decorator to memoize expensive mathematical functions."""
    return global_math_profiler.memoize(func)

@contextmanager
def math_performance_context(track_memory: bool = True):
    """Context manager for tracking mathematical operation performance."""
    if track_memory and not tracemalloc.is_tracing():
        tracemalloc.start()
        started_tracemalloc = True
    else:
        started_tracemalloc = False
    
    start_time = time.perf_counter()
    start_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        print(f"Mathematical operation completed:")
        print(f"  Execution time: {execution_time:.6f} seconds")
        print(f"  Memory usage: {memory_usage} bytes ({memory_usage / 1024:.2f} KB)")
        
        if started_tracemalloc:
            tracemalloc.stop()

# Advanced utility functions
def create_rotation_matrix_from_vectors(from_vector: Vector3D, to_vector: Vector3D) -> np.ndarray:
    """Create rotation matrix to rotate from one vector to another."""
    from_norm = from_vector.normalize()
    to_norm = to_vector.normalize()
    
    # Calculate rotation axis
    cross_product = from_norm.cross(to_norm)
    
    # Check if vectors are parallel
    if cross_product.magnitude() < 1e-10:
        if from_norm.dot(to_norm) > 0:
            return np.eye(3)  # Same direction
        else:
            # Opposite direction - rotate 180 degrees around any perpendicular axis
            if abs(from_norm.x) < 0.9:
                axis = Vector3D(1, 0, 0).cross(from_norm).normalize()
            else:
                axis = Vector3D(0, 1, 0).cross(from_norm).normalize()
            return Quaternion.from_axis_angle(axis, math.pi).to_rotation_matrix()
    
    # Calculate rotation angle
    angle = math.acos(max(-1.0, min(1.0, from_norm.dot(to_norm))))
    axis = cross_product.normalize()
    
    return Quaternion.from_axis_angle(axis, angle).to_rotation_matrix()

def solve_cubic_equation(a: float, b: float, c: float, d: float) -> List[complex]:
    """Solve cubic equation ax³ + bx² + cx + d = 0."""
    if abs(a) < 1e-10:
        # Quadratic equation
        if abs(b) < 1e-10:
            # Linear equation
            if abs(c) < 1e-10:
                return []
            return [-d / c]
        else:
            discriminant = c**2 - 4*b*d
            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                return [(-c + sqrt_disc) / (2*b), (-c - sqrt_disc) / (2*b)]
            else:
                sqrt_disc = math.sqrt(-discriminant)
                return [complex(-c / (2*b), sqrt_disc / (2*b)),
                       complex(-c / (2*b), -sqrt_disc / (2*b))]
    
    # Normalize coefficients
    b /= a
    c /= a
    d /= a
    
    # Use Cardano's method
    p = c - b**2 / 3
    q = 2*b**3/27 - b*c/3 + d
    
    discriminant = q**2/4 + p**3/27
    
    if discriminant > 0:
        # One real root
        sqrt_disc = math.sqrt(discriminant)
        u = (-q/2 + sqrt_disc)**(1/3) if (-q/2 + sqrt_disc) >= 0 else -(abs(-q/2 + sqrt_disc)**(1/3))
        v = (-q/2 - sqrt_disc)**(1/3) if (-q/2 - sqrt_disc) >= 0 else -(abs(-q/2 - sqrt_disc)**(1/3))
        
        root1 = u + v - b/3
        
        # Two complex conjugate roots
        real_part = -(u + v)/2 - b/3
        imag_part = (u - v) * math.sqrt(3)/2
        
        return [root1, complex(real_part, imag_part), complex(real_part, -imag_part)]
    
    elif discriminant == 0:
        # Three real roots, at least two equal
        if abs(q) < 1e-10:
            # Triple root
            return [-b/3, -b/3, -b/3]
        else:
            # One single and one double root
            u = (-q/2)**(1/3) if (-q/2) >= 0 else -(abs(-q/2)**(1/3))
            return [2*u - b/3, -u - b/3, -u - b/3]
    
    else:
        # Three distinct real roots
        rho = math.sqrt(-p**3/27)
        theta = math.acos(-q/2 / rho)
        
        roots = []
        for k in range(3):
            root = 2 * (rho**(1/3)) * math.cos((theta + 2*k*math.pi)/3) - b/3
            roots.append(root)
        
        return roots

def solve_quartic_equation(a: float, b: float, c: float, d: float, e: float) -> List[complex]:
    """Solve quartic equation ax⁴ + bx³ + cx² + dx + e = 0 using Ferrari's method."""
    if abs(a) < 1e-10:
        return solve_cubic_equation(b, c, d, e)
    
    # Normalize coefficients
    b /= a
    c /= a
    d /= a
    e /= a
    
    # Depress the quartic (remove cubic term)
    p = c - 3*b**2/8
    q = b**3/8 - b*c/2 + d
    r = -3*b**4/256 + c*b**2/16 - b*d/4 + e
    
    if abs(q) < 1e-10:
        # Biquadratic equation
        discriminant = p**2 - 4*r
        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            y1 = (-p + sqrt_disc) / 2
            y2 = (-p - sqrt_disc) / 2
            
            roots = []
            if y1 >= 0:
                roots.extend([math.sqrt(y1), -math.sqrt(y1)])
            else:
                roots.extend([complex(0, math.sqrt(-y1)), complex(0, -math.sqrt(-y1))])
            
            if y2 >= 0:
                roots.extend([math.sqrt(y2), -math.sqrt(y2)])
            else:
                roots.extend([complex(0, math.sqrt(-y2)), complex(0, -math.sqrt(-y2))])
            
            # Shift back
            return [root - b/4 for root in roots]
    
    # Solve resolvent cubic
    resolvent_roots = solve_cubic_equation(1, -p, -4*r, 4*p*r - q**2)
    
    # Choose a real root of the resolvent cubic
    m = None
    for root in resolvent_roots:
        if isinstance(root, complex) and abs(root.imag) < 1e-10:
            m = root.real
            break
    
    if m is None:
        m = resolvent_roots[0].real if isinstance(resolvent_roots[0], complex) else resolvent_roots[0]
    
    # Calculate intermediate values
    alpha = math.sqrt(2*m - p) if 2*m - p >= 0 else complex(0, math.sqrt(-(2*m - p)))
    
    if abs(alpha) < 1e-10:
        beta = m - r/m if abs(m) > 1e-10 else 0
        gamma = complex(0, math.sqrt(abs(beta))) if beta < 0 else math.sqrt(beta)
        delta = -gamma
    else:
        beta = -q / (2 * alpha)
        gamma = (m + beta) / 2
        delta = (m - beta) / 2
    
    # Calculate the four roots
    roots = []
    
    # Two quadratics to solve
    disc1 = alpha**2 - 4*gamma
    disc2 = (-alpha)**2 - 4*delta
    
    if isinstance(disc1, complex) or disc1 < 0:
        sqrt_disc1 = cmath.sqrt(disc1)
        roots.extend([(-alpha + sqrt_disc1)/2, (-alpha - sqrt_disc1)/2])
    else:
        sqrt_disc1 = math.sqrt(disc1)
        roots.extend([(-alpha + sqrt_disc1)/2, (-alpha - sqrt_disc1)/2])
    
    if isinstance(disc2, complex) or disc2 < 0:
        sqrt_disc2 = cmath.sqrt(disc2)
        roots.extend([(alpha + sqrt_disc2)/2, (alpha - sqrt_disc2)/2])
    else:
        sqrt_disc2 = math.sqrt(disc2)
        roots.extend([(alpha + sqrt_disc2)/2, (alpha - sqrt_disc2)/2])
    
    # Shift back to original equation
    return [root - b/4 for root in roots]

class NumericalIntegration:
    """Advanced numerical integration methods."""
    
    @staticmethod
    def simpson_rule(func: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
        """Simpson's 1/3 rule for numerical integration."""
        if n % 2 == 1:
            n += 1  # Ensure even number of intervals
        
        h = (b - a) / n
        x = a
        sum_val = func(a) + func(b)
        
        for i in range(1, n):
            x = a + i * h
            if i % 2 == 0:
                sum_val += 2 * func(x)
            else:
                sum_val += 4 * func(x)
        
        return sum_val * h / 3
    
    @staticmethod
    def gauss_legendre_quadrature(func: Callable[[float], float], a: float, b: float, 
                                  n: int = 5) -> float:
        """Gauss-Legendre quadrature for high-precision integration."""
        # Gauss-Legendre nodes and weights (for n=5)
        if n == 5:
            nodes = [-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459]
            weights = [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]
        else:
            # For other values of n, use scipy's implementation
            nodes, weights = scipy.special.roots_legendre(n)
        
        # Transform from [-1, 1] to [a, b]
        transformed_nodes = [(b - a) * node / 2 + (a + b) / 2 for node in nodes]
        
        integral = sum(weight * func(x) for weight, x in zip(weights, transformed_nodes))
        return integral * (b - a) / 2
    
    @staticmethod
    def adaptive_simpson(func: Callable[[float], float], a: float, b: float, 
                        tol: float = 1e-6, max_depth: int = 10) -> float:
        """Adaptive Simpson's rule with error control."""
        def adaptive_simpson_recursive(f, a, b, tol, S, fa, fb, fc, depth):
            c = (a + b) / 2
            h = b - a
            d = (a + c) / 2
            e = (c + b) / 2
            fd = f(d)
            fe = f(e)
            
            Sleft = h/12 * (fa + 4*fd + fc)
            Sright = h/12 * (fc + 4*fe + fb)
            S2 = Sleft + Sright
            
            if depth <= 0 or abs(S2 - S) <= 15 * tol:
                return S2 + (S2 - S) / 15
            else:
                return (adaptive_simpson_recursive(f, a, c, tol/2, Sleft, fa, fd, fc, depth-1) +
                       adaptive_simpson_recursive(f, c, b, tol/2, Sright, fc, fe, fb, depth-1))
        
        c = (a + b) / 2
        h = b - a
        fa = func(a)
        fb = func(b)
        fc = func(c)
        S = h/6 * (fa + 4*fc + fb)
        
        return adaptive_simpson_recursive(func, a, b, tol, S, fa, fb, fc, max_depth)
    
    @staticmethod
    def monte_carlo_integration(func: Callable[[np.ndarray], float], 
                               bounds: List[Tuple[float, float]], 
                               n_samples: int = 100000) -> Dict[str, float]:
        """Multi-dimensional Monte Carlo integration."""
        dimensions = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        # Generate random samples
        samples = np.random.uniform(lower_bounds, upper_bounds, (n_samples, dimensions))
        
        # Evaluate function at samples
        function_values = np.array([func(sample) for sample in samples])
        
        # Calculate volume and integral estimate
        volume = np.prod(upper_bounds - lower_bounds)
        integral_estimate = volume * np.mean(function_values)
        
        # Estimate error
        variance = np.var(function_values)
        standard_error = volume * np.sqrt(variance / n_samples)
        
        return {
            'integral': integral_estimate,
            'error_estimate': standard_error,
            'samples_used': n_samples,
            'function_evaluations': n_samples
        }

class NumericalDifferentiation:
    """Numerical differentiation methods with error analysis."""
    
    @staticmethod
    def forward_difference(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
        """Forward difference approximation of derivative."""
        return (func(x + h) - func(x)) / h
    
    @staticmethod
    def backward_difference(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
        """Backward difference approximation of derivative."""
        return (func(x) - func(x - h)) / h
    
    @staticmethod
    def central_difference(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
        """Central difference approximation of derivative (higher accuracy)."""
        return (func(x + h) - func(x - h)) / (2 * h)
    
    @staticmethod
    def five_point_stencil(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
        """Five-point stencil for higher-order accuracy."""
        return (-func(x + 2*h) + 8*func(x + h) - 8*func(x - h) + func(x - 2*h)) / (12 * h)
    
    @staticmethod
    def second_derivative(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
        """Second derivative using central difference."""
        return (func(x + h) - 2*func(x) + func(x - h)) / (h**2)
    
    @staticmethod
    def gradient(func: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Numerical gradient for multivariable functions."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad
    
    @staticmethod
    def jacobian(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, 
                h: float = 1e-5) -> np.ndarray:
        """Numerical Jacobian matrix."""
        f_x = func(x)
        n = len(x)
        m = len(f_x)
        jac = np.zeros((m, n))
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            jac[:, i] = (func(x_plus) - func(x_minus)) / (2 * h)
        
        return jac
    
    @staticmethod
    def hessian(func: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """Numerical Hessian matrix."""
        n = len(x)
        hess = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h
                    hess[i, j] = (func(x_plus) - 2*func(x) + func(x_minus)) / (h**2)
                else:
                    # Off-diagonal elements
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()
                    
                    x_pp[i] += h
                    x_pp[j] += h
                    x_pm[i] += h
                    x_pm[j] -= h
                    x_mp[i] -= h
                    x_mp[j] += h
                    x_mm[i] -= h
                    x_mm[j] -= h
                    
                    hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h**2)
        
        return hess

class NumericalRootFinding:
    """Advanced root finding algorithms."""
    
    @staticmethod
    def newton_raphson(func: Callable[[float], float], 
                      derivative: Callable[[float], float],
                      initial_guess: float, 
                      tolerance: float = 1e-10,
                      max_iterations: int = 100) -> Dict[str, Any]:
        """Newton-Raphson method for finding roots."""
        x = initial_guess
        iterations = 0
        
        for iteration in range(max_iterations):
            fx = func(x)
            fpx = derivative(x)
            
            if abs(fpx) < 1e-15:
                return {
                    'root': x,
                    'iterations': iteration,
                    'converged': False,
                    'message': 'Derivative too small'
                }
            
            x_new = x - fx / fpx
            
            if abs(x_new - x) < tolerance:
                return {
                    'root': x_new,
                    'iterations': iteration + 1,
                    'converged': True,
                    'message': 'Converged successfully'
                }
            
            x = x_new
            iterations = iteration + 1
        
        return {
            'root': x,
            'iterations': iterations,
            'converged': False,
            'message': 'Maximum iterations reached'
        }
    
    @staticmethod
    def secant_method(func: Callable[[float], float],
                     x0: float, x1: float,
                     tolerance: float = 1e-10,
                     max_iterations: int = 100) -> Dict[str, Any]:
        """Secant method for root finding (no derivative required)."""
        for iteration in range(max_iterations):
            f0 = func(x0)
            f1 = func(x1)
            
            if abs(f1 - f0) < 1e-15:
                return {
                    'root': x1,
                    'iterations': iteration,
                    'converged': False,
                    'message': 'Function values too close'
                }
            
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            
            if abs(x2 - x1) < tolerance:
                return {
                    'root': x2,
                    'iterations': iteration + 1,
                    'converged': True,
                    'message': 'Converged successfully'
                }
            
            x0, x1 = x1, x2
        
        return {
            'root': x1,
            'iterations': max_iterations,
            'converged': False,
            'message': 'Maximum iterations reached'
        }
    
    @staticmethod
    def bisection_method(func: Callable[[float], float],
                        a: float, b: float,
                        tolerance: float = 1e-10,
                        max_iterations: int = 100) -> Dict[str, Any]:
        """Bisection method for root finding (guaranteed convergence)."""
        if func(a) * func(b) > 0:
            return {
                'root': None,
                'iterations': 0,
                'converged': False,
                'message': 'Function has same sign at both endpoints'
            }
        
        for iteration in range(max_iterations):
            c = (a + b) / 2
            fc = func(c)
            
            if abs(fc) < tolerance or abs(b - a) / 2 < tolerance:
                return {
                    'root': c,
                    'iterations': iteration + 1,
                    'converged': True,
                    'message': 'Converged successfully'
                }
            
            if func(a) * fc < 0:
                b = c
            else:
                a = c
        
        return {
            'root': (a + b) / 2,
            'iterations': max_iterations,
            'converged': False,
            'message': 'Maximum iterations reached'
        }
    
    @staticmethod
    def brent_method(func: Callable[[float], float],
                    a: float, b: float,
                    tolerance: float = 1e-10,
                    max_iterations: int = 100) -> Dict[str, Any]:
        """Brent's method combining bisection, secant, and inverse quadratic interpolation."""
        fa = func(a)
        fb = func(b)
        
        if fa * fb > 0:
            return {
                'root': None,
                'iterations': 0,
                'converged': False,
                'message': 'Function has same sign at both endpoints'
            }
        
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        c = a
        fc = fa
        mflag = True
        
        for iteration in range(max_iterations):
            if abs(fb) < tolerance:
                return {
                    'root': b,
                    'iterations': iteration + 1,
                    'converged': True,
                    'message': 'Converged successfully'
                }
            
            if fa != fc and fb != fc:
                # Inverse quadratic interpolation
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                    (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                    (c * fa * fb) / ((fc - fa) * (fc - fb))
            else:
                # Secant method
                s = b - fb * (b - a) / (fb - fa)
            
            # Check if bisection should be used instead
            condition1 = not ((3*a + b)/4 < s < b) and not (b < s < (3*a + b)/4)
            condition2 = mflag and abs(s - b) >= abs(b - c) / 2
            condition3 = not mflag and abs(s - b) >= abs(c - a) / 2
            condition4 = mflag and abs(b - c) < tolerance
            condition5 = not mflag and abs(c - a) < tolerance
            
            if condition1 or condition2 or condition3 or condition4 or condition5:
                s = (a + b) / 2
                mflag = True
            else:
                mflag = False
            
            fs = func(s)
            a = b
            fa = fb
            
            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs
            
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            
            c = a
            fc = fa
        
        return {
            'root': b,
            'iterations': max_iterations,
            'converged': False,
            'message': 'Maximum iterations reached'
        }

class InterpolationMethods:
    """Advanced interpolation and approximation methods."""
    
    @staticmethod
    def lagrange_interpolation(x_points: np.ndarray, y_points: np.ndarray, x: float) -> float:
        """Lagrange polynomial interpolation."""
        n = len(x_points)
        result = 0.0
        
        for i in range(n):
            term = y_points[i]
            for j in range(n):
                if i != j:
                    term *= (x - x_points[j]) / (x_points[i] - x_points[j])
            result += term
        
        return result
    
    @staticmethod
    def newton_interpolation(x_points: np.ndarray, y_points: np.ndarray) -> Callable[[float], float]:
        """Newton's divided difference interpolation."""
        n = len(x_points)
        
        # Calculate divided differences table
        divided_diff = np.zeros((n, n))
        divided_diff[:, 0] = y_points
        
        for j in range(1, n):
            for i in range(n - j):
                divided_diff[i, j] = (divided_diff[i+1, j-1] - divided_diff[i, j-1]) / \
                                   (x_points[i+j] - x_points[i])
        
        def newton_poly(x: float) -> float:
            result = divided_diff[0, 0]
            product = 1.0
            
            for i in range(1, n):
                product *= (x - x_points[i-1])
                result += divided_diff[0, i] * product
            
            return result
        
        return newton_poly
    
    @staticmethod
    def spline_interpolation(x_points: np.ndarray, y_points: np.ndarray, 
                           kind: str = 'cubic') -> Callable[[float], float]:
        """Spline interpolation using scipy."""
        from scipy.interpolate import interp1d
        
        spline = interp1d(x_points, y_points, kind=kind, 
                         bounds_error=False, fill_value='extrapolate')
        
        return spline
    
    @staticmethod
    def chebyshev_approximation(func: Callable[[float], float], 
                               a: float, b: float, n: int = 10) -> Callable[[float], float]:
        """Chebyshev polynomial approximation."""
        # Chebyshev nodes in [-1, 1]
        k = np.arange(1, n + 1)
        chebyshev_nodes = np.cos((2 * k - 1) * np.pi / (2 * n))
        
        # Transform to [a, b]
        x_nodes = 0.5 * (b - a) * chebyshev_nodes + 0.5 * (a + b)
        y_nodes = np.array([func(x) for x in x_nodes])
        
        # Calculate Chebyshev coefficients
        coefficients = np.zeros(n)
        for i in range(n):
            for k in range(n):
                coefficients[i] += y_nodes[k] * np.cos(i * np.arccos(chebyshev_nodes[k]))
            coefficients[i] *= 2.0 / n
            if i == 0:
                coefficients[i] /= 2.0
        
        def chebyshev_poly(x: float) -> float:
            # Transform x to [-1, 1]
            t = (2 * x - a - b) / (b - a)
            
            # Evaluate Chebyshev polynomial
            result = coefficients[0] / 2
            T_prev = 1
            T_curr = t
            
            for i in range(1, n):
                result += coefficients[i] * T_curr
                T_next = 2 * t * T_curr - T_prev
                T_prev, T_curr = T_curr, T_next
            
            return result
        
        return chebyshev_poly
    
    @staticmethod
    def rational_interpolation(x_points: np.ndarray, y_points: np.ndarray, 
                             m: int = None, n: int = None) -> Callable[[float], float]:
        """Rational function interpolation (Padé approximation)."""
        if m is None and n is None:
            total_points = len(x_points)
            m = total_points // 2
            n = total_points - m - 1
        
        # Set up linear system for rational function coefficients
        # P(x)/Q(x) where P has degree m and Q has degree n
        A = np.zeros((len(x_points), m + n + 1))
        b = y_points.copy()
        
        for i, (xi, yi) in enumerate(zip(x_points, y_points)):
            # Coefficients for numerator polynomial P(x)
            for j in range(m + 1):
                A[i, j] = xi ** j
            
            # Coefficients for denominator polynomial Q(x) (negative terms)
            for j in range(1, n + 1):
                A[i, m + j] = -yi * (xi ** j)
        
        # Solve for coefficients
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        
        p_coeffs = coeffs[:m+1]
        q_coeffs = np.concatenate([[1], coeffs[m+1:]])
        
        def rational_func(x: float) -> float:
            numerator = sum(c * (x ** i) for i, c in enumerate(p_coeffs))
            denominator = sum(c * (x ** i) for i, c in enumerate(q_coeffs))
            return numerator / denominator if abs(denominator) > 1e-15 else float('inf')
        
        return rational_func

class AdvancedNumericalMethods:
    """Additional advanced numerical methods for specialized applications."""
    
    @staticmethod
    def runge_kutta_4th_order(f: Callable[[float, float], float], 
                             x0: float, y0: float, h: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """4th-order Runge-Kutta method for solving ODEs."""
        x = np.zeros(n_steps + 1)
        y = np.zeros(n_steps + 1)
        
        x[0] = x0
        y[0] = y0
        
        for i in range(n_steps):
            k1 = h * f(x[i], y[i])
            k2 = h * f(x[i] + h/2, y[i] + k1/2)
            k3 = h * f(x[i] + h/2, y[i] + k2/2)
            k4 = h * f(x[i] + h, y[i] + k3)
            
            y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
            x[i + 1] = x[i] + h
        
        return x, y
    
    @staticmethod
    def euler_method(f: Callable[[float, float], float], 
                    x0: float, y0: float, h: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Euler's method for solving ODEs."""
        x = np.zeros(n_steps + 1)
        y = np.zeros(n_steps + 1)
        
        x[0] = x0
        y[0] = y0
        
        for i in range(n_steps):
            y[i + 1] = y[i] + h * f(x[i], y[i])
            x[i + 1] = x[i] + h
        
        return x, y
    
    @staticmethod
    def adams_bashforth_method(f: Callable[[float, float], float],
                              x0: float, y0: float, h: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Adams-Bashforth method for solving ODEs."""
        x = np.zeros(n_steps + 1)
        y = np.zeros(n_steps + 1)
        
        x[0] = x0
        y[0] = y0
        
        # Use RK4 for first few steps
        for i in range(min(3, n_steps)):
            k1 = h * f(x[i], y[i])
            k2 = h * f(x[i] + h/2, y[i] + k1/2)
            k3 = h * f(x[i] + h/2, y[i] + k2/2)
            k4 = h * f(x[i] + h, y[i] + k3)
            
            y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
            x[i + 1] = x[i] + h
        
        # Adams-Bashforth for remaining steps
        for i in range(3, n_steps):
            f_vals = [f(x[i-j], y[i-j]) for j in range(4)]
            y[i + 1] = y[i] + h/24 * (55*f_vals[0] - 59*f_vals[1] + 37*f_vals[2] - 9*f_vals[3])
            x[i + 1] = x[i] + h
        
        return x, y
    
    @staticmethod
    def finite_difference_pde(u_initial: np.ndarray, dx: float, dt: float, 
                             n_time_steps: int, diffusion_coeff: float = 1.0) -> np.ndarray:
        """Finite difference method for solving heat equation."""
        nx = len(u_initial)
        u = np.zeros((n_time_steps + 1, nx))
        u[0, :] = u_initial
        
        r = diffusion_coeff * dt / (dx**2)
        
        if r > 0.5:
            warnings.warn("Stability condition violated: r > 0.5")
        
        for t in range(n_time_steps):
            for i in range(1, nx - 1):
                u[t + 1, i] = u[t, i] + r * (u[t, i + 1] - 2*u[t, i] + u[t, i - 1])
            
            # Boundary conditions (assuming zero at boundaries)
            u[t + 1, 0] = 0
            u[t + 1, -1] = 0
        
        return u

# Exception classes for mathematical operations
class MathematicalError(Exception):
    """Base exception for mathematical operation errors."""
    pass

class ConvergenceError(MathematicalError):
    """Exception raised when numerical methods fail to converge."""
    pass

class NumericalInstabilityError(MathematicalError):
    """Exception raised when numerical instability is detected."""
    pass

class InvalidDomainError(MathematicalError):
    """Exception raised when function is evaluated outside valid domain."""
    pass

# Testing and validation functions
def run_math_utils_tests():
    """Comprehensive test suite for mathematical utilities."""
    print("Running mathematical utilities test suite...")
    
    # Test Vector3D operations
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    
    assert abs(v1.dot(v2) - 32) < 1e-10, "Vector dot product test failed"
    assert abs(v1.magnitude() - math.sqrt(14)) < 1e-10, "Vector magnitude test failed"
    
    # Test Quaternion operations
    q = Quaternion.from_euler(0, 0, math.pi/2)
    rotated = q.rotate_vector(Vector3D(1, 0, 0))
    assert abs(rotated.y - 1.0) < 1e-10, "Quaternion rotation test failed"
    
    # Test optimization
    def test_func(x):
        return (x[0] - 2)**2 + (x[1] - 3)**2
    
    def test_grad(x):
        return np.array([2*(x[0] - 2), 2*(x[1] - 3)])
    
    result = AdvancedOptimization.gradient_descent_with_momentum(
        test_func, test_grad, np.array([0, 0]), max_iterations=1000
    )
    
    assert result.success, "Optimization test failed"
    assert abs(result.optimal_parameters[0] - 2) < 1e-3, "Optimization accuracy test failed"
    
    # Test interpolation
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([0, 1, 4, 9, 16])  # y = x^2
    
    interp_val = InterpolationMethods.lagrange_interpolation(x_data, y_data, 2.5)
    expected = 2.5**2
    assert abs(interp_val - expected) < 1e-10, "Interpolation test failed"
    
    # Test root finding
    def test_root_func(x):
        return x**2 - 4
    
    def test_root_deriv(x):
        return 2*x
    
    root_result = NumericalRootFinding.newton_raphson(test_root_func, test_root_deriv, 1.0)
    assert root_result['converged'], "Root finding convergence test failed"
    assert abs(root_result['root'] - 2.0) < 1e-10, "Root finding accuracy test failed"
    
    print("All mathematical utilities tests passed successfully!")

# Module initialization
if __name__ == "__main__":
    print("AI Holographic Wristwatch - Mathematical Utilities Module")
    print("=" * 60)
    
    # Run comprehensive test suite
    try:
        run_math_utils_tests()
        print("✓ All tests passed successfully")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
    
    # Performance demonstration
    print("\nPerformance profiling demonstration:")
    with math_performance_context():
        # Test various mathematical operations
        test_matrix = np.random.rand(100, 100)
        eigenvalues, eigenvectors = AdvancedLinearAlgebra.eigendecomposition(test_matrix)
        
        test_signal = np.random.rand(1000)
        fft_result = scipy.fft.fft(test_signal)
        
        # Test holographic calculations
        viewer_pos = Vector3D(0, 0, 5)
        target_pos = Vector3D(0, 0, 0)
        projection_matrix = HolographicCalculations.calculate_projection_matrix(
            viewer_pos, target_pos, 45.0, 16/9, 0.1, 100.0
        )
    
    print("\nMathematical utilities module initialized and ready for use.")
    print("Available classes and functions:")
    print("- Vector3D: 3D vector operations")
    print("- Quaternion: Rotation mathematics")
    print("- AdvancedLinearAlgebra: Matrix operations")
    print("- HolographicCalculations: Holographic projections")
    print("- AdvancedSignalProcessing: Signal analysis")
    print("- AdvancedOptimization: Optimization algorithms")
    print("- ProbabilityDistributions: Statistical distributions")
    print("- AdvancedStatistics: Statistical analysis")
    print("- GeometricCalculations: Geometric operations")
    print("- NumericalIntegration: Integration methods")
    print("- NumericalDifferentiation: Differentiation methods")
    print("- NumericalRootFinding: Root finding algorithms")
    print("- InterpolationMethods: Data interpolation")
    print("- AdvancedNumericalMethods: ODEs and PDEs")
    print("- MathPerformanceProfiler: Performance monitoring")