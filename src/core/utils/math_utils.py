# AI Holographic Wristwatch - Phase 1 Foundation: Core Utilities System

# src/core/utils/math_utils.py
"""
Advanced Mathematical Operations for AI Holographic Wristwatch System

This module provides comprehensive mathematical utilities for holographic calculations,
sensor data processing, 3D transformations, and AI system computations.
"""

import numpy as np
import math
from typing import Tuple, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import scipy.fft
import scipy.stats
import scipy.optimize
import warnings
import time
import functools


class TransformationType(Enum):
    """Types of mathematical transformations supported."""
    ROTATION = "rotation"
    TRANSLATION = "translation"
    SCALING = "scaling"
    PERSPECTIVE = "perspective"
    HOLOGRAPHIC = "holographic"

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
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
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

@dataclass
class Quaternion:
    """Quaternion representation for 3D rotations."""
    w: float
    x: float
    y: float
    z: float
    
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
    
    def normalize(self) -> 'Quaternion':
        """Normalize the quaternion."""
        norm = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm == 0:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)
    
    def conjugate(self) -> 'Quaternion':
        """Return the conjugate of the quaternion."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
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

class SensorFusionMath:
    """Mathematical operations for sensor data fusion."""
    
    @staticmethod
    def kalman_filter_predict(state: np.ndarray, covariance: np.ndarray,
                            transition_matrix: np.ndarray, 
                            process_noise: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman filter prediction step for sensor fusion."""
        predicted_state = transition_matrix @ state
        predicted_covariance = (transition_matrix @ covariance @ 
                              transition_matrix.T + process_noise)
        return predicted_state, predicted_covariance
    
    @staticmethod
    def kalman_filter_update(predicted_state: np.ndarray, 
                           predicted_covariance: np.ndarray,
                           measurement: np.ndarray, observation_matrix: np.ndarray,
                           measurement_noise: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman filter update step for sensor fusion."""
        innovation = measurement - observation_matrix @ predicted_state
        innovation_covariance = (observation_matrix @ predicted_covariance @ 
                               observation_matrix.T + measurement_noise)
        
        kalman_gain = (predicted_covariance @ observation_matrix.T @ 
                      np.linalg.inv(innovation_covariance))
        
        updated_state = predicted_state + kalman_gain @ innovation
        updated_covariance = ((np.eye(len(predicted_state)) - 
                             kalman_gain @ observation_matrix) @ predicted_covariance)
        
        return updated_state, updated_covariance
    
    @staticmethod
    def complementary_filter(accelerometer_data: np.ndarray, gyroscope_data: np.ndarray,
                           alpha: float = 0.98, dt: float = 0.01) -> np.ndarray:
        """Apply complementary filter for orientation estimation."""
        acc_angle = np.arctan2(accelerometer_data[1], accelerometer_data[2])
        gyro_angle = gyroscope_data[0] * dt
        
        filtered_angle = alpha * (acc_angle + gyro_angle) + (1 - alpha) * acc_angle
        return np.array([filtered_angle])
    
    @staticmethod
    def weighted_average_fusion(sensor_readings: List[float], 
                              confidence_weights: List[float],
                              normalize_weights: bool = True) -> float:
        """Combine multiple sensor readings using weighted averaging."""
        if len(sensor_readings) != len(confidence_weights):
            raise ValueError("Sensor readings and weights must have same length")
        
        if normalize_weights:
            total_weight = sum(confidence_weights)
            if total_weight == 0:
                return np.mean(sensor_readings)
            weights = [w / total_weight for w in confidence_weights]
        else:
            weights = confidence_weights
        
        return sum(reading * weight for reading, weight in zip(sensor_readings, weights))

class SignalProcessing:
    """Signal processing utilities for sensor data and audio processing."""
    
    @staticmethod
    def apply_fft(signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Fast Fourier Transform to signal."""
        frequencies = scipy.fft.fftfreq(len(signal), 1/sampling_rate)
        fft_values = scipy.fft.fft(signal)
        return frequencies, fft_values
    
    @staticmethod
    def apply_low_pass_filter(signal: np.ndarray, cutoff_frequency: float,
                            sampling_rate: float, order: int = 4) -> np.ndarray:
        """Apply low-pass filter to remove high-frequency noise."""
        from scipy import signal as scipy_signal
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = scipy_signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = scipy_signal.filtfilt(b, a, signal)
        return filtered_signal
    
    @staticmethod
    def detect_peaks(signal: np.ndarray, height: Optional[float] = None,
                    prominence: Optional[float] = None, 
                    distance: Optional[int] = None) -> np.ndarray:
        """Detect peaks in signal data."""
        from scipy import signal as scipy_signal
        peaks, _ = scipy_signal.find_peaks(signal, height=height, 
                                         prominence=prominence, distance=distance)
        return peaks
    
    @staticmethod
    def calculate_moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving average for noise reduction."""
        if window_size >= len(data):
            return np.full_like(data, np.mean(data))
        
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    @staticmethod
    def normalize_signal(signal: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize signal using specified method."""
        if method == 'minmax':
            min_val, max_val = np.min(signal), np.max(signal)
            if max_val == min_val:
                return np.zeros_like(signal)
            return (signal - min_val) / (max_val - min_val)
        elif method == 'zscore':
            return (signal - np.mean(signal)) / np.std(signal)
        elif method == 'robust':
            median = np.median(signal)
            mad = np.median(np.abs(signal - median))
            return (signal - median) / (1.4826 * mad) if mad != 0 else np.zeros_like(signal)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

class GeometricCalculations:
    """Geometric calculations for 3D space and holographic positioning."""
    
    @staticmethod
    def calculate_distance_3d(point1: Vector3D, point2: Vector3D) -> float:
        """Calculate Euclidean distance between two 3D points."""
        return (point1 - point2).magnitude()
    
    @staticmethod
    def calculate_angle_between_vectors(vector1: Vector3D, vector2: Vector3D) -> float:
        """Calculate angle between two vectors in radians."""
        dot_product = vector1.normalize().dot(vector2.normalize())
        dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to valid range
        return math.acos(dot_product)
    
    @staticmethod
    def project_point_onto_plane(point: Vector3D, plane_point: Vector3D,
                                plane_normal: Vector3D) -> Vector3D:
        """Project a point onto a plane."""
        normal = plane_normal.normalize()
        point_to_plane = point - plane_point
        projection_distance = point_to_plane.dot(normal)
        return point - normal * projection_distance
    
    @staticmethod
    def calculate_sphere_intersection(center1: Vector3D, radius1: float,
                                    center2: Vector3D, radius2: float) -> List[Vector3D]:
        """Calculate intersection points of two spheres."""
        distance = GeometricCalculations.calculate_distance_3d(center1, center2)
        
        if distance > radius1 + radius2 or distance < abs(radius1 - radius2) or distance == 0:
            return []  # No intersection
        
        a = (radius1**2 - radius2**2 + distance**2) / (2 * distance)
        h = math.sqrt(radius1**2 - a**2)
        
        # Calculate intersection circle center
        direction = (center2 - center1) * (a / distance)
        circle_center = center1 + direction
        
        # For simplicity, return center point (full circle calculation omitted)
        return [circle_center]
    
    @staticmethod
    def calculate_viewing_frustum(eye_position: Vector3D, look_at: Vector3D,
                                up_vector: Vector3D, fov: float, aspect: float,
                                near: float, far: float) -> dict:
        """Calculate viewing frustum for holographic display."""
        forward = (look_at - eye_position).normalize()
        right = forward.cross(up_vector.normalize()).normalize()
        up = right.cross(forward).normalize()
        
        fov_rad = math.radians(fov)
        half_height = near * math.tan(fov_rad / 2)
        half_width = half_height * aspect
        
        return {
            'eye_position': eye_position,
            'forward': forward,
            'right': right,
            'up': up,
            'near_height': half_height,
            'near_width': half_width,
            'near_distance': near,
            'far_distance': far
        }

class StatisticalAnalysis:
    """Statistical analysis utilities for data processing and AI systems."""
    
    @staticmethod
    def calculate_correlation_matrix(data_matrix: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix for multivariate data."""
        return np.corrcoef(data_matrix, rowvar=False)
    
    @staticmethod
    def detect_outliers_iqr(data: np.ndarray, iqr_multiplier: float = 1.5) -> np.ndarray:
        """Detect outliers using Interquartile Range method."""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        return np.where((data < lower_bound) | (data > upper_bound))[0]
    
    @staticmethod
    def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(scipy.stats.zscore(data))
        return np.where(z_scores > threshold)[0]
    
    @staticmethod
    def calculate_moving_statistics(data: np.ndarray, window_size: int) -> dict:
        """Calculate moving statistics (mean, std, min, max)."""
        if window_size >= len(data):
            return {
                'mean': np.full(len(data), np.mean(data)),
                'std': np.full(len(data), np.std(data)),
                'min': np.full(len(data), np.min(data)),
                'max': np.full(len(data), np.max(data))
            }
        
        moving_mean = SignalProcessing.calculate_moving_average(data, window_size)
        moving_std = np.array([np.std(data[i:i+window_size]) 
                              for i in range(len(data) - window_size + 1)])
        moving_min = np.array([np.min(data[i:i+window_size]) 
                              for i in range(len(data) - window_size + 1)])
        moving_max = np.array([np.max(data[i:i+window_size]) 
                              for i in range(len(data) - window_size + 1)])
        
        return {
            'mean': moving_mean,
            'std': moving_std,
            'min': moving_min,
            'max': moving_max
        }
    
    @staticmethod
    def fit_polynomial_trend(x_data: np.ndarray, y_data: np.ndarray, 
                           degree: int = 2) -> Tuple[np.ndarray, float]:
        """Fit polynomial trend to data and return coefficients and R-squared."""
        coefficients = np.polyfit(x_data, y_data, degree)
        predicted = np.polyval(coefficients, x_data)
        
        ss_res = np.sum((y_data - predicted) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return coefficients, r_squared

class OptimizationAlgorithms:
    """Optimization algorithms for system performance and AI training."""
    
    @staticmethod
    def gradient_descent(objective_function: callable, initial_params: np.ndarray,
                        learning_rate: float = 0.01, max_iterations: int = 1000,
                        tolerance: float = 1e-6) -> Tuple[np.ndarray, float]:
        """Implement gradient descent optimization."""
        params = initial_params.copy()
        
        for iteration in range(max_iterations):
            # Numerical gradient calculation
            gradient = OptimizationAlgorithms._numerical_gradient(objective_function, params)
            new_params = params - learning_rate * gradient
            
            if np.linalg.norm(new_params - params) < tolerance:
                break
            
            params = new_params
        
        final_cost = objective_function(params)
        return params, final_cost
    
    @staticmethod
    def _numerical_gradient(function: callable, params: np.ndarray, 
                          epsilon: float = 1e-8) -> np.ndarray:
        """Calculate numerical gradient of function."""
        gradient = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            gradient[i] = (function(params_plus) - function(params_minus)) / (2 * epsilon)
        
        return gradient
    
    @staticmethod
    def simulated_annealing(objective_function: callable, initial_solution: np.ndarray,
                          initial_temperature: float = 1000.0,
                          cooling_rate: float = 0.95, min_temperature: float = 1e-8,
                          max_iterations: int = 10000) -> Tuple[np.ndarray, float]:
        """Implement simulated annealing optimization."""
        current_solution = initial_solution.copy()
        current_cost = objective_function(current_solution)
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        temperature = initial_temperature
        
        for iteration in range(max_iterations):
            if temperature < min_temperature:
                break
            
            # Generate neighbor solution with random perturbation
            perturbation = np.random.normal(0, temperature * 0.1, len(current_solution))
            neighbor_solution = current_solution + perturbation
            neighbor_cost = objective_function(neighbor_solution)
            
            # Accept or reject the neighbor solution
            if neighbor_cost < current_cost or np.random.random() < np.exp(-(neighbor_cost - current_cost) / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
            
            temperature *= cooling_rate
        
        return best_solution, best_cost
    
    @staticmethod
    def particle_swarm_optimization(objective_function: callable, bounds: List[Tuple[float, float]],
                                  num_particles: int = 30, max_iterations: int = 1000,
                                  w: float = 0.729, c1: float = 1.49445, 
                                  c2: float = 1.49445) -> Tuple[np.ndarray, float]:
        """Implement Particle Swarm Optimization."""
        dimensions = len(bounds)
        
        # Initialize particles
        particles = np.random.uniform([b[0] for b in bounds], 
                                    [b[1] for b in bounds], 
                                    (num_particles, dimensions))
        velocities = np.zeros((num_particles, dimensions))
        
        # Initialize personal and global best
        personal_best = particles.copy()
        personal_best_costs = np.array([objective_function(p) for p in particles])
        global_best_idx = np.argmin(personal_best_costs)
        global_best = personal_best[global_best_idx].copy()
        global_best_cost = personal_best_costs[global_best_idx]
        
        for iteration in range(max_iterations):
            for i in range(num_particles):
                # Update velocity
                r1, r2 = np.random.random(dimensions), np.random.random(dimensions)
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (personal_best[i] - particles[i]) +
                               c2 * r2 * (global_best - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
                
                # Apply bounds
                for j in range(dimensions):
                    particles[i][j] = np.clip(particles[i][j], bounds[j][0], bounds[j][1])
                
                # Evaluate fitness
                cost = objective_function(particles[i])
                
                # Update personal best
                if cost < personal_best_costs[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_costs[i] = cost
                    
                    # Update global best
                    if cost < global_best_cost:
                        global_best = particles[i].copy()
                        global_best_cost = cost
        
        return global_best, global_best_cost

class MatrixOperations:
    """Advanced matrix operations for 3D transformations and AI computations."""
    
    @staticmethod
    def create_transformation_matrix(translation: Vector3D, rotation: Quaternion,
                                   scale: Vector3D) -> np.ndarray:
        """Create 4x4 transformation matrix from components."""
        rotation_matrix = rotation.to_rotation_matrix()
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix * np.array([[scale.x, 0, 0],
                                                              [0, scale.y, 0],
                                                              [0, 0, scale.z]])
        transform_matrix[:3, 3] = [translation.x, translation.y, translation.z]
        
        return transform_matrix
    
    @staticmethod
    def decompose_transformation_matrix(matrix: np.ndarray) -> Tuple[Vector3D, Quaternion, Vector3D]:
        """Decompose 4x4 transformation matrix into components."""
        # Extract translation
        translation = Vector3D(matrix[0, 3], matrix[1, 3], matrix[2, 3])
        
        # Extract scale
        scale_x = np.linalg.norm(matrix[:3, 0])
        scale_y = np.linalg.norm(matrix[:3, 1])
        scale_z = np.linalg.norm(matrix[:3, 2])
        scale = Vector3D(scale_x, scale_y, scale_z)
        
        # Extract rotation matrix
        rotation_matrix = matrix[:3, :3] / np.array([[scale_x, 0, 0],
                                                    [0, scale_y, 0],
                                                    [0, 0, scale_z]])
        
        # Convert rotation matrix to quaternion
        trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                z = 0.25 * s
        
        rotation = Quaternion(w, x, y, z).normalize()
        
        return translation, rotation, scale
    
    @staticmethod
    def calculate_pseudo_inverse(matrix: np.ndarray, 
                               regularization: float = 1e-10) -> np.ndarray:
        """Calculate Moore-Penrose pseudo-inverse with regularization."""
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Apply regularization to singular values
        s_reg = s / (s**2 + regularization)
        
        return Vt.T @ np.diag(s_reg) @ U.T
    
    @staticmethod
    def solve_least_squares(A: np.ndarray, b: np.ndarray, 
                          regularization: float = 0.0) -> np.ndarray:
        """Solve least squares problem with optional regularization."""
        if regularization > 0:
            # Ridge regression
            AtA = A.T @ A + regularization * np.eye(A.shape[1])
            Atb = A.T @ b
            return np.linalg.solve(AtA, Atb)
        else:
            # Standard least squares
            return np.linalg.lstsq(A, b, rcond=None)[0]

class InterpolationAlgorithms:
    """Interpolation algorithms for smooth data transitions and animations."""
    
    @staticmethod
    def linear_interpolation(start_value: float, end_value: float, t: float) -> float:
        """Linear interpolation between two values."""
        t = max(0.0, min(1.0, t))  # Clamp t to [0, 1]
        return start_value + t * (end_value - start_value)
    
    @staticmethod
    def cubic_interpolation(values: List[float], t: float) -> float:
        """Cubic interpolation using Catmull-Rom spline."""
        if len(values) < 4:
            raise ValueError("Cubic interpolation requires at least 4 control points")
        
        t = max(0.0, min(1.0, t))
        t2 = t * t
        t3 = t2 * t
        
        # Catmull-Rom basis functions
        return (values[1] + 
                0.5 * t * (-values[0] + values[2]) +
                0.5 * t2 * (2*values[0] - 5*values[1] + 4*values[2] - values[3]) +
                0.5 * t3 * (-values[0] + 3*values[1] - 3*values[2] + values[3]))
    
    @staticmethod
    def spherical_linear_interpolation(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
        """Spherical linear interpolation for quaternions (SLERP)."""
        t = max(0.0, min(1.0, t))
        
        # Calculate dot product
        dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        
        # If dot product is negative, use -q2 to take shorter arc
        if dot < 0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            w = q1.w + t * (q2.w - q1.w)
            x = q1.x + t * (q2.x - q1.x)
            y = q1.y + t * (q2.y - q1.y)
            z = q1.z + t * (q2.z - q1.z)
            return Quaternion(w, x, y, z).normalize()
        
        # Calculate interpolation
        theta = math.acos(abs(dot))
        sin_theta = math.sin(theta)
        
        a = math.sin((1 - t) * theta) / sin_theta
        b = math.sin(t * theta) / sin_theta
        
        w = a * q1.w + b * q2.w
        x = a * q1.x + b * q2.x
        y = a * q1.y + b * q2.y
        z = a * q1.z + b * q2.z
        
        return Quaternion(w, x, y, z).normalize()
    
    @staticmethod
    def bezier_interpolation(control_points: List[Vector3D], t: float) -> Vector3D:
        """Bezier curve interpolation for smooth motion paths."""
        n = len(control_points) - 1
        t = max(0.0, min(1.0, t))
        
        # De Casteljau's algorithm
        points = [point for point in control_points]
        
        for r in range(1, n + 1):
            for i in range(n + 1 - r):
                points[i] = Vector3D(
                    (1 - t) * points[i].x + t * points[i + 1].x,
                    (1 - t) * points[i].y + t * points[i + 1].y,
                    (1 - t) * points[i].z + t * points[i + 1].z
                )
        
        return points[0]

class NumericalMethods:
    """Numerical methods for complex calculations."""
    
    @staticmethod
    def runge_kutta_4th_order(f: callable, y0: float, x0: float, x_end: float,
                            num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """4th order Runge-Kutta method for solving ODEs."""
        h = (x_end - x0) / num_steps
        x_values = np.linspace(x0, x_end, num_steps + 1)
        y_values = np.zeros(num_steps + 1)
        y_values[0] = y0
        
        for i in range(num_steps):
            x = x_values[i]
            y = y_values[i]
            
            k1 = h * f(x, y)
            k2 = h * f(x + h/2, y + k1/2)
            k3 = h * f(x + h/2, y + k2/2)
            k4 = h * f(x + h, y + k3)
            
            y_values[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return x_values, y_values
    
    @staticmethod
    def newton_raphson_method(f: callable, df: callable, x0: float,
                            tolerance: float = 1e-10, max_iterations: int = 100) -> float:
        """Newton-Raphson method for finding roots."""
        x = x0
        
        for iteration in range(max_iterations):
            fx = f(x)
            if abs(fx) < tolerance:
                return x
            
            dfx = df(x)
            if abs(dfx) < 1e-15:
                raise ValueError("Derivative is too small, cannot continue iteration")
            
            x_new = x - fx / dfx
            
            if abs(x_new - x) < tolerance:
                return x_new
            
            x = x_new
        
        raise ValueError(f"Failed to converge after {max_iterations} iterations")
    
    @staticmethod
    def trapezoid_integration(f: callable, a: float, b: float, n: int = 1000) -> float:
        """Numerical integration using trapezoidal rule."""
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x])
        
        integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
        return integral

class PerformanceOptimization:
    """Performance optimization utilities for mathematical operations."""
    
    @staticmethod
    def vectorize_operation(operation: callable, data: np.ndarray, 
                          chunk_size: Optional[int] = None) -> np.ndarray:
        """Vectorize operations for improved performance."""
        if chunk_size is None:
            return np.vectorize(operation)(data)
        
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_result = np.vectorize(operation)(chunk)
            results.append(chunk_result)
        
        return np.concatenate(results)
    
    @staticmethod
    def parallel_matrix_multiply(A: np.ndarray, B: np.ndarray, 
                               num_threads: Optional[int] = None) -> np.ndarray:
        """Parallel matrix multiplication for large matrices."""
        import concurrent.futures
        import os
        
        if num_threads is None:
            num_threads = os.cpu_count()
        
        if A.shape[0] < num_threads or A.shape[1] != B.shape[0]:
            return A @ B  # Fall back to standard multiplication
        
        def multiply_rows(start_row: int, end_row: int) -> np.ndarray:
            return A[start_row:end_row] @ B
        
        rows_per_thread = A.shape[0] // num_threads
        futures = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(num_threads):
                start_row = i * rows_per_thread
                end_row = start_row + rows_per_thread if i < num_threads - 1 else A.shape[0]
                future = executor.submit(multiply_rows, start_row, end_row)
                futures.append(future)
        
        results = [future.result() for future in futures]
        return np.vstack(results)
    
    @staticmethod
    def memoized_calculation(cache_size: int = 1000):
        """Decorator for memoizing expensive calculations."""
        from functools import wraps, lru_cache
        
        def decorator(func):
            @wraps(func)
            @lru_cache(maxsize=cache_size)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def profile_performance(func: callable) -> callable:
        """Decorator for profiling mathematical function performance."""
        import time
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            print(f"Function {func.__name__} executed in {execution_time:.6f} seconds")
            
            return result
        return wrapper

# Error handling utilities for mathematical operations
class MathematicalError(Exception):
    """Base exception for mathematical operation errors."""
    pass

class DivisionByZeroError(MathematicalError):
    """Exception for division by zero in mathematical operations."""
    pass

class InvalidInputError(MathematicalError):
    """Exception for invalid input parameters."""
    pass

class ConvergenceError(MathematicalError):
    """Exception for numerical methods that fail to converge."""
    pass

class DimensionMismatchError(MathematicalError):
    """Exception for matrix/vector dimension mismatches."""
    pass

# Utility functions for common mathematical operations
def safe_divide(numerator: float, denominator: float, 
               default_value: float = 0.0) -> float:
    """Safely divide two numbers, handling division by zero."""
    try:
        if abs(denominator) < 1e-15:
            return default_value
        return numerator / denominator
    except ZeroDivisionError:
        return default_value

def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to specified range."""
    return max(min_val, min(max_val, value))

def lerp_array(start_array: np.ndarray, end_array: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two arrays."""
    t = clamp_value(t, 0.0, 1.0)
    return start_array + t * (end_array - start_array)

def calculate_rms(data: np.ndarray) -> float:
    """Calculate Root Mean Square of data."""
    return np.sqrt(np.mean(data**2))

def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_linear = signal_power / noise_power
    return 10 * math.log10(snr_linear)

def generate_transformation_sequence(transformations: List[Tuple[TransformationType, Any]]) -> np.ndarray:
    """Generate sequence of transformation matrices."""
    result_matrix = np.eye(4)
    
    for transform_type, params in transformations:
        if transform_type == TransformationType.TRANSLATION:
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = [params.x, params.y, params.z]
            result_matrix = result_matrix @ translation_matrix
        
        elif transform_type == TransformationType.ROTATION:
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = params.to_rotation_matrix()
            result_matrix = result_matrix @ rotation_matrix
        
        elif transform_type == TransformationType.SCALING:
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = params.x
            scale_matrix[1, 1] = params.y
            scale_matrix[2, 2] = params.z
            result_matrix = result_matrix @ scale_matrix
    
    return result_matrix

# Performance monitoring utilities
class MathPerformanceMonitor:
    """Monitor performance of mathematical operations."""
    
    def __init__(self):
        self.operation_times = {}
        self.operation_counts = {}
    
    def record_operation(self, operation_name: str, execution_time: float):
        """Record performance data for an operation."""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
            self.operation_counts[operation_name] = 0
        
        self.operation_times[operation_name].append(execution_time)
        self.operation_counts[operation_name] += 1
    
    def get_performance_summary(self) -> dict:
        """Get performance summary for all recorded operations."""
        summary = {}
        
        for operation in self.operation_times:
            times = self.operation_times[operation]
            summary[operation] = {
                'count': self.operation_counts[operation],
                'total_time': sum(times),
                'average_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times)
            }
        
        return summary
    
    def clear_records(self):
        """Clear all recorded performance data."""
        self.operation_times.clear()
        self.operation_counts.clear()

# Global performance monitor instance
math_performance_monitor = MathPerformanceMonitor()

# Decorators for automatic performance monitoring
def monitor_math_performance(operation_name: str):
    """Decorator to automatically monitor mathematical operation performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            math_performance_monitor.record_operation(operation_name, execution_time)
            
            return result
        return wrapper
    return decorator

# Initialize module-level constants
MATH_PRECISION = 1e-12
DEFAULT_TOLERANCE = 1e-10
MAX_ITERATIONS = 10000
PI_2 = math.pi * 2
PI_HALF = math.pi / 2
SQRT_2 = math.sqrt(2)
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2

# Validation functions for mathematical inputs
def validate_vector_3d(vector: Any) -> bool:
    """Validate that input is a valid 3D vector."""
    return (isinstance(vector, Vector3D) and 
            all(isinstance(getattr(vector, attr), (int, float)) 
                for attr in ['x', 'y', 'z']) and
            all(math.isfinite(getattr(vector, attr)) 
                for attr in ['x', 'y', 'z']))

def validate_quaternion(quat: Any) -> bool:
    """Validate that input is a valid quaternion."""
    return (isinstance(quat, Quaternion) and
            all(isinstance(getattr(quat, attr), (int, float)) 
                for attr in ['w', 'x', 'y', 'z']) and
            all(math.isfinite(getattr(quat, attr)) 
                for attr in ['w', 'x', 'y', 'z']))

def validate_matrix_dimensions(matrix: np.ndarray, expected_shape: Tuple[int, ...]) -> bool:
    """Validate matrix dimensions."""
    return isinstance(matrix, np.ndarray) and matrix.shape == expected_shape

def validate_numeric_range(value: Union[int, float], min_val: float, max_val: float) -> bool:
    """Validate that numeric value is within specified range."""
    return (isinstance(value, (int, float)) and 
            math.isfinite(value) and 
            min_val <= value <= max_val)

if __name__ == "__main__":
    # Example usage and testing
    print("AI Holographic Wristwatch - Mathematical Utilities Module")
    print("Testing core mathematical operations...")
    
    # Test Vector3D operations
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    print(f"Vector addition: {v1 + v2}")
    print(f"Vector magnitude: {v1.magnitude()}")
    print(f"Vector cross product: {v1.cross(v2)}")
    
    # Test Quaternion operations
    q1 = Quaternion.from_euler(0, math.pi/4, 0)
    q2 = Quaternion.from_euler(0, math.pi/2, 0)
    interpolated = InterpolationAlgorithms.spherical_linear_interpolation(q1, q2, 0.5)
    print(f"SLERP result: w={interpolated.w:.4f}, x={interpolated.x:.4f}, "
          f"y={interpolated.y:.4f}, z={interpolated.z:.4f}")
    
    # Test signal processing
    test_signal = np.sin(2 * np.pi * np.linspace(0, 1, 100)) + 0.1 * np.random.randn(100)
    filtered_signal = SignalProcessing.apply_low_pass_filter(test_signal, 10, 100)
    print(f"Signal filtering completed. Original std: {np.std(test_signal):.4f}, "
          f"Filtered std: {np.std(filtered_signal):.4f}")
    
    print("Mathematical utilities module initialized successfully.")