# src/core/utils/validation_utils.py
"""
Comprehensive Input Validation Framework for AI Holographic Wristwatch System

This module provides robust validation capabilities including schema validation,
type checking, range validation, pattern matching, sanitization functions,
security validation, and custom validation rule engines.
"""
import numpy as np
import re
import json
import math
import ipaddress
from typing import Tuple, Any, Dict, List, Optional, Union, Callable, Type, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import datetime
from urllib.parse import urlparse
import base64
import hashlib
import time


class ValidationSeverity(Enum):
    """Validation result severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

class ValidationType(Enum):
    """Types of validation checks."""
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    PATTERN_CHECK = "pattern_check"
    SECURITY_CHECK = "security_check"
    CUSTOM_CHECK = "custom_check"
    SCHEMA_CHECK = "schema_check"

@dataclass
class ValidationIssue:
    """Represents a validation issue found during validation."""
    field_path: str
    issue_type: ValidationType
    severity: ValidationSeverity
    message: str
    suggested_fix: Optional[str] = None
    actual_value: Optional[Any] = None
    expected_value: Optional[Any] = None

@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    sanitized_data: Optional[Any] = None
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue to result."""
        self.issues.append(issue)
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get validation issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def has_errors(self) -> bool:
        """Check if validation result has error-level issues."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of validation issues by severity."""
        summary = {severity.value: 0 for severity in ValidationSeverity}
        for issue in self.issues:
            summary[issue.severity.value] += 1
        return summary

class BaseValidator(ABC):
    """Abstract base class for all validators."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
        """Validate input value and return list of issues."""
        pass
    
    def __call__(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
        """Make validator callable."""
        return self.validate(value, field_path)

class TypeValidator(BaseValidator):
    """Validator for type checking."""
    
    def __init__(self, expected_type: Union[Type, Tuple[Type, ...]], 
                 allow_none: bool = False, strict: bool = True):
        super().__init__("type_validator", f"Validates type matches {expected_type}")
        self.expected_type = expected_type
        self.allow_none = allow_none
        self.strict = strict
    
    def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
        """Validate value type."""
        issues = []
        
        if value is None and self.allow_none:
            return issues
        
        if value is None and not self.allow_none:
            issues.append(ValidationIssue(
                field_path=field_path,
                issue_type=ValidationType.TYPE_CHECK,
                severity=ValidationSeverity.ERROR,
                message="Value cannot be None",
                expected_value=self.expected_type,
                actual_value=type(value)
            ))
            return issues
        
        if self.strict:
            type_check = type(value) == self.expected_type if not isinstance(self.expected_type, tuple) else type(value) in self.expected_type
        else:
            type_check = isinstance(value, self.expected_type)
        
        if not type_check:
            issues.append(ValidationIssue(
                field_path=field_path,
                issue_type=ValidationType.TYPE_CHECK,
                severity=ValidationSeverity.ERROR,
                message=f"Expected type {self.expected_type}, got {type(value)}",
                expected_value=self.expected_type,
                actual_value=type(value),
                suggested_fix=f"Convert value to {self.expected_type}"
            ))
        
        return issues

class RangeValidator(BaseValidator):
    """Validator for numeric range checking."""
    
    def __init__(self, min_value: Optional[Union[int, float]] = None,
                 max_value: Optional[Union[int, float]] = None,
                 inclusive_min: bool = True, inclusive_max: bool = True):
        super().__init__("range_validator", f"Validates range [{min_value}, {max_value}]")
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive_min = inclusive_min
        self.inclusive_max = inclusive_max
    
    def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
        """Validate value is within specified range."""
        issues = []
        
        try:
            numeric_value = float(value)
            
            if not math.isfinite(numeric_value):
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.RANGE_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Value must be finite, got {numeric_value}",
                    actual_value=numeric_value
                ))
                return issues
            
            if self.min_value is not None:
                min_check = (numeric_value >= self.min_value if self.inclusive_min 
                           else numeric_value > self.min_value)
                if not min_check:
                    operator = ">=" if self.inclusive_min else ">"
                    issues.append(ValidationIssue(
                        field_path=field_path,
                        issue_type=ValidationType.RANGE_CHECK,
                        severity=ValidationSeverity.ERROR,
                        message=f"Value must be {operator} {self.min_value}, got {numeric_value}",
                        expected_value=self.min_value,
                        actual_value=numeric_value
                    ))
            
            if self.max_value is not None:
                max_check = (numeric_value <= self.max_value if self.inclusive_max 
                           else numeric_value < self.max_value)
                if not max_check:
                    operator = "<=" if self.inclusive_max else "<"
                    issues.append(ValidationIssue(
                        field_path=field_path,
                        issue_type=ValidationType.RANGE_CHECK,
                        severity=ValidationSeverity.ERROR,
                        message=f"Value must be {operator} {self.max_value}, got {numeric_value}",
                        expected_value=self.max_value,
                        actual_value=numeric_value
                    ))
                    
        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                field_path=field_path,
                issue_type=ValidationType.TYPE_CHECK,
                severity=ValidationSeverity.ERROR,
                message=f"Value must be numeric for range validation, got {type(value)}",
                actual_value=value,
                suggested_fix="Provide a numeric value"
            ))
        
        return issues

class PatternValidator(BaseValidator):
    """Validator for pattern matching using regular expressions."""
    
    def __init__(self, pattern: str, flags: int = 0, negate: bool = False):
        super().__init__("pattern_validator", f"Validates pattern: {pattern}")
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern, flags)
        self.negate = negate
    
    def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
        """Validate value matches specified pattern."""
        issues = []
        
        try:
            string_value = str(value)
            match_result = bool(self.compiled_pattern.match(string_value))
            
            if self.negate:
                match_result = not match_result
                message_prefix = "must not match"
            else:
                message_prefix = "must match"
            
            if not match_result:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.PATTERN_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Value {message_prefix} pattern '{self.pattern}', got '{string_value}'",
                    expected_value=self.pattern,
                    actual_value=string_value
                ))
                
        except Exception as e:
            issues.append(ValidationIssue(
                field_path=field_path,
                issue_type=ValidationType.PATTERN_CHECK,
                severity=ValidationSeverity.ERROR,
                message=f"Pattern validation failed: {str(e)}",
                actual_value=value
            ))
        
        return issues

class SecurityValidator(BaseValidator):
    """Security-focused validator for preventing common attacks."""
    
    def __init__(self, check_sql_injection: bool = True, check_xss: bool = True,
                 check_path_traversal: bool = True, max_length: Optional[int] = None):
        super().__init__("security_validator", "Security validation against common attacks")
        self.check_sql_injection = check_sql_injection
        self.check_xss = check_xss
        self.check_path_traversal = check_path_traversal
        self.max_length = max_length
        
        # Common attack patterns
        self.sql_injection_patterns = [
            r"(\b(union|select|insert|update|delete|drop|exec|execute)\b)",
            r"(\b(or|and)\s+\w*\s*=\s*\w*)",
            r"(--|#|/\*|\*/)",
            r"(\b(information_schema|sys\.|mysql\.)\b)"
        ]
        
        self.xss_patterns = [
            r"<\s*script[^>]*>",
            r"javascript\s*:",
            r"on\w+\s*=",
            r"<\s*iframe[^>]*>",
            r"<\s*object[^>]*>",
            r"<\s*embed[^>]*>"
        ]
        
        self.path_traversal_patterns = [
            r"\.\.[\\/]",
            r"[\\/]\.\.[\\/]",
            r"\.\.%2[fF]",
            r"%2[eE]%2[eE][\\/]"
        ]
    
    def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
        """Validate value for security vulnerabilities."""
        issues = []
        
        try:
            string_value = str(value)
            
            # Check maximum length
            if self.max_length and len(string_value) > self.max_length:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.SECURITY_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Input exceeds maximum length of {self.max_length}",
                    expected_value=self.max_length,
                    actual_value=len(string_value)
                ))
            
            # SQL Injection check
            if self.check_sql_injection:
                for pattern in self.sql_injection_patterns:
                    if re.search(pattern, string_value, re.IGNORECASE):
                        issues.append(ValidationIssue(
                            field_path=field_path,
                            issue_type=ValidationType.SECURITY_CHECK,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Potential SQL injection detected: {pattern}",
                            actual_value=string_value,
                            suggested_fix="Remove SQL keywords and special characters"
                        ))
            
            # XSS check
            if self.check_xss:
                for pattern in self.xss_patterns:
                    if re.search(pattern, string_value, re.IGNORECASE):
                        issues.append(ValidationIssue(
                            field_path=field_path,
                            issue_type=ValidationType.SECURITY_CHECK,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Potential XSS attack detected: {pattern}",
                            actual_value=string_value,
                            suggested_fix="Remove or escape HTML/JavaScript content"
                        ))
            
            # Path traversal check
            if self.check_path_traversal:
                for pattern in self.path_traversal_patterns:
                    if re.search(pattern, string_value, re.IGNORECASE):
                        issues.append(ValidationIssue(
                            field_path=field_path,
                            issue_type=ValidationType.SECURITY_CHECK,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Potential path traversal attack detected: {pattern}",
                            actual_value=string_value,
                            suggested_fix="Remove directory traversal sequences"
                        ))
                        
        except Exception as e:
            issues.append(ValidationIssue(
                field_path=field_path,
                issue_type=ValidationType.SECURITY_CHECK,
                severity=ValidationSeverity.ERROR,
                message=f"Security validation failed: {str(e)}",
                actual_value=value
            ))
        
        return issues

class SchemaValidator(BaseValidator):
    """JSON Schema-like validator for complex data structures."""
    
    def __init__(self, schema: Dict[str, Any]):
        super().__init__("schema_validator", "Validates data against defined schema")
        self.schema = schema
    
    def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
        """Validate value against schema definition."""
        return self._validate_recursive(value, self.schema, field_path)
    
    def _validate_recursive(self, value: Any, schema: Dict[str, Any], 
                          field_path: str) -> List[ValidationIssue]:
        """Recursively validate value against schema."""
        issues = []
        
        # Check required fields
        if isinstance(value, dict) and "required" in schema:
            for required_field in schema["required"]:
                if required_field not in value:
                    issues.append(ValidationIssue(
                        field_path=f"{field_path}.{required_field}" if field_path else required_field,
                        issue_type=ValidationType.SCHEMA_CHECK,
                        severity=ValidationSeverity.ERROR,
                        message=f"Required field '{required_field}' is missing",
                        expected_value="present"
                    ))
        
        # Check type
        if "type" in schema:
            expected_type = schema["type"]
            if not self._check_schema_type(value, expected_type):
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.TYPE_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Expected type '{expected_type}', got {type(value).__name__}",
                    expected_value=expected_type,
                    actual_value=type(value).__name__
                ))
        
        # Check properties for objects
        if isinstance(value, dict) and "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in value:
                    prop_path = f"{field_path}.{prop_name}" if field_path else prop_name
                    prop_issues = self._validate_recursive(value[prop_name], prop_schema, prop_path)
                    issues.extend(prop_issues)
        
        # Check array items
        if isinstance(value, list) and "items" in schema:
            item_schema = schema["items"]
            for i, item in enumerate(value):
                item_path = f"{field_path}[{i}]" if field_path else f"[{i}]"
                item_issues = self._validate_recursive(item, item_schema, item_path)
                issues.extend(item_issues)
        
        # Check string constraints
        if isinstance(value, str) and "string" in schema:
            string_constraints = schema["string"]
            
            if "minLength" in string_constraints and len(value) < string_constraints["minLength"]:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.RANGE_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"String length {len(value)} below minimum {string_constraints['minLength']}",
                    expected_value=string_constraints["minLength"],
                    actual_value=len(value)
                ))
            
            if "maxLength" in string_constraints and len(value) > string_constraints["maxLength"]:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.RANGE_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"String length {len(value)} exceeds maximum {string_constraints['maxLength']}",
                    expected_value=string_constraints["maxLength"],
                    actual_value=len(value)
                ))
            
            if "pattern" in string_constraints:
                pattern_validator = PatternValidator(string_constraints["pattern"])
                pattern_issues = pattern_validator.validate(value, field_path)
                issues.extend(pattern_issues)
        
        # Check numeric constraints
        if isinstance(value, (int, float)) and "numeric" in schema:
            numeric_constraints = schema["numeric"]
            
            range_validator = RangeValidator(
                min_value=numeric_constraints.get("minimum"),
                max_value=numeric_constraints.get("maximum"),
                inclusive_min=numeric_constraints.get("inclusiveMinimum", True),
                inclusive_max=numeric_constraints.get("inclusiveMaximum", True)
            )
            range_issues = range_validator.validate(value, field_path)
            issues.extend(range_issues)
        
        return issues
    
    def _check_schema_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches schema type specification."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        if expected_type not in type_mapping:
            return False
        
        expected_python_type = type_mapping[expected_type]
        return isinstance(value, expected_python_type)

class AIInputValidator(BaseValidator):
    """Specialized validator for AI system inputs."""
    
    def __init__(self, max_tokens: int = 4000, check_harmful_content: bool = True,
                 check_prompt_injection: bool = True):
        super().__init__("ai_input_validator", "Validates AI system inputs")
        self.max_tokens = max_tokens
        self.check_harmful_content = check_harmful_content
        self.check_prompt_injection = check_prompt_injection
        
        self.harmful_keywords = [
            "violence", "hate", "discrimination", "harassment", "abuse",
            "exploit", "hack", "crack", "bypass", "override"
        ]
        
        self.prompt_injection_patterns = [
            r"ignore\s+(previous|all)\s+(instructions|prompts)",
            r"forget\s+(everything|all|previous)",
            r"new\s+(instructions|rules|system)",
            r"act\s+as\s+(if|though)",
            r"pretend\s+(to\s+be|you\s+are)",
            r"roleplay\s+as",
            r"simulate\s+(being|that)"
        ]
    
    def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
        """Validate AI input for safety and security."""
        issues = []
        
        try:
            text_input = str(value)
            
            # Token length check (approximate)
            estimated_tokens = len(text_input.split())
            if estimated_tokens > self.max_tokens:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.RANGE_CHECK,
                    severity=ValidationSeverity.WARNING,
                    message=f"Input may exceed token limit: ~{estimated_tokens} tokens",
                    expected_value=self.max_tokens,
                    actual_value=estimated_tokens,
                    suggested_fix="Reduce input length or split into multiple requests"
                ))
            
            # Harmful content check
            if self.check_harmful_content:
                text_lower = text_input.lower()
                found_harmful = [keyword for keyword in self.harmful_keywords 
                               if keyword in text_lower]
                
                if found_harmful:
                    issues.append(ValidationIssue(
                        field_path=field_path,
                        issue_type=ValidationType.SECURITY_CHECK,
                        severity=ValidationSeverity.WARNING,
                        message=f"Potentially harmful content detected: {', '.join(found_harmful)}",
                        actual_value=found_harmful,
                        suggested_fix="Review and modify content if necessary"
                    ))
            
            # Prompt injection check
            if self.check_prompt_injection:
                for pattern in self.prompt_injection_patterns:
                    if re.search(pattern, text_input, re.IGNORECASE):
                        issues.append(ValidationIssue(
                            field_path=field_path,
                            issue_type=ValidationType.SECURITY_CHECK,
                            severity=ValidationSeverity.ERROR,
                            message=f"Potential prompt injection detected: {pattern}",
                            actual_value=text_input,
                            suggested_fix="Rephrase input to avoid instruction override patterns"
                        ))
                        
        except Exception as e:
            issues.append(ValidationIssue(
                field_path=field_path,
                issue_type=ValidationType.SECURITY_CHECK,
                severity=ValidationSeverity.ERROR,
                message=f"AI input validation failed: {str(e)}",
                actual_value=value
            ))
        
        return issues

class SensorDataValidator(BaseValidator):
    """Specialized validator for sensor data inputs."""
    
    def __init__(self, sensor_type: str, expected_range: Tuple[float, float],
                 sampling_rate_range: Tuple[float, float] = (1.0, 1000.0),
                 check_noise_level: bool = True):
        super().__init__("sensor_data_validator", f"Validates {sensor_type} sensor data")
        self.sensor_type = sensor_type
        self.expected_range = expected_range
        self.sampling_rate_range = sampling_rate_range
        self.check_noise_level = check_noise_level
    
    def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
        """Validate sensor data for quality and consistency."""
        issues = []
        
        try:
            # Convert to numpy array for analysis
            if isinstance(value, list):
                sensor_data = np.array(value)
            elif isinstance(value, np.ndarray):
                sensor_data = value
            else:
                raise ValueError("Sensor data must be array-like")
            
            # Check data shape
            if sensor_data.ndim > 2:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.SCHEMA_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Sensor data has too many dimensions: {sensor_data.ndim}",
                    expected_value="1D or 2D array",
                    actual_value=f"{sensor_data.ndim}D array"
                ))
            
            # Check for empty data
            if sensor_data.size == 0:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.SCHEMA_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message="Sensor data is empty",
                    actual_value="empty array"
                ))
                return issues
            
            # Check value ranges
            min_val, max_val = np.min(sensor_data), np.max(sensor_data)
            expected_min, expected_max = self.expected_range
            
            if min_val < expected_min:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.RANGE_CHECK,
                    severity=ValidationSeverity.WARNING,
                    message=f"Sensor values below expected range: {min_val} < {expected_min}",
                    expected_value=expected_min,
                    actual_value=min_val
                ))
            
            if max_val > expected_max:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.RANGE_CHECK,
                    severity=ValidationSeverity.WARNING,
                    message=f"Sensor values above expected range: {max_val} > {expected_max}",
                    expected_value=expected_max,
                    actual_value=max_val
                ))
            
            # Check for invalid values
            invalid_count = np.sum(~np.isfinite(sensor_data))
            if invalid_count > 0:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.RANGE_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Found {invalid_count} invalid values (NaN/Inf)",
                    actual_value=invalid_count,
                    suggested_fix="Clean data by removing or interpolating invalid values"
                ))
            
            # Noise level check
            if self.check_noise_level and len(sensor_data) > 10:
                signal_std = np.std(sensor_data)
                signal_mean = np.abs(np.mean(sensor_data))
                
                if signal_mean > 0:
                    noise_ratio = signal_std / signal_mean
                    if noise_ratio > 0.5:  # High noise threshold
                        issues.append(ValidationIssue(
                            field_path=field_path,
                            issue_type=ValidationType.CUSTOM_CHECK,
                            severity=ValidationSeverity.WARNING,
                            message=f"High noise level detected: {noise_ratio:.3f}",
                            actual_value=noise_ratio,
                            suggested_fix="Apply noise filtering or check sensor calibration"
                        ))
                        
        except Exception as e:
            issues.append(ValidationIssue(
                field_path=field_path,
                issue_type=ValidationType.SCHEMA_CHECK,
                severity=ValidationSeverity.ERROR,
                message=f"Sensor data validation failed: {str(e)}",
                actual_value=value
            ))
        
        return issues

class CompositeValidator(BaseValidator):
    """Combines multiple validators for comprehensive validation."""
    
    def __init__(self, validators: List[BaseValidator], name: str = "composite_validator"):
        super().__init__(name, "Composite validator combining multiple validation rules")
        self.validators = validators
    
    def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
        """Apply all validators and combine results."""
        all_issues = []
        
        for validator in self.validators:
            try:
                validator_issues = validator.validate(value, field_path)
                all_issues.extend(validator_issues)
            except Exception as e:
                all_issues.append(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.CUSTOM_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validator '{validator.name}' failed: {str(e)}",
                    actual_value=value
                ))
        
        return all_issues
    
    def add_validator(self, validator: BaseValidator):
        """Add validator to composite."""
        self.validators.append(validator)
    
    def remove_validator(self, validator_name: str) -> bool:
        """Remove validator by name."""
        for i, validator in enumerate(self.validators):
            if validator.name == validator_name:
                del self.validators[i]
                return True
        return False

class DataSanitizer:
    """Data sanitization utilities for secure input processing."""
    
    @staticmethod
    def sanitize_string(input_string: str, 
                       allow_html: bool = False,
                       max_length: Optional[int] = None,
                       remove_control_chars: bool = True) -> str:
        """Sanitize string input for security."""
        if not isinstance(input_string, str):
            input_string = str(input_string)
        
        sanitized = input_string
        
        # Remove control characters
        if remove_control_chars:
            sanitized = ''.join(char for char in sanitized 
                              if ord(char) >= 32 or char in '\t\n\r')
        
        # HTML sanitization
        if not allow_html:
            # Basic HTML tag removal
            sanitized = re.sub(r'<[^>]+>', '', sanitized)
            # HTML entity decoding and re-encoding
            import html
            sanitized = html.escape(html.unescape(sanitized))
        
        # Length limiting
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Remove common injection attempts
        sanitized = re.sub(r'(--|#|/\*|\*/)', '', sanitized)
        sanitized = re.sub(r'javascript\s*:', 'javascript_', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations."""
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        # Ensure not empty and doesn't start with dot
        if not filename or filename.startswith('.'):
            filename = 'sanitized_' + filename
        
        return filename
    
    @staticmethod
    def sanitize_email(email: str) -> Optional[str]:
        """Sanitize and validate email address."""
        email = email.strip().lower()
        
        # Basic email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(email_pattern, email):
            return email
        else:
            return None
    
    @staticmethod
    def sanitize_url(url: str) -> Optional[str]:
        """Sanitize and validate URL."""
        try:
            parsed = urlparse(url.strip())
            
            # Check for valid scheme
            if parsed.scheme not in ['http', 'https']:
                return None
            
            # Check for valid domain
            if not parsed.netloc:
                return None
            
            # Reconstruct clean URL
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean_url += f"?{parsed.query}"
            
            return clean_url
            
        except Exception:
            return None
    
    @staticmethod
    def sanitize_numeric_input(input_value: Any, 
                             target_type: Type[Union[int, float]],
                             default_value: Optional[Union[int, float]] = None) -> Optional[Union[int, float]]:
        """Sanitize numeric input with type conversion."""
        try:
            if isinstance(input_value, str):
                # Remove non-numeric characters except decimal point and minus
                cleaned = re.sub(r'[^\d.-]', '', input_value)
                if not cleaned or cleaned in ['-', '.', '-.']:
                    return default_value
                input_value = cleaned
            
            if target_type == int:
                result = int(float(input_value))  # Handle decimal inputs
            else:
                result = float(input_value)
            
            # Check for reasonable bounds
            if not math.isfinite(result):
                return default_value
            
            return result
            
        except (ValueError, TypeError, OverflowError):
            return default_value

class ValidationRuleEngine:
    """Advanced validation rule engine with conditional logic."""
    
    def __init__(self):
        self.rules = []
        self.rule_groups = {}
        self.global_validators = []
    
    def add_rule(self, condition: Callable[[Any], bool], 
                 validator: BaseValidator,
                 rule_name: str, group: Optional[str] = None):
        """Add conditional validation rule."""
        rule = {
            'name': rule_name,
            'condition': condition,
            'validator': validator,
            'group': group
        }
        
        self.rules.append(rule)
        
        if group:
            if group not in self.rule_groups:
                self.rule_groups[group] = []
            self.rule_groups[group].append(rule)
    
    def add_global_validator(self, validator: BaseValidator):
        """Add validator that applies to all inputs."""
        self.global_validators.append(validator)
    
    def validate(self, data: Any, field_path: str = "") -> ValidationResult:
        """Apply rule engine validation to data."""
        result = ValidationResult(is_valid=True)
        
        # Apply global validators
        for validator in self.global_validators:
            try:
                issues = validator.validate(data, field_path)
                for issue in issues:
                    result.add_issue(issue)
            except Exception as e:
                result.add_issue(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.CUSTOM_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Global validator '{validator.name}' failed: {str(e)}",
                    actual_value=data
                ))
        
        # Apply conditional rules
        for rule in self.rules:
            try:
                if rule['condition'](data):
                    issues = rule['validator'].validate(data, field_path)
                    for issue in issues:
                        result.add_issue(issue)
            except Exception as e:
                result.add_issue(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.CUSTOM_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Rule '{rule['name']}' failed: {str(e)}",
                    actual_value=data
                ))
        
        return result
    
    def validate_group(self, data: Any, group_name: str, 
                      field_path: str = "") -> ValidationResult:
        """Validate data using specific rule group."""
        result = ValidationResult(is_valid=True)
        
        if group_name not in self.rule_groups:
            result.add_issue(ValidationIssue(
                field_path=field_path,
                issue_type=ValidationType.CUSTOM_CHECK,
                severity=ValidationSeverity.ERROR,
                message=f"Rule group '{group_name}' not found"
            ))
            return result
        
        for rule in self.rule_groups[group_name]:
            try:
                if rule['condition'](data):
                    issues = rule['validator'].validate(data, field_path)
                    for issue in issues:
                        result.add_issue(issue)
            except Exception as e:
                result.add_issue(ValidationIssue(
                    field_path=field_path,
                    issue_type=ValidationType.CUSTOM_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Group rule '{rule['name']}' failed: {str(e)}",
                    actual_value=data
                ))
        
        return result

# Pre-configured validators for common use cases
class CommonValidators:
    """Collection of pre-configured validators for common scenarios."""
    
    @staticmethod
    def email_validator() -> PatternValidator:
        """Email address validator."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return PatternValidator(email_pattern)
    
    @staticmethod
    def phone_validator() -> PatternValidator:
        """Phone number validator (US format)."""
        phone_pattern = r'^\+?1?[-.\s]?\(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}$'
        return PatternValidator(phone_pattern)
    
    @staticmethod
    def uuid_validator() -> PatternValidator:
        """UUID validator."""
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        return PatternValidator(uuid_pattern, re.IGNORECASE)
    
    @staticmethod
    def ip_address_validator() -> BaseValidator:
        """IP address validator (IPv4 and IPv6)."""
        class IPValidator(BaseValidator):
            def __init__(self):
                super().__init__("ip_validator", "Validates IP addresses")
            
            def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
                issues = []
                try:
                    ipaddress.ip_address(str(value))
                except ValueError:
                    issues.append(ValidationIssue(
                        field_path=field_path,
                        issue_type=ValidationType.PATTERN_CHECK,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid IP address: {value}",
                        actual_value=value,
                        suggested_fix="Provide valid IPv4 or IPv6 address"
                    ))
                return issues
        
        return IPValidator()
    
    @staticmethod
    def datetime_validator(date_format: str = "%Y-%m-%d %H:%M:%S") -> BaseValidator:
        """DateTime validator with custom format."""
        class DateTimeValidator(BaseValidator):
            def __init__(self, fmt: str):
                super().__init__("datetime_validator", f"Validates datetime format: {fmt}")
                self.format = fmt
            
            def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
                issues = []
                try:
                    if isinstance(value, str):
                        datetime.datetime.strptime(value, self.format)
                    elif isinstance(value, (int, float)):
                        datetime.datetime.fromtimestamp(value)
                    elif not isinstance(value, datetime.datetime):
                        raise ValueError("Invalid datetime type")
                except ValueError as e:
                    issues.append(ValidationIssue(
                        field_path=field_path,
                        issue_type=ValidationType.PATTERN_CHECK,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid datetime format: {str(e)}",
                        expected_value=self.format,
                        actual_value=value
                    ))
                return issues
        
        return DateTimeValidator(date_format)
    
    @staticmethod
    def json_validator() -> BaseValidator:
        """JSON format validator."""
        class JSONValidator(BaseValidator):
            def __init__(self):
                super().__init__("json_validator", "Validates JSON format")
            
            def validate(self, value: Any, field_path: str = "") -> List[ValidationIssue]:
                issues = []
                try:
                    if isinstance(value, str):
                        json.loads(value)
                    else:
                        json.dumps(value)
                except (json.JSONDecodeError, TypeError) as e:
                    issues.append(ValidationIssue(
                        field_path=field_path,
                        issue_type=ValidationType.SCHEMA_CHECK,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid JSON format: {str(e)}",
                        actual_value=value,
                        suggested_fix="Provide valid JSON format"
                    ))
                return issues
        
        return JSONValidator()

class ValidationCache:
    """Cache validation results to improve performance."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self._lock = threading.Lock()
    
    def _generate_cache_key(self, data: Any, validators: List[BaseValidator]) -> str:
        """Generate cache key for validation result."""
        # Create hash of data and validator configuration
        data_hash = hashlib.sha256(str(data).encode()).hexdigest()[:16]
        validator_hash = hashlib.sha256(
            ''.join(v.name for v in validators).encode()
        ).hexdigest()[:16]
        return f"{data_hash}_{validator_hash}"
    
    def get_cached_result(self, data: Any, validators: List[BaseValidator]) -> Optional[ValidationResult]:
        """Get cached validation result if available and fresh."""
        cache_key = self._generate_cache_key(data, validators)
        
        with self._lock:
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                
                # Check if cache entry is still valid
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end (LRU)
                    self.cache.move_to_end(cache_key)
                    return cached_data
                else:
                    # Remove expired entry
                    del self.cache[cache_key]
        
        return None
    
    def cache_result(self, data: Any, validators: List[BaseValidator], 
                    result: ValidationResult):
        """Cache validation result."""
        cache_key = self._generate_cache_key(data, validators)
        
        with self._lock:
            # Remove oldest entries if cache is full
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = (result, time.time())
    
    def clear_cache(self):
        """Clear all cached validation results."""
        with self._lock:
            self.cache.clear()

class ValidationProfiler:
    """Performance profiler for validation operations."""
    
    def __init__(self):
        self.validation_times = {}
        self.validation_counts = {}
        self.slow_validations = []
    
    def profile_validation(self, validator_name: str):
        """Decorator for profiling validation performance."""
        def decorator(validation_func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = validation_func(*args, **kwargs)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                self.record_validation_time(validator_name, execution_time)
                
                # Track slow validations (>10ms)
                if execution_time > 0.01:
                    self.slow_validations.append({
                        'validator': validator_name,
                        'time': execution_time,
                        'timestamp': time.time()
                    })
                
                return result
            return wrapper
        return decorator
    
    def record_validation_time(self, validator_name: str, execution_time: float):
        """Record validation execution time."""
        if validator_name not in self.validation_times:
            self.validation_times[validator_name] = []
            self.validation_counts[validator_name] = 0
        
        self.validation_times[validator_name].append(execution_time)
        self.validation_counts[validator_name] += 1
        
        # Keep only recent measurements
        if len(self.validation_times[validator_name]) > 1000:
            self.validation_times[validator_name] = self.validation_times[validator_name][-1000:]
    
    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive performance report."""
        report = {}
        
        for validator_name in self.validation_times:
            times = self.validation_times[validator_name]
            report[validator_name] = {
                'count': self.validation_counts[validator_name],
                'total_time': sum(times),
                'average_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times),
                'p95_time': np.percentile(times, 95),
                'p99_time': np.percentile(times, 99)
            }
        
        return report

# Custom exceptions for validation
class ValidationError(Exception):
    """Base exception for validation errors."""
    pass

class SchemaValidationError(ValidationError):
    """Exception for schema validation failures."""
    pass

class SecurityValidationError(ValidationError):
    """Exception for security validation failures."""
    pass

class SanitizationError(ValidationError):
    """Exception for data sanitization errors."""
    pass

# Utility functions for common validation scenarios
def validate_holographic_parameters(projection_distance: float, viewing_angle: float,
                                  hologram_size: float) -> ValidationResult:
    """Validate holographic projection parameters."""
    result = ValidationResult(is_valid=True)
    
    # Distance validation (1-100 cm)
    distance_validator = RangeValidator(min_value=0.01, max_value=1.0)
    distance_issues = distance_validator.validate(projection_distance, "projection_distance")
    for issue in distance_issues:
        result.add_issue(issue)
    
    # Viewing angle validation (15-120 degrees)
    angle_validator = RangeValidator(min_value=15.0, max_value=120.0)
    angle_issues = angle_validator.validate(viewing_angle, "viewing_angle")
    for issue in angle_issues:
        result.add_issue(issue)
    
    # Hologram size validation (0.1-50 cm)
    size_validator = RangeValidator(min_value=0.001, max_value=0.5)
    size_issues = size_validator.validate(hologram_size, "hologram_size")
    for issue in size_issues:
        result.add_issue(issue)
    
    return result

def validate_ai_conversation_input(user_input: str, max_length: int = 2000) -> ValidationResult:
    """Validate AI conversation input for safety and quality."""
    result = ValidationResult(is_valid=True)
    
    # Create composite validator
    validators = [
        TypeValidator(str),
        SecurityValidator(max_length=max_length),
        AIInputValidator(max_tokens=max_length//4)
    ]
    
    composite_validator = CompositeValidator(validators, "conversation_input_validator")
    issues = composite_validator.validate(user_input, "user_input")
    
    for issue in issues:
        result.add_issue(issue)
    
    # Add sanitized version
    if result.is_valid or not result.has_errors():
        result.sanitized_data = DataSanitizer.sanitize_string(
            user_input, 
            allow_html=False, 
            max_length=max_length
        )
    
    return result

def validate_sensor_configuration(sensor_config: Dict[str, Any]) -> ValidationResult:
    """Validate sensor configuration parameters."""
    schema = {
        "type": "object",
        "required": ["sensor_type", "sampling_rate", "range"],
        "properties": {
            "sensor_type": {
                "type": "string",
                "string": {
                    "pattern": r"^(accelerometer|gyroscope|magnetometer|heart_rate|temperature|pressure)$"
                }
            },
            "sampling_rate": {
                "type": "number",
                "numeric": {
                    "minimum": 1.0,
                    "maximum": 1000.0
                }
            },
            "range": {
                "type": "array",
                "items": {
                    "type": "number"
                }
            },
            "calibration_offset": {
                "type": "number",
                "numeric": {
                    "minimum": -100.0,
                    "maximum": 100.0
                }
            }
        }
    }
    
    result = ValidationResult(is_valid=True)
    schema_validator = SchemaValidator(schema)
    issues = schema_validator.validate(sensor_config, "sensor_config")
    
    for issue in issues:
        result.add_issue(issue)
    
    return result

def validate_device_credentials(device_id: str, auth_token: str, 
                              timestamp: float) -> ValidationResult:
    """Validate device authentication credentials."""
    result = ValidationResult(is_valid=True)
    
    # Device ID validation (UUID format)
    uuid_validator = CommonValidators.uuid_validator()
    device_id_issues = uuid_validator.validate(device_id, "device_id")
    for issue in device_id_issues:
        result.add_issue(issue)
    
    # Auth token validation (base64 encoded, minimum 32 characters)
    try:
        decoded_token = base64.b64decode(auth_token, validate=True)
        if len(decoded_token) < 32:
            result.add_issue(ValidationIssue(
                field_path="auth_token",
                issue_type=ValidationType.SECURITY_CHECK,
                severity=ValidationSeverity.ERROR,
                message="Auth token too short",
                expected_value="At least 32 bytes",
                actual_value=f"{len(decoded_token)} bytes"
            ))
    except Exception:
        result.add_issue(ValidationIssue(
            field_path="auth_token",
            issue_type=ValidationType.SECURITY_CHECK,
            severity=ValidationSeverity.ERROR,
            message="Invalid base64 auth token format",
            actual_value=auth_token,
            suggested_fix="Provide valid base64 encoded token"
        ))
    
    # Timestamp validation (not too old or in future)
    current_time = time.time()
    time_diff = abs(current_time - timestamp)
    
    if time_diff > 300:  # 5 minutes tolerance
        result.add_issue(ValidationIssue(
            field_path="timestamp",
            issue_type=ValidationType.RANGE_CHECK,
            severity=ValidationSeverity.WARNING,
            message=f"Timestamp differs from current time by {time_diff:.1f} seconds",
            expected_value="Within 5 minutes of current time",
            actual_value=timestamp,
            suggested_fix="Synchronize device clock"
        ))
    
    return result

def create_ai_personality_validator() -> ValidationRuleEngine:
    """Create validation rule engine for AI personality parameters."""
    engine = ValidationRuleEngine()
    
    # Add global validators
    engine.add_global_validator(SecurityValidator(max_length=1000))
    
    # Personality trait validation
    trait_condition = lambda data: isinstance(data, dict) and 'personality_traits' in data
    trait_schema = {
        "type": "object",
        "properties": {
            "personality_traits": {
                "type": "object", 
                "properties": {
                    "extraversion": {"type": "number", "numeric": {"minimum": 0.0, "maximum": 1.0}},
                    "agreeableness": {"type": "number", "numeric": {"minimum": 0.0, "maximum": 1.0}},
                    "conscientiousness": {"type": "number", "numeric": {"minimum": 0.0, "maximum": 1.0}},
                    "neuroticism": {"type": "number", "numeric": {"minimum": 0.0, "maximum": 1.0}},
                    "openness": {"type": "number", "numeric": {"minimum": 0.0, "maximum": 1.0}}
                }
            }
        }
    }
    
    engine.add_rule(
        condition=trait_condition,
        validator=SchemaValidator(trait_schema),
        rule_name="personality_traits_validation",
        group="personality"
    )
    
    # Voice characteristics validation
    voice_condition = lambda data: isinstance(data, dict) and 'voice_config' in data
    voice_validator = CompositeValidator([
        TypeValidator(dict),
        RangeValidator(min_value=0.5, max_value=2.0)  # For pitch multiplier
    ])
    
    engine.add_rule(
        condition=voice_condition,
        validator=voice_validator,
        rule_name="voice_config_validation", 
        group="personality"
    )
    
    return engine

# Global validation instances
global_validation_cache = ValidationCache()
global_validation_profiler = ValidationProfiler()

# Thread-safe validation result aggregator
class ValidationResultAggregator:
    """Thread-safe aggregation of validation results across the system."""
    
    def __init__(self):
        self._results = []
        self._lock = threading.Lock()
        self._total_validations = 0
        self._successful_validations = 0
    
    def add_result(self, result: ValidationResult, context: str = ""):
        """Add validation result to aggregator."""
        with self._lock:
            self._results.append({
                'result': result,
                'context': context,
                'timestamp': time.time()
            })
            
            self._total_validations += 1
            if result.is_valid:
                self._successful_validations += 1
            
            # Keep only recent results
            if len(self._results) > 10000:
                self._results = self._results[-5000:]
    
    def get_success_rate(self) -> float:
        """Calculate overall validation success rate."""
        with self._lock:
            if self._total_validations == 0:
                return 1.0
            return self._successful_validations / self._total_validations
    
    def get_recent_failures(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent validation failures."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            recent_failures = [
                entry for entry in self._results
                if entry['timestamp'] > cutoff_time and not entry['result'].is_valid
            ]
        
        return recent_failures
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        with self._lock:
            total_issues_by_severity = {severity.value: 0 for severity in ValidationSeverity}
            issues_by_type = {vtype.value: 0 for vtype in ValidationType}
            context_stats = {}
            
            for entry in self._results:
                result = entry['result']
                context = entry['context']
                
                # Update context statistics
                if context not in context_stats:
                    context_stats[context] = {'total': 0, 'successful': 0}
                context_stats[context]['total'] += 1
                if result.is_valid:
                    context_stats[context]['successful'] += 1
                
                # Update issue statistics
                for issue in result.issues:
                    total_issues_by_severity[issue.severity.value] += 1
                    issues_by_type[issue.issue_type.value] += 1
            
            return {
                'total_validations': self._total_validations,
                'successful_validations': self._successful_validations,
                'success_rate': self.get_success_rate(),
                'issues_by_severity': total_issues_by_severity,
                'issues_by_type': issues_by_type,
                'context_statistics': context_stats,
                'report_timestamp': time.time()
            }

# Global validation result aggregator
global_validation_aggregator = ValidationResultAggregator()

# Utility functions for complex validation scenarios
def validate_with_context(data: Any, validators: List[BaseValidator],
                         context: str = "", use_cache: bool = True,
                         profile_performance: bool = True) -> ValidationResult:
    """Validate data with caching, profiling, and result aggregation."""
    # Check cache first
    if use_cache:
        cached_result = global_validation_cache.get_cached_result(data, validators)
        if cached_result:
            global_validation_aggregator.add_result(cached_result, context)
            return cached_result
    
    # Perform validation
    start_time = time.perf_counter()
    result = ValidationResult(is_valid=True)
    
    for validator in validators:
        try:
            issues = validator.validate(data)
            for issue in issues:
                result.add_issue(issue)
        except Exception as e:
            result.add_issue(ValidationIssue(
                field_path="",
                issue_type=ValidationType.CUSTOM_CHECK,
                severity=ValidationSeverity.ERROR,
                message=f"Validation error in {validator.name}: {str(e)}",
                actual_value=data
            ))
    
    end_time = time.perf_counter()
    
    # Record performance
    if profile_performance:
        execution_time = end_time - start_time
        global_validation_profiler.record_validation_time(context or "unknown", execution_time)
    
    # Cache result
    if use_cache:
        global_validation_cache.cache_result(data, validators, result)
    
    # Add to aggregator
    global_validation_aggregator.add_result(result, context)
    
    return result

def batch_validate(data_list: List[Any], validators: List[BaseValidator],
                  max_workers: Optional[int] = None) -> List[ValidationResult]:
    """Validate multiple data items in parallel."""
    import concurrent.futures
    
    def validate_single_item(data: Any) -> ValidationResult:
        return validate_with_context(data, validators, "batch_validation")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(validate_single_item, data_list))
    
    return results

def create_sensor_data_validator(sensor_type: str) -> CompositeValidator:
    """Create specialized validator for specific sensor type."""
    validators = []
    
    # Common sensor validations
    validators.append(TypeValidator((list, np.ndarray)))
    validators.append(SecurityValidator(max_length=100000))  # Prevent DOS via large arrays
    
    # Sensor-specific validations
    if sensor_type == "accelerometer":
        validators.append(SensorDataValidator(sensor_type, (-20.0, 20.0)))  # ±20g range
    elif sensor_type == "gyroscope":
        validators.append(SensorDataValidator(sensor_type, (-2000.0, 2000.0)))  # ±2000 dps
    elif sensor_type == "magnetometer":
        validators.append(SensorDataValidator(sensor_type, (-100.0, 100.0)))  # ±100 µT
    elif sensor_type == "heart_rate":
        validators.append(SensorDataValidator(sensor_type, (30.0, 220.0)))  # 30-220 BPM
    elif sensor_type == "temperature":
        validators.append(SensorDataValidator(sensor_type, (15.0, 45.0)))  # 15-45°C
    else:
        # Default range for unknown sensors
        validators.append(SensorDataValidator(sensor_type, (-1000.0, 1000.0)))
    
    return CompositeValidator(validators, f"{sensor_type}_validator")

def sanitize_user_input_comprehensive(user_input: str, 
                                    input_type: str = "general") -> Tuple[str, ValidationResult]:
    """Comprehensively sanitize user input and return validation result."""
    result = ValidationResult(is_valid=True)
    
    # Input type specific sanitization
    if input_type == "conversation":
        sanitized = DataSanitizer.sanitize_string(
            user_input,
            allow_html=False,
            max_length=2000,
            remove_control_chars=True
        )
        validation_result = validate_ai_conversation_input(sanitized)
        
    elif input_type == "filename":
        sanitized = DataSanitizer.sanitize_filename(user_input)
        validation_result = ValidationResult(is_valid=True)
        
    elif input_type == "email":
        sanitized = DataSanitizer.sanitize_email(user_input)
        if sanitized is None:
            sanitized = ""
            result.add_issue(ValidationIssue(
                field_path="email",
                issue_type=ValidationType.PATTERN_CHECK,
                severity=ValidationSeverity.ERROR,
                message="Invalid email format",
                actual_value=user_input
            ))
        validation_result = result
        
    elif input_type == "url":
        sanitized = DataSanitizer.sanitize_url(user_input)
        if sanitized is None:
            sanitized = ""
            result.add_issue(ValidationIssue(
                field_path="url",
                issue_type=ValidationType.PATTERN_CHECK,
                severity=ValidationSeverity.ERROR,
                message="Invalid URL format",
                actual_value=user_input
            ))
        validation_result = result
        
    else:
        # General sanitization
        sanitized = DataSanitizer.sanitize_string(user_input)
        validation_result = validate_with_context(
            sanitized,
            [SecurityValidator()],
            f"general_input_{input_type}"
        )
    
    return sanitized, validation_result

if __name__ == "__main__":
    # Example usage and testing
    print("AI Holographic Wristwatch - Validation Utilities Module")
    print("Testing validation framework...")
    
    # Test type validation
    type_validator = TypeValidator(str)
    type_issues = type_validator.validate(123, "test_field")
    print(f"Type validation issues: {len(type_issues)}")
    
    # Test range validation
    range_validator = RangeValidator(min_value=0, max_value=100)
    range_issues = range_validator.validate(150, "test_range")
    print(f"Range validation issues: {len(range_issues)}")
    
    # Test security validation
    security_validator = SecurityValidator()
    security_issues = security_validator.validate("SELECT * FROM users", "test_input")
    print(f"Security validation issues: {len(security_issues)}")
    
    # Test holographic parameters
    holo_result = validate_holographic_parameters(0.3, 45.0, 0.1)
    print(f"Holographic validation result: valid={holo_result.is_valid}, issues={len(holo_result.issues)}")
    
    # Test AI conversation input
    conversation_result = validate_ai_conversation_input("Hello, how can you help me today?")
    print(f"Conversation validation: valid={conversation_result.is_valid}")
    print(f"Sanitized input: '{conversation_result.sanitized_data}'")
    
    # Test comprehensive sanitization
    test_input = "<script>alert('xss')</script>Hello world!"
    sanitized, validation_result = sanitize_user_input_comprehensive(test_input, "conversation")
    print(f"Comprehensive sanitization: '{sanitized}'")
    print(f"Validation issues: {len(validation_result.issues)}")
    
    print("Validation utilities module initialized successfully.")