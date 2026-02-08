"""
Input Validation Gateway using transportations-validator.

This module provides input validation for MCP function calls using
the semantic validator from transportations-validator. It acts as
a gating layer before inputs reach the transportations-library.

Usage:
    from hcm_mcp_server.core.validation import validate_input, require_valid_input

    # Direct validation
    result = validate_input(highway_data)
    if not result["success"]:
        return result  # Return validation errors

    # Or use decorator
    @require_valid_input
    def my_function(data: dict) -> dict:
        # Only runs if validation passes
        ...
"""

from functools import wraps
from typing import Any, Callable

from transportations_validator.validators import semantic


def validate_input(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate highway input data against HCM/AASHTO constraints.

    This function validates the input before it reaches the
    transportations-library computational core.

    Args:
        data: Highway data dictionary, typically containing:
            - lane_width: Lane width in feet
            - shoulder_width: Shoulder width in feet
            - segments: List of segment dictionaries with passing_type, spl, etc.

    Returns:
        dict with:
            - success: True if validation passed
            - validated: True if input was checked
            - violations: List of constraint violations (if any)
            - message: Summary message

    Example:
        >>> result = validate_input({
        ...     "lane_width": 8,  # Invalid - below 9 ft
        ...     "segments": [{"passing_type": 0}]
        ... })
        >>> result["success"]
        False
        >>> result["violations"][0]["rule_id"]
        'SV-001'
    """
    # Use the highway validator for nested structures
    result = semantic.validate_highway(data)

    if result.is_valid:
        return {
            "success": True,
            "validated": True,
            "constraints_checked": result.constraints_checked,
            "message": f"Validation passed ({result.constraints_checked} constraints checked)",
        }

    # Format violations for MCP response
    violations = [
        {
            "rule_id": v.rule_id,
            "parameter": v.parameter,
            "value": v.value,
            "constraint": v.constraint,
            "citation": v.citation,
            "severity": v.severity.value,
        }
        for v in result.violations
    ]

    return {
        "success": False,
        "validated": True,
        "error": "Input validation failed",
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "constraints_checked": result.constraints_checked,
        "violations": violations,
        "message": f"Validation failed: {result.error_count} error(s), {result.warning_count} warning(s)",
    }


def require_valid_input(func: Callable) -> Callable:
    """
    Decorator that validates highway_data before calling the function.

    If validation fails, returns the validation error response
    instead of calling the wrapped function.

    Usage:
        @require_valid_input
        def complete_highway_analysis_function(data: dict) -> dict:
            # This only runs if data["highway_data"] is valid
            ...

    Args:
        func: Function that takes a dict with "highway_data" key

    Returns:
        Wrapped function that validates input first
    """

    @wraps(func)
    def wrapper(data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        # Extract highway data from the input
        highway_data = data.get("highway_data", data)

        # Validate
        validation_result = validate_input(highway_data)

        if not validation_result["success"]:
            return validation_result

        # Validation passed - call the original function
        return func(data, *args, **kwargs)

    return wrapper


def format_validation_error(result: dict[str, Any]) -> str:
    """
    Format a validation error for human-readable output.

    Args:
        result: Result from validate_input() when success=False

    Returns:
        Formatted error message string
    """
    if result.get("success", True):
        return "No validation errors"

    lines = ["Input Validation Failed:", ""]

    for v in result.get("violations", []):
        lines.append(f"  [{v['rule_id']}] {v['parameter']}")
        lines.append(f"    Value: {v['value']}")
        lines.append(f"    Constraint: {v['constraint']}")
        lines.append(f"    Source: {v['citation']}")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "validate_input",
    "require_valid_input",
    "format_validation_error",
]
