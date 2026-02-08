"""
Tests for the validation integration between hcm-mcp-server and transportations-validator.
"""

import pytest


class TestValidationIntegration:
    """Test the validation gateway integration."""

    def test_import_validation_module(self):
        """Test that the validation module can be imported."""
        from hcm_mcp_server.core.validation import (
            validate_input,
            require_valid_input,
            format_validation_error,
        )
        assert callable(validate_input)
        assert callable(require_valid_input)
        assert callable(format_validation_error)

    def test_valid_highway_data(self):
        """Test validation with valid highway data."""
        from hcm_mcp_server.core.validation import validate_input

        data = {
            "lane_width": 11.0,
            "shoulder_width": 6.0,
            "segments": [
                {
                    "passing_type": 0,
                    "spl": 50,
                    "grade": 2.0,
                    "phf": 0.95,
                    "phv": 5.0,
                }
            ],
        }

        result = validate_input(data)
        assert result["success"] is True
        assert result["validated"] is True
        assert "violations" not in result or len(result.get("violations", [])) == 0

    def test_invalid_lane_width(self):
        """Test validation catches invalid lane width."""
        from hcm_mcp_server.core.validation import validate_input

        data = {
            "lane_width": 8.0,  # Invalid: below 9 ft
            "shoulder_width": 6.0,
            "segments": [{"passing_type": 0}],
        }

        result = validate_input(data)
        assert result["success"] is False
        assert result["error_count"] >= 1

        violations = result.get("violations", [])
        assert len(violations) >= 1
        assert violations[0]["rule_id"] == "SV-001"
        assert violations[0]["parameter"] == "lane_width"

    def test_invalid_shoulder_width(self):
        """Test validation catches invalid shoulder width."""
        from hcm_mcp_server.core.validation import validate_input

        data = {
            "lane_width": 11.0,
            "shoulder_width": 10.0,  # Invalid: above 8 ft
            "segments": [],
        }

        result = validate_input(data)
        assert result["success"] is False

        violations = result.get("violations", [])
        rule_ids = [v["rule_id"] for v in violations]
        assert "SV-002" in rule_ids

    def test_invalid_passing_type(self):
        """Test validation catches invalid passing type."""
        from hcm_mcp_server.core.validation import validate_input

        data = {
            "lane_width": 11.0,
            "shoulder_width": 6.0,
            "segments": [
                {"passing_type": 5}  # Invalid: must be 0, 1, or 2
            ],
        }

        result = validate_input(data)
        assert result["success"] is False

        violations = result.get("violations", [])
        rule_ids = [v["rule_id"] for v in violations]
        assert "SV-004" in rule_ids

    def test_invalid_horizontal_class(self):
        """Test validation catches invalid horizontal class."""
        from hcm_mcp_server.core.validation import validate_input

        data = {
            "lane_width": 11.0,
            "shoulder_width": 6.0,
            "segments": [
                {"passing_type": 0, "hor_class": 7}  # Invalid: must be 0-5
            ],
        }

        result = validate_input(data)
        assert result["success"] is False

        violations = result.get("violations", [])
        rule_ids = [v["rule_id"] for v in violations]
        assert "SV-003" in rule_ids

    def test_speed_radius_validation(self):
        """Test validation catches inadequate radius for speed."""
        from hcm_mcp_server.core.validation import validate_input

        data = {
            "lane_width": 11.0,
            "shoulder_width": 6.0,
            "segments": [
                {
                    "passing_type": 0,
                    "spl": 60,  # 60 mph requires minimum 1000 ft radius
                    "subsegments": [
                        {"design_rad": 500}  # Too small for 60 mph
                    ],
                }
            ],
        }

        result = validate_input(data)
        assert result["success"] is False

        violations = result.get("violations", [])
        rule_ids = [v["rule_id"] for v in violations]
        assert "SV-005" in rule_ids

    def test_multiple_violations(self):
        """Test validation catches multiple violations."""
        from hcm_mcp_server.core.validation import validate_input

        data = {
            "lane_width": 8.0,  # Invalid
            "shoulder_width": 10.0,  # Invalid
            "segments": [
                {"passing_type": 5}  # Invalid
            ],
        }

        result = validate_input(data)
        assert result["success"] is False
        assert result["error_count"] >= 3

    def test_format_validation_error(self):
        """Test error formatting."""
        from hcm_mcp_server.core.validation import (
            validate_input,
            format_validation_error,
        )

        data = {"lane_width": 8.0}
        result = validate_input(data)

        formatted = format_validation_error(result)
        assert "SV-001" in formatted
        assert "lane_width" in formatted

    def test_decorator_blocks_invalid_input(self):
        """Test the require_valid_input decorator blocks invalid input."""
        from hcm_mcp_server.core.validation import require_valid_input

        @require_valid_input
        def mock_function(data):
            return {"success": True, "message": "Function executed"}

        # Invalid input should be blocked
        result = mock_function({"highway_data": {"lane_width": 8.0}})
        assert result["success"] is False
        assert "violations" in result

    def test_decorator_allows_valid_input(self):
        """Test the require_valid_input decorator allows valid input."""
        from hcm_mcp_server.core.validation import require_valid_input

        @require_valid_input
        def mock_function(data):
            return {"success": True, "message": "Function executed"}

        # Valid input should pass through
        result = mock_function({
            "highway_data": {
                "lane_width": 11.0,
                "shoulder_width": 6.0,
                "segments": [],
            }
        })
        assert result["success"] is True
        assert result["message"] == "Function executed"


class TestSemanticValidatorDirect:
    """Test the semantic validator module directly."""

    def test_import_semantic_module(self):
        """Test importing the semantic validator."""
        from transportations_validator.validators import semantic
        from transportations_validator.validators.semantic import (
            validate,
            validate_highway,
            ValidationResult,
            Violation,
        )
        assert callable(validate)
        assert callable(validate_highway)

    def test_validate_function(self):
        """Test the flat validate function."""
        from transportations_validator.validators.semantic import validate

        result = validate({"lane_width": 8.0})
        assert not result.is_valid
        assert len(result.violations) >= 1
        assert result.violations[0].rule_id == "SV-001"

    def test_validate_highway_function(self):
        """Test the highway-specific validator."""
        from transportations_validator.validators.semantic import validate_highway

        result = validate_highway({
            "lane_width": 11.0,
            "shoulder_width": 6.0,
            "segments": [{"passing_type": 0, "spl": 50}],
        })
        assert result.is_valid

    def test_speed_radius_table(self):
        """Test the AASHTO speed-radius table."""
        from transportations_validator.validators.semantic import SPEED_RADIUS_TABLE

        # Verify key values from AASHTO Table 3-7
        assert SPEED_RADIUS_TABLE[50] == 710
        assert SPEED_RADIUS_TABLE[60] == 1000
        assert SPEED_RADIUS_TABLE[70] == 1310


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
