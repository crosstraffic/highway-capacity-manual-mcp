"""Full-corpus validation — the entire rule base, in-process, no database.

The Chapter 15/12 ``validate_input`` gateway uses the lightweight semantic firewall (a handful of SV-* checks). This exposes the *full* validator instead: the complete rule corpus (HCM, AASHTO, MUTCD, HSM, ADA, MUTCD, OpenDRIVE, ...) across every parameter, with citations, terrain/context-gated rules, and clarification requests when an input is missing or its context is ambiguous.

It runs entirely in-process via ``ValidationEngine.from_seed()`` — the same evaluation engine the API server uses, but reading the bundled seed corpus instead of Postgres. No database server is required.
"""

from typing import Any, Dict

from transportations_validator.models.validation import ValidationContext
from transportations_validator.validators.engine import ValidationEngine

_ENGINE: ValidationEngine | None = None


def _engine() -> ValidationEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ValidationEngine.from_seed()
    return _ENGINE


async def validate_design_full_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a design against the full rule corpus, with citations and clarifications.

    ``data``: ``design`` (a flat dict of parameter -> value, e.g. ``{"lane_width": 9.0, "facility_type": "TwoLaneHighway"}``) and optional ``context`` (e.g. ``{"terrain_type": "mountainous", "jurisdiction": "state"}``).
    """
    try:
        ctx_in = data.get("context")
        ctx = ValidationContext(**ctx_in) if ctx_in else None
        result, extraction = await _engine().validate(data["design"], context=ctx)
        return {
            "success": True,
            "is_valid": result.is_valid,
            "facility_type": extraction.facility_type,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "violations": [
                {
                    "parameter": pv.rust_field,
                    "rule": v.rule_name,
                    "severity": v.severity,
                    "message": v.message,
                    "expected": v.expected,
                    "actual": v.actual,
                    "citation": v.citation,
                }
                for pv in result.parameters
                for v in (pv.violations + pv.warnings)
            ],
            "clarifications": [
                {
                    "type": getattr(c.type, "value", str(c.type)),
                    "parameter": c.parameter,
                    "message": c.message,
                    "question": c.suggested_question,
                    "options": c.options,
                }
                for c in result.clarifications
            ],
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}
