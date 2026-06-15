"""HCM Chapter 12 — Basic Freeway Segments.

The freeway counterpart to ``chapter15.py``: each function wraps the verified ``transportations_library.BasicFreeways`` implementation (a *different equation family* than the two-lane methodology) and follows the same ``(data: dict) -> dict`` convention. A basic-freeway analysis is a single directional segment, so the input is ``data["freeway_data"]`` (a flat ``BasicFreewaysInput``), not a multi-segment facility.

The HCM Ch.12 step sequence is stateful, so each step function rebuilds the segment and runs its prerequisites in order before the requested step (mirroring ``chapter15.py``). The library tabulates heavy-vehicle effects only at discrete grade/length grid points and raises a Rust panic off-grid; the ``_guarded`` decorator catches that (a PyO3 PanicException is NOT a Python Exception) and returns a clean error instead of crashing.
"""

from functools import wraps
from typing import Any, Callable, Dict

from transportations_library import BasicFreeways

from hcm_mcp_server.core.models import BasicFreewaysInput


def _guarded(step: float | None = None) -> Callable:
    """Wrap a function so any failure — including a Rust PanicException (off the heavy-vehicle PCE grid), which is not a Python Exception — returns a clean error dict."""
    def decorator(fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable:
        @wraps(fn)
        def wrapper(data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                return fn(data)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:  # noqa: BLE001
                err: Dict[str, Any] = {"success": False, "error": str(e)}
                if step is not None:
                    err["step"] = step
                return err
        return wrapper
    return decorator


def create_freeway_from_input(fw: BasicFreewaysInput) -> BasicFreeways:
    """Build a verified BasicFreeways segment from validated input."""
    return BasicFreeways(
        bffs=fw.bffs,
        lane_width=fw.lw,
        lane_count=fw.lane_count,
        lc_r=fw.lc_r,
        lc_l=fw.lc_l,
        trd=fw.trd,
        apd=fw.apd,
        grade=fw.grade,
        terrain_type=fw.terrain_type,
        speed_limit=fw.speed_limit,
        phf=fw.phf,
        p_t=fw.p_t,
        demand_flow_i=fw.demand_flow_i,
        length=fw.length,
        highway_type=fw.highway_type,
        city_type=fw.city_type,
    )


@_guarded(step=2)
def determine_free_flow_speed_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 2: estimate and adjust free-flow speed (mi/h)."""
    bf = create_freeway_from_input(BasicFreewaysInput(**data["freeway_data"]))
    ffs = bf.determine_free_flow_speed()
    return {"success": True, "step": 2, "free_flow_speed": ffs}


@_guarded(step=3)
def estimate_capacity_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 3: base and adjusted capacity (pc/h/ln)."""
    bf = create_freeway_from_input(BasicFreewaysInput(**data["freeway_data"]))
    bf.determine_free_flow_speed()
    capacity = bf.estimate_capacity()
    return {
        "success": True, "step": 3,
        "capacity": capacity, "adjusted_capacity": bf.adjusted_capacity(),
    }


@_guarded(step=4)
def estimate_demand_volume_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 4: convert directional demand to per-lane flow rate v_p (pc/h/ln)."""
    bf = create_freeway_from_input(BasicFreewaysInput(**data["freeway_data"]))
    bf.determine_free_flow_speed()
    bf.estimate_capacity()
    v_p = bf.estimate_demand_volume()
    return {"success": True, "step": 4, "flow_rate": v_p}


@_guarded(step=5)
def calculate_speed_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 5a: space mean speed via the speed-flow curve (mi/h)."""
    bf = create_freeway_from_input(BasicFreewaysInput(**data["freeway_data"]))
    bf.determine_free_flow_speed()
    bf.estimate_capacity()
    bf.estimate_demand_volume()
    bf.calculate_vc_ratio()
    speed = bf.calculate_speed()
    return {"success": True, "step": 5, "speed": speed, "vc_ratio": bf.vc_ratio()}


@_guarded(step=5)
def estimate_density_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 5b: density D = v_p / S (pc/mi/ln)."""
    bf = create_freeway_from_input(BasicFreewaysInput(**data["freeway_data"]))
    bf.determine_free_flow_speed()
    bf.estimate_capacity()
    bf.estimate_demand_volume()
    bf.calculate_vc_ratio()
    bf.calculate_speed()
    density = bf.estimate_density()
    return {"success": True, "step": 5, "density": density}


@_guarded(step=6)
def determine_segment_los_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 6: segment Level of Service (A-F)."""
    bf = create_freeway_from_input(BasicFreewaysInput(**data["freeway_data"]))
    los = bf.run_operational_analysis()
    return {
        "success": True, "step": 6,
        "level_of_service": los, "density": bf.density(), "vc_ratio": bf.vc_ratio(),
    }


@_guarded()
def complete_freeway_analysis_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Complete HCM Chapter 12 basic-freeway analysis (FFS -> capacity -> flow -> speed -> density -> LOS)."""
    fw = BasicFreewaysInput(**data["freeway_data"])
    bf = create_freeway_from_input(fw)
    los = bf.run_operational_analysis()
    return {
        "success": True,
        "analysis_type": "complete_hcm_chapter12",
        "results": {
            "inputs": {
                "bffs": fw.bffs, "lane_width": fw.lw, "lane_count": fw.lane_count,
                "grade": fw.grade, "demand_flow_i": fw.demand_flow_i,
                "heavy_vehicle_pct": fw.p_t,
            },
            "free_flow_speed": bf.ffs(),
            "capacity": bf.capacity(),
            "adjusted_capacity": bf.adjusted_capacity(),
            "speed": bf.speed(),
            "density": bf.density(),
            "vc_ratio": bf.vc_ratio(),
            "level_of_service": los,
        },
    }
