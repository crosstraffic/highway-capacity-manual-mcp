from typing import Dict, Any
from transportations_library import SubSegment, Segment, TwoLaneHighways
from hcm_mcp_server.core.models import TwoLaneHighwaysInput
from hcm_mcp_server.core.validation import validate_input


def create_highway_from_input(highway_input: TwoLaneHighwaysInput) -> TwoLaneHighways:
    """Helper function to create highway object from input data."""
    py_segments = []
    for seg in highway_input.segments:
        subsegments = [
            SubSegment(
                length=sub.length,
                avg_speed=sub.avg_speed,
                hor_class=sub.hor_class,
                design_rad=sub.design_rad,
                central_angle=sub.central_angle,
                sup_ele=sub.sup_ele
            ) for sub in seg.subsegments
        ]
        
        py_segments.append(Segment(
            passing_type=seg.passing_type,
            length=seg.length,
            grade=seg.grade,
            spl=seg.spl,
            is_hc=seg.is_hc,
            volume=seg.volume,
            volume_op=seg.volume_op,
            flow_rate=seg.flow_rate,
            flow_rate_o=seg.flow_rate_o,
            capacity=seg.capacity,
            ffs=seg.ffs,
            avg_speed=seg.avg_speed,
            vertical_class=seg.vertical_class,
            subsegments=subsegments,
            phf=seg.phf,
            phv=seg.phv,
            pf=seg.pf,
            fd=seg.fd,
            fd_mid=seg.fd_mid,
            hor_class=seg.hor_class,
        ))

    return TwoLaneHighways(
        segments=py_segments,
        lane_width=highway_input.lane_width,
        shoulder_width=highway_input.shoulder_width,
        apd=highway_input.apd,
        pmhvfl=highway_input.pmhvfl,
        l_de=highway_input.l_de
    )


def identify_vertical_class_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 1: Identify vertical class range for a segment."""
    try:
        # Validate input
        validation_result = validate_input(data["highway_data"])
        if not validation_result["success"]:
            return validation_result

        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        vertical_class_range = highway.identify_vertical_class(segment_index)
        
        return {
            "success": True,
            "step": 1,
            "segment_index": segment_index,
            "vertical_class_range": {
                "min": vertical_class_range[0],
                "max": vertical_class_range[1]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 1}


def determine_demand_flow_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 2: Determine demand flow rates and capacity."""
    try:
        # Validate input
        validation_result = validate_input(data["highway_data"])
        if not validation_result["success"]:
            return validation_result

        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        demand_flow_results = highway.determine_demand_flow(segment_index)
        
        return {
            "success": True,
            "step": 2,
            "segment_index": segment_index,
            "demand_flow_inbound": demand_flow_results[0],
            "demand_flow_outbound": demand_flow_results[1],
            "capacity": demand_flow_results[2]
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 2}


def determine_vertical_alignment_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 3: Determine vertical alignment classification."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        vertical_alignment = highway.determine_vertical_alignment(segment_index)
        
        return {
            "success": True,
            "step": 3,
            "segment_index": segment_index,
            "vertical_alignment": vertical_alignment
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 3}


def determine_free_flow_speed_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 4: Calculate free flow speed."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        free_flow_speed = highway.determine_free_flow_speed(segment_index)
        
        return {
            "success": True,
            "step": 4,
            "segment_index": segment_index,
            "free_flow_speed": free_flow_speed
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 4}


def estimate_average_speed_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 5: Estimate average speed."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        
        avg_speed_results = highway.estimate_average_speed(segment_index)
        
        return {
            "success": True,
            "step": 5,
            "segment_index": segment_index,
            "average_speed": avg_speed_results[0],
            "horizontal_class": avg_speed_results[1]
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 5}


def estimate_percent_followers_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 6: Estimate percent followers."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        
        percent_followers = highway.estimate_percent_followers(segment_index)
        
        return {
            "success": True,
            "step": 6,
            "segment_index": segment_index,
            "percent_followers": percent_followers
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 6}


def determine_follower_density_pl_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 8a: Calculate follower density for passing lane segments."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        highway.estimate_average_speed(segment_index)
        highway.estimate_percent_followers(segment_index)
        
        follower_density_results = highway.determine_follower_density_pl(segment_index)
        
        return {
            "success": True,
            "step": 8,
            "subtype": "passing_lane",
            "segment_index": segment_index,
            "follower_density": follower_density_results[0],
            "follower_density_mid": follower_density_results[1]
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 8}


def determine_follower_density_pc_pz_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 8b: Calculate follower density for PC/PZ segments."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        highway.estimate_average_speed(segment_index)
        highway.estimate_percent_followers(segment_index)
        
        # Gotta need to think about adjustment
        follower_density = highway.determine_follower_density_pc_pz(segment_index)
        
        return {
            "success": True,
            "step": 8,
            "subtype": "pc_pz",
            "segment_index": segment_index,
            "follower_density": follower_density
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 8}


def determine_adjustment_to_follower_density_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 8.5: Calculate adjustment to follower density."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        highway.estimate_average_speed(segment_index)
        highway.estimate_percent_followers(segment_index)
        
        highway.determine_follower_density_pc_pz(segment_index)
        adjustment = highway.determine_adjustment_to_follower_density(segment_index)
        
        return {
            "success": True,
            "step": 8.5,
            "segment_index": segment_index,
            "follower_density_adjustment": adjustment
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 8.5}


def determine_segment_los_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 9: Determine segment Level of Service."""
    try:
        segment_index = data["segment_index"]
        s_pl = data["s_pl"]
        capacity = data["capacity"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        segment_los = highway.determine_segment_los(segment_index, s_pl, capacity)
        
        return {
            "success": True,
            "step": 9,
            "segment_index": segment_index,
            "level_of_service": segment_los,
            "average_speed": s_pl,
            "capacity": capacity
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 9}


def determine_facility_los_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 10: Determine facility Level of Service."""
    try:
        # Validate input
        validation_result = validate_input(data["highway_data"])
        if not validation_result["success"]:
            return validation_result

        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run complete analysis for all segments
        total_length = 0.0
        weighted_fd = 0.0
        weighted_speed = 0.0
        
        for seg_idx in range(len(highway_input.segments)):
            # Run all prerequisite steps
            highway.determine_demand_flow(seg_idx)
            highway.determine_free_flow_speed(seg_idx)
            avg_speed_results = highway.estimate_average_speed(seg_idx)
            highway.estimate_percent_followers(seg_idx)
            
            segment_length = highway_input.segments[seg_idx].length
            total_length += segment_length
            
            # Calculate follower density based on segment type
            if highway_input.segments[seg_idx].passing_type == 2: # Passing lane
                fd_results = highway.determine_follower_density_pl(seg_idx)
                fd_value = fd_results[1] # Use fd_mid for passing lanes
            else: # PC or PZ
                fd_value = highway.determine_follower_density_pc_pz(seg_idx)
                fd_adj = highway.determine_adjustment_to_follower_density(seg_idx)
                if fd_adj > 0.0: # Replace to adjustment if positive
                    fd_value = fd_adj
            
            weighted_fd += fd_value * segment_length
            weighted_speed += avg_speed_results[0] * segment_length
        
        facility_fd = weighted_fd / total_length
        facility_speed = weighted_speed / total_length
        facility_los = highway.determine_facility_los(facility_fd, facility_speed)
        
        return {
            "success": True,
            "step": 10,
            "facility_follower_density": facility_fd,
            "facility_average_speed": facility_speed,
            "facility_level_of_service": facility_los,
            "total_length": total_length
        }
    except Exception as e:
        return {"success": False, "error": str(e), "step": 10}


def complete_highway_analysis_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform complete HCM Chapter 15 analysis following standard procedure."""
    try:
        # Validate input before processing
        validation_result = validate_input(data["highway_data"])
        if not validation_result["success"]:
            return validation_result

        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        analysis_results = {
            "facility_info": {
                "total_length": sum(seg.length for seg in highway_input.segments),
                "num_segments": len(highway_input.segments),
                "lane_width": highway_input.lane_width,
                "shoulder_width": highway_input.shoulder_width,
                "apd": highway_input.apd
            },
            "segments": {}
        }
        
        # Analyze each segment following HCM procedure
        for seg_idx in range(len(highway_input.segments)):
            segment_data = highway_input.segments[seg_idx]
            seg_results = {
                "basic_info": {
                    "passing_type": segment_data.passing_type,
                    "length": segment_data.length,
                    "grade": segment_data.grade,
                    "speed_limit": segment_data.spl,
                    "volume": segment_data.volume,
                    "volume_opposite": segment_data.volume_op
                }
            }
            
            # Run all HCM steps in sequence
            steps = [
                ("step_1", lambda: highway.identify_vertical_class(seg_idx)),
                ("step_2", lambda: highway.determine_demand_flow(seg_idx)),
                ("step_3", lambda: highway.determine_vertical_alignment(seg_idx)),
                ("step_4", lambda: highway.determine_free_flow_speed(seg_idx)),
                ("step_5", lambda: highway.estimate_average_speed(seg_idx)),
                ("step_6", lambda: highway.estimate_percent_followers(seg_idx))
            ]
            
            for step_name, step_func in steps:
                try:
                    result = step_func()
                    seg_results[step_name] = result
                except Exception as step_error:
                    seg_results[step_name] = {"error": str(step_error)}
            
            # Step 8: Follower density (depends on segment type)
            try:
                if segment_data.passing_type == 2:  # Passing lane
                    fd_results = highway.determine_follower_density_pl(seg_idx)
                    seg_results["step_8"] = {
                        "type": "passing_lane",
                        "fd": fd_results[0],
                        "fd_mid": fd_results[1]
                    }
                    # fd_for_los = fd_results[1]
                else:  # PC or PZ
                    fd_value = highway.determine_follower_density_pc_pz(seg_idx)
                    seg_results["step_8"] = {
                        "type": "pc_or_pz",
                        "fd": fd_value
                    }

                    # Step 8.5: Adjustment to follower density
                    fd_adj = highway.determine_adjustment_to_follower_density(seg_idx)
                    if fd_adj > 0.0: # Replace to adjustment if positive
                        fd_value = fd_adj

                        seg_results["step_8.5"] = {
                            "type": "fd_adjustment",
                            "fd": fd_adj,
                            "fd_adj": fd_adj
                        }
                    # fd_for_los = fd_value

            except Exception as fd_error:
                seg_results["step_8"] = {"error": str(fd_error)}
                # fd_for_los = 0.0
            
            # Step 9: Segment LOS
            try:
                if "step_5" in seg_results and not isinstance(seg_results["step_5"], dict):
                    avg_speed = seg_results["step_5"][0] if isinstance(seg_results["step_5"], (list, tuple)) else seg_results["step_5"]
                    capacity = seg_results["step_2"][2] if "step_2" in seg_results else 1700
                    segment_los = highway.determine_segment_los(seg_idx, avg_speed, int(capacity))
                    seg_results["step_9"] = segment_los
                else:
                    seg_results["step_9"] = {"error": "Prerequisites not met"}
            except Exception as los_error:
                seg_results["step_9"] = {"error": str(los_error)}
            
            analysis_results["segments"][f"segment_{seg_idx}"] = seg_results
        
        # Step 10: Facility LOS
        try:
            facility_los_result = determine_facility_los_function({"highway_data": data["highway_data"]})
            if facility_los_result["success"]:
                analysis_results["facility_los"] = {
                    "follower_density": facility_los_result["facility_follower_density"],
                    "average_speed": facility_los_result["facility_average_speed"],
                    "level_of_service": facility_los_result["facility_level_of_service"]
                }
        except Exception as facility_error:
            analysis_results["facility_los"] = {"error": str(facility_error)}
        
        return {
            "success": True,
            "analysis_type": "complete_hcm_chapter15",
            "results": analysis_results
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "error_type": type(e).__name__}