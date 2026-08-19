# Changelog

## 0.3.0

### The paper surface

The ten public tools (`analyze_facility`, `describe_facility_inputs`, `query_hcm`, the six `reasoning_*` tools and `validation_validate_design_full`) and the twenty-two legacy per-step tools (`chapter12_*`, `chapter15_*` and the three research extras loaded with `include_legacy`) are **the tool surface the published ESWA ablation experiment ran against**. This release does not change any of them: not a name, not a JSON schema, not a description, not a response shape. Everything below is additive.

`tests/data/paper_surface.json` records that surface and `tests/test_frozen_surface.py` compares the live registry against it on every run. A failure there means a tool the paper depends on has moved, and the fix is almost always to revert the change rather than to update the snapshot. `mcp_server_fastapi.py` is the `ct` ablation arm, so its `PUBLIC_OPERATIONS` default is asserted to be exactly those ten operations in exactly that order.

### Added: full HCM chapter coverage behind three capability tools

Every method the compute library implements, chapters 10 through 28, reachable through three tools. The method is an argument, not a tool: one tool per method would put thirty-two near-identical schemas into every caller's context and blunt tool selection, and the ten published tools are already capability-shaped.

- **`hcm_analyze`** `{method, config}` — runs one method. `method` is an enum of the thirty-two ids and the tool description carries the catalog, one compact line each.
- **`hcm_describe`** `{method?}` — with a method, its input schema sketch, result-field meanings and worked-example fixture; without one, the catalog of every method with its chapter and summary.
- **`hcm_validate`** `{method, config}` — parses and checks a config without running the analysis, returning the library's own validation errors or ok. Twenty-three methods have a real step between parse and compute (serde deserialisation, keyword-constructor range checks, and for Chapter 15 the Exhibit 15-8 parameter ranges through `tl.validate_input`). The other nine are single JSON entry points where parsing and computation are the same call; those report `valid: null` with the reason rather than running the analysis and calling the result a validation.

Advertised schema cost: **3 tools / ~4.9 KB**, against ~13.9 KB had each method been its own tool. For scale, the ten frozen paper tools advertise ~5.8 KB.

The thirty-two methods:

| Chapter | `method` | What it computes |
| --- | --- | --- |
| 10 | `analyze_freeway_facility` | Freeway facility, via the Chapter 25 engine |
| 10 | `analyze_managed_lanes` | Managed-lane freeway facility |
| 11 | `analyze_freeway_reliability` | Freeway travel-time reliability |
| 12 | `analyze_basic_freeway` | Basic freeway and multilane segment |
| 13 | `analyze_weaving` | Freeway weaving segment (HCM 7 and 7.1) |
| 14 | `analyze_merge_diverge` | Freeway merge and diverge segment (HCM 7 and 7.1) |
| 15 | `analyze_two_lane_highway` | Two-lane highway facility |
| 16 | `analyze_urban_facility` | Urban street facility |
| 17 | `analyze_urban_reliability` | Urban street travel-time reliability |
| 18 | `analyze_urban_segment` | Urban street segment, automobile mode |
| 18 | `analyze_pedestrian_segment` | Urban street segment, pedestrian mode |
| 18 | `analyze_bicycle_segment` | Urban street segment, bicycle mode |
| 18 | `analyze_transit_segment` | Urban street segment, transit mode |
| 19 | `analyze_signalized` | Signalized intersection, automobile mode |
| 19 | `analyze_signalized_pedestrian` | Signalized intersection, pedestrian mode |
| 19 | `analyze_signalized_bicycle` | Signalized intersection, bicycle mode |
| 19 | `analyze_two_stage_crossing` | Two-stage pedestrian crossing delay |
| 20 | `analyze_twsc` | Two-way STOP-controlled intersection, vehicular |
| 20 | `analyze_twsc_pedestrian` | TWSC and midblock crossing, pedestrian mode |
| 21 | `analyze_awsc` | All-way STOP-controlled intersection |
| 22 | `analyze_roundabout` | Roundabout |
| 23 | `analyze_ramp_terminal` | Interchange ramp terminals (Part B) |
| 23 | `analyze_alternative_intersection` | RCUT and MUT alternative intersections (Part C) |
| 23 | `analyze_displaced_left_turn` | Displaced left-turn intersection (Part C) |
| 24 | `analyze_pedestrian_walkway` | Exclusive pedestrian walkway or stairwell |
| 24 | `analyze_shared_use_path_pedestrian` | Shared-use path, pedestrian mode |
| 24 | `analyze_offstreet_bicycle` | Off-street path, bicycle mode |
| 25 | `analyze_composite_grade` | Mixed-flow model, composite grade |
| 25 | `analyze_planning_facility` | Planning-level freeway facility |
| 26 | `analyze_mixed_flow` | Mixed-flow model, single grade |
| 27 | `analyze_weaving_service_volumes` | Weaving segment service volumes |
| 28 | `analyze_ramp_service_volumes` | Merge and diverge service volumes |

Every method takes the compute library's own example-case (fixture) JSON as its `config`, not a second flattened schema invented for the MCP layer, so an example case can be handed over unmodified. Each ships the example-problem fixture that validates it under `hcm_mcp_server/data/examples/`, and `hcm_describe` serves that fixture together with a key sketch and the method's documentation so a caller never has to read the Rust bindings.

Each method also keeps a method-shaped REST route at `/analysis/hcm/<method-with-hyphens>`. Routes are not MCP tools, so a readable per-method URL stays available to direct API callers at no context cost.

Where the library exposes a JSON entry point the tool passes the config straight through. Where it exposes a keyword constructor (`WeavingSegment`, `RampSegment`, `BasicFreeways`, `TwoLaneHighways`) the tool maps the fixture schema onto it using the mapping from the library's own integration tests.

Domain refusals carry the library's message unchanged. An un-digitised mixed-flow grade, an off-domain specific-upgrade PCE and a malformed serde config all come back as `{"success": false, "error": "<the library's own words>"}`, because those messages describe what the published HCM data covers and rewording them would hide why an input was rejected.

### Added: an opt-in switch for the full MCP surface

`HCM_MCP_FULL_COVERAGE=true` appends the three capability tools to the MCP mount's default tool list.

It defaults **off**, and that is deliberate. `mcp_server_fastapi.py` is the `ct` arm of the ablation, and it reads the default surface with no `HCM_MCP_INCLUDE_OPS` override, so widening that default would change what the experiment measures rather than extend the server. With the flag off the new tools are still fully reachable over REST at `/analysis/hcm/*` and through `/tools/call`; only what the MCP mount advertises is gated.

`tests/test_frozen_surface.py` pins the addition to exactly `{hcm_analyze, hcm_describe, hcm_validate}`, so a later change cannot quietly reintroduce a tool-per-method surface.

### Corrections

`tests/test_analysis.py` asserted an interchange ETT of 52.8 s/veh for the Chapter 34 Example Problem 1 fixture. That figure predates the `transportations-library` 0.3.1 Chapter 23 corrections (interchange aggregate LOS from weighted ETT only, `d2` on lane-group capacity) and the engine has returned about 50.7 s/veh since. The library's own Chapter 23 test asserts 50.7 +-0.5 for the same fixture; the published Exhibit 34-16 figure is 52.4. The assertion is re-anchored to the library's value with the provenance recorded at the test. Nothing in the analysis changed; the pin was stale.

### Changed

- Minimum `transportations-library` raised from 0.2.0 to 0.3.6. Earlier releases do not expose `analyze_mixed_flow`, `analyze_composite_grade`, `analyze_twsc_pedestrian`, `ManagedLaneFacility`, `PlanningFacility`, `AlternativeIntersection` or `DisplacedLeftTurn` to Python at all.

### Known gaps

The Chapter 15 bicycle mode (`BicycleLOS`) is implemented in the Rust library and has its own worked-example fixture, but it is not exported through the PyO3 bindings, so it has no tool here. Closing that gap is a change to `transportations-library`, not to this repository.
