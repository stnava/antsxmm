"""Public API facade.

Historically, most implementation lived in this module. It is now decomposed into
smaller units under antsxmm/, but we keep these imports for backwards
compatibility (CLI/tests/third-party scripts).
"""

from .inputs import (
    plan_session_inputs,
    _extract_run_id_from_filename,
    _is_nifti,
    _as_path_list,
    _collect_discovered_inputs,
    _sidecar_paths_for_nifti,
)
from .fingerprint import compute_input_fingerprint
from .status import write_session_status
from .staging import extract_image_id, get_modality_variant, sanitize_and_stage_file
from .wide_table import bind_mm_rows, check_modality_order, build_wide_table_from_mmwide
from .session import print_expected_tree, process_session, antspymm, ants
from .run_id import normalize_run_id
from .bids_entities import parse_entities
from .pymm_execution import generate_xmm_dataframe, run_xmm_mm_csv
from .execution_plan import ExecutionUnit, build_execution_plan, validate_execution_plan

__all__ = [
    "plan_session_inputs",
    "compute_input_fingerprint",
    "write_session_status",
    "extract_image_id",
    "get_modality_variant",
    "sanitize_and_stage_file",
    "bind_mm_rows",
    "check_modality_order",
    "build_wide_table_from_mmwide",
    "print_expected_tree",
    "process_session",
    "normalize_run_id",
    "parse_entities",
    "generate_xmm_dataframe",
    "run_xmm_mm_csv",
    "ExecutionUnit",
    "build_execution_plan",
    "validate_execution_plan",
    "antspymm",
    "ants",
]
