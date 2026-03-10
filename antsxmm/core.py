"""Public API facade and shared runtime dependency owner.

Historically, most implementation lived in this module. It is now decomposed into
smaller units under antsxmm/, but we keep these imports for backwards
compatibility (CLI/tests/third-party scripts).
"""

from .runtime_imports import import_optional_module

antspymm = import_optional_module("antspymm")
ants = import_optional_module("ants")

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



def process_session(*args, **kwargs):
    from .session import process_session as _process_session
    return _process_session(*args, **kwargs)

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
    "process_session",
    "antspymm",
    "ants",
]
