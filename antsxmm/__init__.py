from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("antsxmm")
    except PackageNotFoundError:
        __version__ = "0.0.0-unknown"

__all__ = [
    "__version__",
    "parse_antsxbids_layout",
    "build_wide_table_from_mmwide",
    "bind_mm_rows",
    "process_session",
    "run_study",
    "check_modality_order",
]


def __getattr__(name: str):
    if name == "parse_antsxbids_layout":
        from .bids import parse_antsxbids_layout

        return parse_antsxbids_layout
    if name in {"process_session", "bind_mm_rows", "check_modality_order", "build_wide_table_from_mmwide"}:
        from .core import (
            bind_mm_rows,
            build_wide_table_from_mmwide,
            check_modality_order,
            process_session,
        )

        return {
            "process_session": process_session,
            "bind_mm_rows": bind_mm_rows,
            "check_modality_order": check_modality_order,
            "build_wide_table_from_mmwide": build_wide_table_from_mmwide,
        }[name]
    if name == "run_study":
        from .pipeline import run_study

        return run_study

    if name in {"session", "pipeline", "environment", "core", "bids"}:
        from importlib import import_module

        return import_module(f".{name}", __name__)
    raise AttributeError(f"module 'antsxmm' has no attribute {name!r}")
