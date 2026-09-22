"""Unified registry and dispatch interface for modality-specific visualizers."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModalityReport:
    """Standardized report payload returned by each modality visualizer.

    Attributes
    ----------
    name : str
        Canonical modality identifier (e.g. 'T1w', 'T1wHierarchical', 'T2Flair',
        'DTI', 'rsfMRI', 'perf', 'pet3d', 'NM2DMT').
    title : str
        Human-readable clinical modality title.
    status : str
        Execution status: 'success', 'partial', 'warning', or 'error'.
    kpis : list of dict
        Structured KPI card specifications with keys:
        'label', 'value', 'unit', 'status', 'tooltip'.
    html_content : str
        Complete HTML payload for the modality tab or panel.
    errors : list of str
        List of warning or error messages encountered during generation.
    """

    name: str
    title: str
    status: str = "success"
    kpis: list[dict[str, Any]] = field(default_factory=list)
    html_content: str = ""
    errors: list[str] = field(default_factory=list)


def find_run_dir(modality_dir: Path | str) -> Path:
    """Locate the active execution run directory within a modality path.

    Checks for standard 'run-01' or 'run-*' subdirectories. If none are found,
    returns the modality directory itself.
    """
    p = Path(modality_dir)
    if not p.is_dir():
        return p

    run01 = p / "run-01"
    if run01.is_dir():
        return run01

    runs = sorted([d for d in p.iterdir() if d.is_dir() and d.name.startswith("run-")])
    if runs:
        return runs[0]

    return p


def find_file(directory: Path | str, pattern: str) -> Path | None:
    """Robustly find a file by exact name, suffix, or glob pattern.

    Searches direct path first, then directory glob, then recursive glob.
    """
    d = Path(directory)
    if not d.is_dir():
        return None

    # 1. Exact match
    direct = d / pattern
    if direct.is_file():
        return direct

    # 2. Match with wildcard prefix in same dir
    matches = list(d.glob(pattern))
    if not matches and not pattern.startswith("*"):
        matches = list(d.glob(f"*{pattern}"))
    if matches:
        return sorted(matches)[0]

    # 3. Recursive match if nested
    rmatches = list(d.rglob(pattern))
    if not rmatches and not pattern.startswith("*"):
        rmatches = list(d.rglob(f"*{pattern}"))
    if rmatches:
        return sorted(rmatches)[0]

    return None


def load_nifti_data(
    path_or_array: Any,
    canonical: bool = True,
) -> np.ndarray | None:
    """Load a 3D or 4D NIfTI file or array, standardizing orientation to RAS.

    Returns None gracefully if the file cannot be loaded or is invalid.
    """
    if path_or_array is None:
        return None

    if isinstance(path_or_array, np.ndarray):
        return path_or_array.astype(np.float32, copy=False)

    if hasattr(path_or_array, "numpy"):
        return np.asarray(path_or_array.numpy(), dtype=np.float32)

    try:
        p = Path(path_or_array)
        if not p.is_file():
            return None
        img = nib.load(str(p))
        if canonical:
            img = nib.as_closest_canonical(img)
        arr = img.get_fdata(dtype=np.float32)
        # Squeeze leading or trailing singleton dimensions
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        elif arr.ndim == 4 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        return arr
    except Exception as exc:
        logger.warning("Failed to load NIfTI from %s: %s", path_or_array, exc)
        return None


def safe_read_csv(path: Path | str | None) -> pd.DataFrame | None:
    """Safely load a CSV file, returning None on missing file or read error."""
    if path is None:
        return None
    try:
        p = Path(path)
        if not p.is_file():
            return None
        return pd.read_csv(p)
    except Exception as exc:
        logger.warning("Failed to read CSV %s: %s", path, exc)
        return None


def coalesce_multi_row_df(df: pd.DataFrame | None) -> pd.Series:
    """Coalesce multi-row summary CSV (Row 0 global, Row 1 regional) into single Series."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    if len(df) == 1:
        return df.iloc[0]
    return df.iloc[0].combine_first(df.iloc[1])


# Canonical modalities list
AVAILABLE_MODALITIES = [
    "T1w",
    "T1wHierarchical",
    "T2Flair",
    "DTI",
    "rsfMRI",
    "perf",
    "pet3d",
    "NM2DMT",
]

# Alias resolution mapping
MODALITY_ALIASES: dict[str, str] = {
    "t1": "T1w",
    "t1w": "T1w",
    "structural": "T1w",
    "t1whierarchical": "T1wHierarchical",
    "hierarchical": "T1wHierarchical",
    "t2flair": "T2Flair",
    "t2": "T2Flair",
    "flair": "T2Flair",
    "wmh": "T2Flair",
    "dti": "DTI",
    "diffusion": "DTI",
    "dwi": "DTI",
    "rsfmri": "rsfMRI",
    "fmri": "rsfMRI",
    "functional": "rsfMRI",
    "bold": "rsfMRI",
    "perf": "perf",
    "perfusion": "perf",
    "asl": "perf",
    "pcasl": "perf",
    "pet": "pet3d",
    "pet3d": "pet3d",
    "nm": "NM2DMT",
    "nm2dmt": "NM2DMT",
    "neuromelanin": "NM2DMT",
}


def list_available_modalities() -> list[str]:
    """Return list of canonical supported modality names."""
    return list(AVAILABLE_MODALITIES)


def get_modality_visualizer(name: str) -> Callable[..., ModalityReport]:
    """Retrieve the visualizer callable for a given modality name or alias."""
    norm_name = name.strip()
    canon_name = MODALITY_ALIASES.get(norm_name.lower(), norm_name)

    if canon_name in {"T1w", "T1wHierarchical"}:
        from antsxmm.visualize.modalities.structural import visualize_structural
        return visualize_structural
    elif canon_name == "T2Flair":
        from antsxmm.visualize.modalities.wmh import visualize_wmh
        return visualize_wmh
    elif canon_name == "DTI":
        from antsxmm.visualize.modalities.dti import visualize_dti
        return visualize_dti
    elif canon_name == "rsfMRI":
        from antsxmm.visualize.modalities.fmri import visualize_fmri
        return visualize_fmri
    elif canon_name == "perf":
        from antsxmm.visualize.modalities.perfusion import visualize_perfusion
        return visualize_perfusion
    elif canon_name == "pet3d":
        from antsxmm.visualize.modalities.pet import visualize_pet
        return visualize_pet
    elif canon_name == "NM2DMT":
        from antsxmm.visualize.modalities.neuromelanin import visualize_neuromelanin
        return visualize_neuromelanin
    else:
        raise ValueError(
            f"Unsupported modality '{name}'. Available: {AVAILABLE_MODALITIES}"
        )


def visualize_modality(
    name_or_dir: str | Path,
    modality_dir: Path | str | None = None,
    session_dir: Path | str | None = None,
    **kwargs: Any,
) -> ModalityReport:
    """Dispatch visualization to the appropriate modality visualizer.

    Supports two calling conventions:
    1. visualize_modality(name, modality_dir, session_dir=None, **kwargs)
    2. visualize_modality(modality_dir, session_dir=None, **kwargs) [auto-detects modality]
    """
    if modality_dir is None:
        p = Path(name_or_dir)
        # Attempt auto-detection based on directory name or can_visualize
        detected_name: str | None = None
        dir_name = p.name.lower()
        if dir_name in MODALITY_ALIASES:
            detected_name = MODALITY_ALIASES[dir_name]
        else:
            # Check can_visualize across visualizers
            from antsxmm.visualize.modalities import (
                dti,
                fmri,
                neuromelanin,
                perfusion,
                pet,
                structural,
                wmh,
            )

            for mod_name, mod_module in [
                ("T1wHierarchical", structural),
                ("T1w", structural),
                ("T2Flair", wmh),
                ("DTI", dti),
                ("rsfMRI", fmri),
                ("perf", perfusion),
                ("pet3d", pet),
                ("NM2DMT", neuromelanin),
            ]:
                if hasattr(mod_module, "can_visualize") and mod_module.can_visualize(p):
                    detected_name = mod_name
                    break

        if detected_name is None:
            detected_name = "T1w"

        visualizer = get_modality_visualizer(detected_name)
        return visualizer(p, session_dir=session_dir, **kwargs)
    else:
        visualizer = get_modality_visualizer(str(name_or_dir))
        return visualizer(Path(modality_dir), session_dir=session_dir, **kwargs)


__all__ = [
    "AVAILABLE_MODALITIES",
    "MODALITY_ALIASES",
    "ModalityReport",
    "coalesce_multi_row_df",
    "find_file",
    "find_run_dir",
    "get_modality_visualizer",
    "list_available_modalities",
    "load_nifti_data",
    "safe_read_csv",
    "visualize_modality",
]
