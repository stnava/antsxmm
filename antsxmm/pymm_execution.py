from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import pandas as pd

from .execution_plan import modality_from_path, build_execution_plan, plan_to_row


def generate_xmm_dataframe(session_data: dict[str, Any], output_root: str, project_id: str) -> pd.DataFrame:
    """Build a deterministic antsxmm execution dataframe for one session.

    The dataframe is derived from the antsxmm execution plan, making planning the
    sole authority for canonical output layout.
    """
    plan = build_execution_plan(session_data, output_root=output_root, project_id=project_id)
    return pd.DataFrame([plan_to_row(plan, output_root=output_root)])


def _images_for_modality(row: pd.Series, modality: str) -> list[str]:
    if modality == 'T1w':
        vals = [row.get('filename')]
    elif modality == 'T2Flair':
        vals = [row.get('flairid')]
    elif modality == 'perf':
        vals = [row.get('perfid')]
    elif modality == 'pet3d':
        vals = [row.get('pet3did')]
    elif modality == 'rsfMRI':
        vals = [row.get('rsfid1'), row.get('rsfid2'), row.get('rsfid3')]
    elif modality == 'DTI':
        vals = [row.get('dtid1'), row.get('dtid2'), row.get('dtid3')]
    elif modality == 'NM2DMT':
        vals = [row.get(f'nmid{i}') for i in range(1, 12)]
    else:
        vals = []
    out = []
    for v in vals:
        if v is None:
            continue
        sv = str(v)
        if not sv or sv.lower() == 'nan':
            continue
        out.append(sv)
    return out



from contextlib import contextmanager
from typing import Any


def _canonical_prefix_from_row(row: pd.Series, modality: str) -> str:
    output_root = row.get("outputdir")
    project_id = row.get("projectID")
    subject_id = row.get("subjectID")
    session_id = row.get("date")
    run_id = row.get("xmm_run") or row.get("imageID") or "run-01"

    missing = [
        name
        for name, value in (
            ("outputdir", output_root),
            ("projectID", project_id),
            ("subjectID", subject_id),
            ("date", session_id),
        )
        if value is None or str(value) == "" or str(value).lower() == "nan"
    ]
    if missing:
        raise KeyError(
            f"Missing deterministic prefix inputs for modality {modality}: {', '.join(missing)}"
        )

    out_dir = os.path.join(str(output_root), str(project_id), str(subject_id), str(session_id), str(modality), str(run_id))
    return os.path.join(
        out_dir,
        f"{project_id}+{subject_id}+{session_id}+{modality}+{run_id}",
    )


@contextmanager
def _patched_docsamson(antspymm_module: Any, study_df: pd.DataFrame):
    """
    Patch the exact docsamson symbol resolved by antspymm.mm_csv.

    The prior implementation only patched `antspymm_module.docsamson`, but
    mm_csv resolves `docsamson` from its own module globals. If mm_csv was
    imported from a submodule (e.g. antspymm.mm), patching the package
    attribute is insufficient and the legacy filename-derived layout still
    executes.
    """
    mm_csv_fn = getattr(antspymm_module, "mm_csv", None)
    if mm_csv_fn is None:
        raise ModuleNotFoundError("antspymm.mm_csv is required")

    mm_csv_globals = getattr(mm_csv_fn, "__globals__", None)
    can_patch_globals = isinstance(mm_csv_globals, dict)

    original_global_docsamson = mm_csv_globals.get("docsamson", None) if can_patch_globals else None
    had_module_attr = hasattr(antspymm_module, "docsamson")
    original_module_docsamson = getattr(antspymm_module, "docsamson", None)

    def deterministic_docsamson(
        locmod,
        studycsv,
        outputdir,
        projid,
        sid,
        dtid,
        mysep,
        t1iid=None,
        verbose=True,
    ):
        row = study_df.iloc[0]

        prefix_key = f"xmm_prefix_{locmod}"
        prefix = row.get(prefix_key)
        if prefix is None and locmod == "T1wHierarchical":
            prefix = row.get("xmm_prefix_T1wHierarchical")

        if prefix is None or str(prefix) == '' or str(prefix).lower() == 'nan':
            prefix = _canonical_prefix_from_row(row, locmod)

        images = _images_for_modality(row, locmod)

        if verbose:
            print(
                {
                    "modality": locmod,
                    "outprefix": prefix,
                    "images": images,
                    "patched_by": "antsxmm",
                }
            )

        return {
            "modality": locmod,
            "outprefix": prefix,
            "images": images,
        }

    # Critical: patch the symbol mm_csv actually resolves when that global namespace
    # is available. Mocks/builtins may not expose a writable __globals__ dict.
    if can_patch_globals:
        mm_csv_globals["docsamson"] = deterministic_docsamson

    # Also patch the package/module attribute for compatibility / introspection.
    setattr(antspymm_module, "docsamson", deterministic_docsamson)

    try:
        yield
    finally:
        if can_patch_globals:
            if original_global_docsamson is None:
                mm_csv_globals.pop("docsamson", None)
            else:
                mm_csv_globals["docsamson"] = original_global_docsamson

        if had_module_attr:
            setattr(antspymm_module, "docsamson", original_module_docsamson)
        else:
            try:
                delattr(antspymm_module, "docsamson")
            except AttributeError:
                pass

            
def run_xmm_mm_csv(study_df: pd.DataFrame, antspymm_module: Any, **mm_csv_kwargs: Any):
    if not hasattr(antspymm_module, 'mm_csv'):
        raise ModuleNotFoundError('antspymm.mm_csv is required to execute antsxmm processing')
    row = study_df.iloc[0]
    for key in [
        'xmm_prefix_T1w', 'xmm_prefix_T2Flair', 'xmm_prefix_perf', 'xmm_prefix_pet3d',
        'xmm_prefix_DTI', 'xmm_prefix_rsfMRI', 'xmm_prefix_NM2DMT', 'xmm_prefix_T1wHierarchical'
    ]:
        prefix = row.get(key)
        if isinstance(prefix, str) and prefix:
            os.makedirs(os.path.dirname(prefix), exist_ok=True)

    with _patched_docsamson(antspymm_module, study_df):
        return antspymm_module.mm_csv(study_df, **mm_csv_kwargs)
