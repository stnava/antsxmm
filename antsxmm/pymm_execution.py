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


@contextmanager
def _patched_docsamson(antspymm_module: Any, study_df: pd.DataFrame):
    original = getattr(antspymm_module, 'docsamson', None)

    def deterministic_docsamson(locmod, studycsv, outputdir, projid, sid, dtid, mysep, t1iid=None, verbose=True):
        row = study_df.iloc[0]
        prefix_key = f'xmm_prefix_{locmod}'
        prefix = row.get(prefix_key)
        if prefix is None and locmod == 'T1wHierarchical':
            prefix = row.get('xmm_prefix_T1wHierarchical')
        images = _images_for_modality(row, locmod)
        return {'modality': locmod, 'outprefix': prefix, 'images': images}

    setattr(antspymm_module, 'docsamson', deterministic_docsamson)
    try:
        yield
    finally:
        if original is not None:
            setattr(antspymm_module, 'docsamson', original)
        else:
            delattr(antspymm_module, 'docsamson')


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
