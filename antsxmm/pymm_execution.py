from __future__ import annotations

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Any

import pandas as pd

from .bids_entities import parse_entities
from .run_id import normalize_run_id
from .inputs import _as_path_list

_SUFFIX_TO_MODALITY = {
    'T1w': 'T1w',
    'FLAIR': 'T2Flair',
    'T2w': 'T2Flair',
    'dwi': 'DTI',
    'bold': 'rsfMRI',
    'asl': 'perf',
}


def modality_from_path(path: str) -> str | None:
    entities = parse_entities(path)
    return _SUFFIX_TO_MODALITY.get(entities.get('suffix', ''))


def _canonical_output_prefix(output_root: str, project_id: str, subject: str, session: str, modality: str, run: str) -> str:
    out_dir = Path(output_root) / project_id / subject / session / modality / run
    prefix = f"{project_id}+{subject}+{session}+{modality}+{run}"
    return str(out_dir / prefix)


def _group_runs(paths: list[str], fallback_run: str = 'run-01') -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for path in sorted(paths):
        entities = parse_entities(path)
        run = normalize_run_id(entities.get('run') or fallback_run)
        grouped.setdefault(run, []).append(path)
    return grouped


def _first_or_none(paths: list[str]) -> str | None:
    return paths[0] if paths else None


def generate_xmm_dataframe(session_data: dict[str, Any], output_root: str, project_id: str) -> pd.DataFrame:
    """Build a deterministic antsxmm execution dataframe for one session.

    The returned dataframe carries both a minimal antspymm-compatible row and
    antsxmm-owned canonical prefix metadata used during execution.
    """
    subject = str(session_data.get('subjectID'))
    session = str(session_data.get('date') or session_data.get('sessionID'))

    t1_paths = _as_path_list(session_data.get('t1_filenames'))
    if not t1_paths:
        t1_single = session_data.get('t1_filename')
        if t1_single:
            t1_paths = [str(t1_single)]
    if not t1_paths:
        raise ValueError('generate_xmm_dataframe requires at least one T1w input')

    t1_path = sorted(t1_paths)[0]
    t1_entities = parse_entities(t1_path)
    canonical_run = normalize_run_id(t1_entities.get('run'))

    flair_candidates = _as_path_list(session_data.get('flair_filenames')) or _as_path_list(session_data.get('t2w_filenames'))
    flair_path = _first_or_none(sorted(flair_candidates))

    rsf_paths = sorted(_as_path_list(session_data.get('rsf_filenames')))[:2]
    dti_paths = sorted(_as_path_list(session_data.get('dti_filenames')))[:2]
    nm_paths = sorted(_as_path_list(session_data.get('nm_filenames')))[:11]

    perf_paths = _as_path_list(session_data.get('perf_filenames'))
    perf_single = session_data.get('perf_filename')
    if perf_single and str(perf_single) not in perf_paths:
        perf_paths = [str(perf_single)] + perf_paths
    perf_path = _first_or_none(sorted(perf_paths))

    pet_paths = _as_path_list(session_data.get('pet3d_filenames'))
    pet_single = session_data.get('pet3d_filename')
    if pet_single and str(pet_single) not in pet_paths:
        pet_paths = [str(pet_single)] + pet_paths
    pet_path = _first_or_none(sorted(pet_paths))

    prefixes = {
        'T1w': _canonical_output_prefix(output_root, project_id, subject, session, 'T1w', canonical_run),
        'T2Flair': _canonical_output_prefix(output_root, project_id, subject, session, 'T2Flair', canonical_run),
        'perf': _canonical_output_prefix(output_root, project_id, subject, session, 'perf', canonical_run),
        'pet3d': _canonical_output_prefix(output_root, project_id, subject, session, 'pet3d', canonical_run),
        'DTI': _canonical_output_prefix(output_root, project_id, subject, session, 'DTI', canonical_run),
        'rsfMRI': _canonical_output_prefix(output_root, project_id, subject, session, 'rsfMRI', canonical_run),
        'NM2DMT': _canonical_output_prefix(output_root, project_id, subject, session, 'NM2DMT', canonical_run),
        'T1wHierarchical': _canonical_output_prefix(output_root, project_id, subject, session, 'T1wHierarchical', canonical_run),
    }

    row = {
        'projectID': project_id,
        'subjectID': subject,
        'date': session,
        'imageID': canonical_run,
        'modality': 'T1w',
        'sourcedir': str(session_data.get('session_path') or Path(t1_path).parent),
        'outputdir': output_root,
        'filename': t1_path,
        'flairid': flair_path,
        'perfid': perf_path,
        'pet3did': pet_path,
        'rsfid1': rsf_paths[0] if len(rsf_paths) > 0 else None,
        'rsfid2': rsf_paths[1] if len(rsf_paths) > 1 else None,
        'rsfid3': None,
        'dtid1': dti_paths[0] if len(dti_paths) > 0 else None,
        'dtid2': dti_paths[1] if len(dti_paths) > 1 else None,
        'dtid3': None,
        'nmid1': nm_paths[0] if len(nm_paths) > 0 else None,
        'nmid2': nm_paths[1] if len(nm_paths) > 1 else None,
        'nmid3': nm_paths[2] if len(nm_paths) > 2 else None,
        'nmid4': nm_paths[3] if len(nm_paths) > 3 else None,
        'nmid5': nm_paths[4] if len(nm_paths) > 4 else None,
        'nmid6': nm_paths[5] if len(nm_paths) > 5 else None,
        'nmid7': nm_paths[6] if len(nm_paths) > 6 else None,
        'nmid8': nm_paths[7] if len(nm_paths) > 7 else None,
        'nmid9': nm_paths[8] if len(nm_paths) > 8 else None,
        'nmid10': nm_paths[9] if len(nm_paths) > 9 else None,
        'nmid11': nm_paths[10] if len(nm_paths) > 10 else None,
        'xmm_run': canonical_run,
        'xmm_prefix_T1w': prefixes['T1w'],
        'xmm_prefix_T2Flair': prefixes['T2Flair'],
        'xmm_prefix_perf': prefixes['perf'],
        'xmm_prefix_pet3d': prefixes['pet3d'],
        'xmm_prefix_DTI': prefixes['DTI'],
        'xmm_prefix_rsfMRI': prefixes['rsfMRI'],
        'xmm_prefix_NM2DMT': prefixes['NM2DMT'],
        'xmm_prefix_T1wHierarchical': prefixes['T1wHierarchical'],
    }
    return pd.DataFrame([row])


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
